#!/usr/bin/env python3
"""Offline closed-loop bench for DualArmTactileController -- no Gazebo, no PX4, no TacTip.

A fake plant pretends to be the drone, the arms and the two TacTips, so the REAL controller node
runs against it unmodified: it integrates the commanded joint velocities, uses the real URDF
kinematics for how deep each pad is in the object, turns that into a pad force, rotates it back
into the sensor frame through the real transforms, and publishes it on the real topics. A whole
grasp takes about a minute instead of a Gazebo session.

VALIDATES the control logic, sign conventions, frame transforms, safety layer and state machine.
DOES NOT validate the sensor, the force model, the pad stiffness or the friction -- PAD_STIFFNESS
and the friction are assumptions written here, so every number it prints is a round trip through
them. Good for catching a broken sign or a deadlocked loop; not an experimental result.

USAGE (needs the workspace sourced):
    python3 bench_grasp_loop.py --all            every case, pass/fail summary
    python3 bench_grasp_loop.py --mu 0.55        one case, phase-by-phase table
    python3 bench_grasp_loop.py --mu 1.0 --sim [--fake-tactile]

CASES
    mu 1.00  good friction        -> grip settles, object held, mass estimate valid
    mu 0.55  marginal friction    -> slip detected, mu_est drops, grip rises, object held
    mu 0.30  too slippery         -> fault raised, mass published but flagged invalid
    sim      fake tactile data    -> closes and squeezes on joint stall, as Gazebo does
    sim + fake-tactile            -> as above, but the pose model publishes what the REAL sim
                                     driver publishes: a CONSTANT 3.0 mm depth, whatever the arms
                                     are doing. That constant tripped the depth limit and made the
                                     arms back off through every grasp, so a Gazebo mission
                                     'completed' without ever touching the object while all four
                                     other cases here passed. This case exists so that cannot recur.
"""

import argparse
import subprocess
import sys
import time

import numpy as np
import rclpy
from geometry_msgs.msg import TwistStamped, WrenchStamped
from px4_msgs.msg import VehicleOdometry
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64, Int8, Int32

from pose_based_ats import arm_kinematics as ak
from pose_based_ats.dual_arm_tactile_controller import DualArmTactileController

OBJECT_WIDTH_M = 0.10
OBJECT_WEIGHT_N = 1.48        # 150 g
PAD_STIFFNESS_N_PER_M = 600.0
WEIGHT_TRANSFER_S = 1.5       # how long the drone takes to put the load on the pads
OPEN_POSE = [4.0435, 0.0, -1.95, -4.0435, 0.0, 1.95]

# Phase numbers, kept in step with the controller and the mission.
PRE_GRASP, CLOSE, SQUEEZE, LIFT, CARRY, PLACE, RELEASE = 30, 31, 32, 33, 34, 35, 36


class Plant(Node):
    """Stands in for the drone, the arms and both sensors."""

    def __init__(self, mu_true, hard_stall, fake_tactile=False):
        super().__init__('bench_plant')
        self.mu_true = mu_true
        self.hard_stall = hard_stall        # emulate Gazebo, where the object blocks the joint
        self.fake_tactile = fake_tactile    # publish what the sim driver publishes, not the truth
        self.q = np.array(OPEN_POSE, dtype=float)
        self.cmd = np.zeros(6)
        self.phase = 0
        self.shear = 0.0
        self.lift_t = 0.0
        self.status, self.width, self.n_target = -1, float('nan'), 0.0
        self.mass, self.mass_valid, self.turning = float('nan'), False, 0.0

        self.create_subscription(JointState, '/controller/out/servo_state', self._cb_cmd, 10)
        self._watch('/grasp/status', 'status', Int8)
        self._watch('/grasp/measured_width', 'width')
        self._watch('/grasp/normal_target', 'n_target')
        self._watch('/grasp/estimated_mass', 'mass')
        self._watch('/grasp/estimated_mass_valid', 'mass_valid', Bool)
        self._watch('/grasp/shear_turning', 'turning')

        self.pub_joints = self.create_publisher(JointState, '/servo/out/state', 10)
        self.pub_phase = self.create_publisher(Int32, '/md/grasp_phase', 10)
        self.pub_odom = self.create_publisher(VehicleOdometry,
                                              '/fmu/in/vehicle_visual_odometry', 10)
        self.pub_force = {1: self.create_publisher(WrenchStamped, '/tactip_left/force', 10),
                          2: self.create_publisher(WrenchStamped, '/tactip_right/force', 10)}
        self.pub_pose = {1: self.create_publisher(TwistStamped, '/tactip_left/pose', 10),
                         2: self.create_publisher(TwistStamped, '/tactip_right/pose', 10)}
        self.create_timer(0.01, self.step)

    def _watch(self, topic, attr, msg_type=Float64):
        self.create_subscription(msg_type, topic,
                                 lambda m, a=attr: setattr(self, a, m.data), 10)

    def _cb_cmd(self, msg):
        self.cmd = np.array(msg.velocity[:6])

    def indentation(self):
        """How far each pad is into the object, plus the two arms' joint triples."""
        q1, q2 = tuple(self.q[0:3]), tuple(self.q[3:6])
        gap = ak.grasp_width_m(q1, q2)
        return max(0.0, (OBJECT_WIDTH_M - gap) / 2.0), q1, q2

    def step(self):
        """One plant tick: move the joints, work out the contact, publish what the pads see."""
        move = self.cmd * 0.01
        depth, _, _ = self.indentation()
        if self.hard_stall and depth > 0.0015:
            for idx, closing_sign in ((0, -1.0), (3, +1.0)):   # closing lowers |shoulder|
                if move[idx] * closing_sign > 0:
                    move[idx] = 0.0
        self.q += move

        depth, q1, q2 = self.indentation()
        normal = PAD_STIFFNESS_N_PER_M * depth
        tangential = self._contact_load(normal)

        now = self.get_clock().now().to_msg()
        joints = JointState()
        joints.header.stamp = now
        joints.name = [f'q{i + 1}' for i in range(6)]
        joints.position = [float(v) for v in self.q]
        self.pub_joints.publish(joints)
        self.pub_phase.publish(Int32(data=self.phase))

        odom = VehicleOdometry()
        odom.q = [1.0, 0.0, 0.0, 0.0]          # level hover
        odom.velocity = [0.0, 0.0, 0.0]
        self.pub_odom.publish(odom)

        R_wb = ak.R_world_from_body(odom.q)
        for arm, joints_arm in ((1, q1), (2, q2)):
            press = R_wb @ ak.press_direction_body(arm, q1, q2)
            f_world = normal * press + tangential * np.array([0.0, 0.0, 1.0])
            # Back into the sensor's own frame, using the same mounting convention as the rig.
            R_ws = R_wb @ ak.R_body_from_sensor(arm, *joints_arm,
                                                mark_along_plus_z=ak.mark_plus_z(arm, True))
            f_sensor = R_ws.T @ f_world

            wrench = WrenchStamped()
            wrench.header.stamp = now
            (wrench.wrench.force.x, wrench.wrench.force.y,
             wrench.wrench.force.z) = map(float, f_sensor)
            self.pub_force[arm].publish(wrench)

            pose = TwistStamped()
            pose.header.stamp = now
            if self.fake_tactile:
                # Exactly what tactip_ros2_driver.publish_fake_data() sends: a constant depth and
                # no tilt or shear at all, whatever the arms are actually doing.
                pose.twist.linear.z = -3.0
            else:
                pose.twist.linear.z = float(-depth * 1000.0)
                pose.twist.linear.x = float(self.shear)
                pose.twist.angular.x = 24.9    # the contact angle at the nominal pose
            self.pub_pose[arm].publish(pose)

    def _contact_load(self, normal):
        """What each pad carries along the face, and what that does to its shear displacement.

        Friction caps the load at mu_true * normal. Past that the object slides: the membrane
        breaks away and re-centres, which is the signal the controller's slip detector looks for.
        """
        if self.phase not in (LIFT, CARRY) or normal <= 0.05:
            self.lift_t = 0.0
            self.shear += (0.0 - self.shear) * 0.3
            return 0.0
        self.lift_t += 0.01
        demand = (OBJECT_WEIGHT_N / 2.0) * min(1.0, self.lift_t / WEIGHT_TRANSFER_S)
        capacity = self.mu_true * normal
        load = min(demand, capacity)
        sliding = demand > capacity
        stuck_shear = min(1.2, load / 0.8)
        target = 0.25 * stuck_shear if sliding else stuck_shear
        self.shear += (target - self.shear) * (0.25 if sliding else 0.12)
        return load


def run_case(mu_true, sim, verbose=True, fake_tactile=False):
    """Drive one whole grasp and report what happened at each phase."""
    rclpy.init(args=['--ros-args',
                     '-p', f'sim:={"true" if sim else "false"}',
                     '-p', 'frequency:=100.0',
                     '-p', 'grasp.tube_m:=0.228',
                     '-p', 'grasp.forearm_working_rad:=1.95',
                     '-p', 'grasp.shoulder_open_rad:=4.0435',
                     '-p', 'grasp.shoulder_min_rad:=3.88',
                     '-p', 'limits.relative_backstop_rad:=0.0075',
                     '-p', 'grip.mu_initial:=1.0'])
    controller = DualArmTactileController()
    plant = Plant(mu_true, hard_stall=sim, fake_tactile=fake_tactile)
    executor = SingleThreadedExecutor()
    executor.add_node(controller)
    executor.add_node(plant)

    def hold(phase, seconds, label):
        plant.phase = phase
        deadline = time.time() + seconds
        while time.time() < deadline:
            executor.spin_once(timeout_sec=0.01)
        if verbose:
            depth, _, _ = plant.indentation()
            width = plant.width * 1000 if plant.width == plant.width else float('nan')
            print(f'  {label:10s} phase {phase}: |shoulder| {plant.q[0]:.4f}/{-plant.q[3]:.4f}, '
                  f'indent {depth * 1000:5.2f} mm, N {PAD_STIFFNESS_N_PER_M * depth:5.2f}, '
                  f'target {plant.n_target:.2f}, status {plant.status}, width {width:.1f} mm')

    hold(PRE_GRASP, 0.5, 'pre-grasp')
    hold(CLOSE, 8.0, 'close')
    hold(SQUEEZE, 6.0, 'squeeze')
    hold(LIFT, 14.0, 'lift')
    carry_status = None
    hold(CARRY, 12.0, 'carry')
    carry_status = plant.status
    hold(PLACE, 3.0, 'place')
    hold(RELEASE, 4.0, 'release')

    result = {'carry_status': carry_status, 'mass': plant.mass, 'mass_valid': plant.mass_valid,
              'width_mm': plant.width * 1000, 'mu_est': controller.mu_est}
    rclpy.shutdown()
    return result


CASES = [
    # (mu_true, sim, expected carry status, expect a trustworthy mass?)
    (1.00, False, 2, True),
    (0.55, False, 2, True),
    (0.30, False, 3, False),
    (1.00, True, 2, False),      # sim path publishes no mass at all
]
# Same as the last case but with the pose model faked the way the simulator fakes it. The grasp
# must still close: a constant 3 mm depth must not be allowed to drive the safety layer.
FAKE_CASE = (1.00, True, 2, False)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--mu', type=float, help='the plant\'s true friction coefficient')
    parser.add_argument('--sim', action='store_true', help="exercise the controller's sim path")
    parser.add_argument('--all', action='store_true', help='run every case and summarise')
    parser.add_argument('--fake-tactile', action='store_true',
                        help='publish the pose model the way the real sim driver fakes it')
    args = parser.parse_args()

    if not args.all:
        result = run_case(args.mu if args.mu is not None else 0.55, args.sim,
                          fake_tactile=args.fake_tactile)
        print(f'\n  mass {result["mass"] * 1000:.1f} g (true {OBJECT_WEIGHT_N / 9.80665 * 1000:.1f}'
              f'), valid={result["mass_valid"]}, mu_est {result["mu_est"]:.3f}')
        return 0

    # Each case needs its own rclpy context, so run them as separate processes.
    failures = 0
    for mu, sim, want_status, want_valid, fake in (
            [(a, b, c, d, False) for a, b, c, d in CASES] + [FAKE_CASE + (True,)]):
        cmd = ([sys.executable, __file__, '--mu', str(mu)] + (['--sim'] if sim else [])
               + (['--fake-tactile'] if fake else []))
        out = subprocess.run(cmd, capture_output=True, text=True).stdout
        got_status = _grep_int(out, 'phase 34:', 'status ')
        got_valid = 'valid=True' in out
        ok = (got_status == want_status) and (got_valid == want_valid)
        failures += 0 if ok else 1
        tag = ('  sim+fake' if fake else '  sim     ') if sim else '           '
        print(f'{"PASS" if ok else "FAIL"}  mu {mu:.2f}{tag}  '
              f'carry status {got_status} (want {want_status}), '
              f'mass trustworthy {got_valid} (want {want_valid})')
    print('\nAll cases passed.' if not failures else f'\n{failures} case(s) FAILED.')
    return 1 if failures else 0


def _grep_int(text, line_marker, field):
    for line in text.splitlines():
        if line_marker in line and field in line:
            tail = line.split(field, 1)[1]
            return int(tail.split(',')[0].strip())
    return None


if __name__ == '__main__':
    sys.exit(main())
