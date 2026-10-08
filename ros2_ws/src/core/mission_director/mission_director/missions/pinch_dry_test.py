"""Dry (bench) test of the pinch grasp: real servos, TacTips and controller, no PX4, no flight.

Stands in for the three things PinchGraspMission does that need a flying drone:
  * ATTITUDE. Publishes a level, motionless VehicleOdometry on /dry_test/vehicle_odometry. The
    controller reads it instead of the motion-capture feed when launched with dry_test:=true.
    Nothing is published on any /fmu topic.
  * GRASP PHASES. Steps /md/grasp_phase on a keypress instead of after the flight waypoints.
  * ARM COMMANDS. Forwards /controller/out/servo_state to /servo/in/state.
Everything that grasps -- the TacTip drivers, the controller, the grip law, slip detection and
the safety checks S1-S11 -- runs exactly as in flight.

SETUP. The drone level on a stand, with the space under its centre free: the pads hang about
0.40 m below the body at the 0.208 m tube (logged at startup). The object stands on a support
centred under the body, at the height the pads will reach. Its mass, weighed beforehand, is the
reference for the estimate in the weigh step.

    terminal 1:  ros2 launch ats_bringup real_dual_pinch.launch.py dry_test:=true
    terminal 2:  ros2 run mission_director pinch_dry_test

KEYS (type, then Enter):  <Enter> next step   o  open the pads now   l  arms to the takeoff pose
                          q  quit (arms stopped)                     Ctrl-C  same as q

STEPS
  1. Arms move to the OPEN pre-grasp pose, slowly, after the planned move is shown and confirmed.
  2. Tare + close (phase 31): the force models are re-zeroed off contact, then the shoulders close
     until both pads touch. The angle at first contact gives the object width.
  3. Squeeze (32): the pads press to the grip target. Now take the support away by hand so the
     pads carry the object -- the load along the faces rises and the grip target follows it.
  4. Lift (33): the in-flight checks S7, S9 and S10 switch on. Pull the object down gently to
     raise the load; a sharp pull should read as a slip and tighten the grip.
  5. Weigh (34): hold still. The controller publishes a mass estimate after its steady window.
  6. Place (35): put the support back under the object. The load collapses (touchdown).
  7. Release (36): the pads open, then the arms return to the open pose for another run.
  A fault (status 3) at any step opens the pads; the controller logs which check fired ([S1]...).

The servo driver has no command timeout -- it keeps the last velocity it received -- so this node
sends zero on every exit path. Stopping the driver itself (Ctrl-C in terminal 1) disables torque:
the arms go limp and swing down.
"""

import os
import sys
import math
import time
import threading

import numpy as np
import rclpy
import yaml
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from ament_index_python.packages import get_package_share_directory
from std_msgs.msg import Int8, Int32, Float64
from std_srvs.srv import SetBool, Trigger
from geometry_msgs.msg import WrenchStamped
from sensor_msgs.msg import JointState
from px4_msgs.msg import VehicleOdometry

from pose_based_ats import arm_kinematics as ak


class PinchDryTest(Node):

    # Must match PinchGraspMission and DualArmTactileController.
    PHASE_IDLE = 0
    PHASE_PRE_GRASP = 30
    PHASE_CLOSE = 31
    PHASE_SQUEEZE = 32
    PHASE_LIFT = 33
    PHASE_CARRY = 34
    PHASE_PLACE = 35
    PHASE_RELEASE = 36

    ST_OPEN, ST_WORKING, ST_DONE, ST_FAULT = 0, 1, 2, 3

    # Same as PinchGraspMission.land_arms.
    LAND_ARMS = [1.5708, 0.00, -1.5708, -1.5708, 0.00, 1.5708]
    JOINT_NAMES = ['shoulder 1', 'elbow 1', 'forearm 1', 'shoulder 2', 'elbow 2', 'forearm 2']

    def __init__(self):
        super().__init__('pinch_dry_test')

        self.declare_parameter('geometry_file', os.path.join(
            get_package_share_directory('ats_bringup'), 'config', 'grasp_geometry.yaml'))
        self.declare_parameter('frequency', 30.0)
        self.declare_parameter('move_speed_rad_s', 0.4)     # cap per joint for the pose moves
        self.declare_parameter('move_kp', 1.0)
        self.declare_parameter('move_tolerance_rad', 0.04)  # every joint within this = arrived
        self.declare_parameter('move_timeout_s', 30.0)
        self.declare_parameter('controller_timeout_s', 0.2)  # older controller output -> zero
        self.declare_parameter('touchdown_load_n', 0.25)    # as in PinchGraspMission
        self.declare_parameter('odometry_topic', '/dry_test/vehicle_odometry')
        g = lambda n: self.get_parameter(n).get_parameter_value()

        geometry_file = g('geometry_file').string_value
        with open(geometry_file) as f:
            p = yaml.safe_load(f)['/**']['ros__parameters']
        grasp, timeouts = p['grasp'], p['timeouts']
        self.q_fore = float(grasp['forearm_working_rad'])
        pose = ak.derive_grasp_pose(float(grasp['tube_m']), self.q_fore,
                                    float(grasp['nominal_width_m']),
                                    float(grasp['widest_width_m']),
                                    float(grasp['open_clearance_m']), float(grasp['min_gap_m']))
        sh_open = pose['shoulder_open_rad']
        # Same as PinchGraspMission.pre_grasp_arms.
        self.pre_grasp_arms = [-sh_open, 0.0, self.q_fore, sh_open, 0.0, -self.q_fore]
        lim = [float(v) for v in grasp['shoulder_limits_rad']]
        self.shoulder_limits = {1: (lim[0], lim[1]), 2: (lim[2], lim[3])}
        self.close_timeout = float(timeouts['close_s'])
        self.squeeze_timeout = float(timeouts['squeeze_s'])
        self.release_timeout = float(timeouts['release_s'])

        self.frequency = g('frequency').double_value
        self.move_speed = g('move_speed_rad_s').double_value
        self.move_kp = g('move_kp').double_value
        self.move_tol = g('move_tolerance_rad').double_value
        self.move_timeout = g('move_timeout_s').double_value
        self.ctrl_timeout = g('controller_timeout_s').double_value
        self.touchdown_load = g('touchdown_load_n').double_value

        # --- inputs -------------------------------------------------------------------------------
        self.servo_state = None
        self.ctrl_ref = None
        self.ctrl_ref_t = None
        self.status = None
        self.status_t = None
        self.force_t = {1: None, 2: None}
        self.load = 0.0
        self.n_target = float('nan')
        self.width = float('nan')
        self.mass = float('nan')
        self.create_subscription(JointState, '/servo/out/state', self.cb_servo, 10)
        self.create_subscription(JointState, '/controller/out/servo_state', self.cb_ctrl, 10)
        self.create_subscription(Int8, '/grasp/status', self.cb_status, 10)
        self.create_subscription(Float64, '/grasp/tangential_load',
                                 lambda m: setattr(self, 'load', m.data), 10)
        self.create_subscription(Float64, '/grasp/normal_target',
                                 lambda m: setattr(self, 'n_target', m.data), 10)
        self.create_subscription(Float64, '/grasp/measured_width',
                                 lambda m: setattr(self, 'width', m.data), 10)
        self.create_subscription(Float64, '/grasp/estimated_mass',
                                 lambda m: setattr(self, 'mass', m.data), 10)
        self.create_subscription(WrenchStamped, '/tactip_left/force',
                                 lambda m: self.cb_force(1), 10)
        self.create_subscription(WrenchStamped, '/tactip_right/force',
                                 lambda m: self.cb_force(2), 10)

        # --- outputs ------------------------------------------------------------------------------
        self.pub_servo = self.create_publisher(JointState, '/servo/in/state', 10)
        self.pub_phase = self.create_publisher(Int32, '/md/grasp_phase', 10)
        self.pub_odom = self.create_publisher(VehicleOdometry, g('odometry_topic').string_value, 10)
        self.cli_ssim = {1: self.create_client(SetBool, 'set_ssim_ref_left'),
                         2: self.create_client(SetBool, 'set_ssim_ref_right')}
        self.cli_tare = {1: self.create_client(Trigger, 'tare_force_left'),
                         2: self.create_client(Trigger, 'tare_force_right')}

        # --- state --------------------------------------------------------------------------------
        self.step = 'wait_inputs'
        self.step_t = self.now_s()
        self.first_loop = True
        self.phase = self.PHASE_IDLE
        self.move_target = None
        self.move_next = None
        self.stable_since = None
        self.touchdown_since = None
        self.last_mass = float('nan')
        self.width_reported = False
        self.key = None
        self.key_lock = threading.Lock()
        self.done = False

        threading.Thread(target=self.read_keys, daemon=True).start()
        self.create_timer(1.0 / self.frequency, self.loop)
        self.create_timer(0.02, self.publish_odometry)

        self.get_logger().info(
            f'Dry test ready. Geometry from {geometry_file}: tube {float(grasp["tube_m"]):.3f} m, '
            f'shoulders open at {sh_open:.4f} rad, forearms {self.q_fore:.2f} rad, pads '
            f'{pose["pad_drop_m"]:.3f} m below the body. Odometry (level, still) on '
            f'{g("odometry_topic").string_value}.')

    # ------------------------------------------------------------------ callbacks
    def cb_servo(self, msg):
        self.servo_state = msg

    def cb_ctrl(self, msg):
        self.ctrl_ref = msg
        self.ctrl_ref_t = self.now_s()

    def cb_status(self, msg):
        self.status = msg.data
        self.status_t = self.now_s()

    def cb_force(self, arm):
        self.force_t[arm] = self.now_s()

    def read_keys(self):
        for line in sys.stdin:
            with self.key_lock:
                self.key = line.strip().lower()

    def take_key(self):
        with self.key_lock:
            k, self.key = self.key, None
        return k

    def now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    # ------------------------------------------------------------------ outputs
    def publish_odometry(self):
        """Level and still: PX4 NED <- FRD identity, zero velocity."""
        msg = VehicleOdometry()
        msg.timestamp = int(self.now_s() * 1e6)
        msg.timestamp_sample = msg.timestamp
        msg.pose_frame = VehicleOdometry.POSE_FRAME_NED
        msg.velocity_frame = VehicleOdometry.VELOCITY_FRAME_NED
        msg.q = [1.0, 0.0, 0.0, 0.0]
        msg.position = [0.0, 0.0, 0.0]
        msg.velocity = [0.0, 0.0, 0.0]
        msg.angular_velocity = [0.0, 0.0, 0.0]
        self.pub_odom.publish(msg)

    def send_velocities(self, q_dot):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = [f'q{i + 1}' for i in range(6)]
        msg.velocity = [float(v) for v in q_dot]
        self.pub_servo.publish(msg)

    def stop_arms(self):
        self.send_velocities([0.0] * 6)

    def forward_controller(self):
        """Pass the controller's joint velocities to the servos; zero if they are stale."""
        fresh = (self.ctrl_ref is not None and len(self.ctrl_ref.velocity) >= 6
                 and self.now_s() - self.ctrl_ref_t < self.ctrl_timeout)
        if fresh:
            self.send_velocities(list(self.ctrl_ref.velocity)[:6])
        else:
            self.stop_arms()
            self.get_logger().warn('No fresh controller output -- arms held.',
                                   throttle_duration_sec=2.0)

    # ------------------------------------------------------------------ arm moves
    def joints(self):
        if self.servo_state is None or len(self.servo_state.position) < 6:
            return None
        return [float(v) for v in self.servo_state.position[:6]]

    def wound(self, q_des, q_now):
        """Shoulders onto the legal winding nearest the present angle, as PinchGraspMission does."""
        q = list(q_des)
        for arm, idx in ((1, 0), (2, 3)):
            lo, hi = self.shoulder_limits[arm]
            cands = [q[idx] + 2.0 * math.pi * k for k in range(-3, 4)]
            cands = [c for c in cands if lo <= c <= hi]
            if cands:
                q[idx] = min(cands, key=lambda c: abs(c - q_now[idx]))
        return q

    def start_move(self, q_des, next_step, confirm=True):
        q_now = self.joints()
        self.move_target = self.wound(q_des, q_now)
        self.move_next = next_step
        rows = '\n'.join(f'    {n:<11} {a:+7.3f} -> {b:+7.3f}   ({b - a:+6.3f} rad)'
                         for n, a, b in zip(self.JOINT_NAMES, q_now, self.move_target))
        if confirm:
            print(f'\nPlanned arm move (measured -> target):\n{rows}\n'
                  'Check the directions and sizes against the arms. Press Enter to move, '
                  'q to quit.', flush=True)
            self.go('confirm_move')
        else:
            self.go('move')

    def move_command(self):
        """Slow P move to the target. Returns True on arrival."""
        q_now = self.joints()
        err = np.array(self.move_target) - np.array(q_now)
        if np.all(np.abs(err) < self.move_tol):
            self.stop_arms()
            return True
        self.send_velocities(np.clip(self.move_kp * err, -self.move_speed, self.move_speed))
        return False

    # ------------------------------------------------------------------ helpers
    def go(self, step):
        self.step = step
        self.step_t = self.now_s()
        self.first_loop = True
        self.stable_since = None
        self.touchdown_since = None

    def elapsed(self):
        return self.now_s() - self.step_t

    def prompt(self, text):
        if self.first_loop:
            print(f'\n{text}', flush=True)
            self.first_loop = False

    def call_all(self, clients, request):
        for arm, cli in clients.items():
            if cli.service_is_ready():
                cli.call_async(request)
            else:
                self.get_logger().warn(f'Service {cli.srv_name} not available (arm {arm}).')

    def missing_inputs(self):
        now = self.now_s()
        missing = []
        if self.joints() is None:
            missing.append('/servo/out/state with 6 joints')
        for arm, topic in ((1, '/tactip_left/force'), (2, '/tactip_right/force')):
            if self.force_t[arm] is None or now - self.force_t[arm] > 1.0:
                missing.append(topic)
        if self.status_t is None or now - self.status_t > 1.0:
            missing.append('/grasp/status (controller)')
        return missing

    def fault(self):
        return self.status == self.ST_FAULT

    def report(self):
        """Print the width once and every new mass estimate."""
        if not self.width_reported and np.isfinite(self.width) and self.width > 0.0:
            print(f'  Width at first contact: {self.width * 1000:.1f} mm', flush=True)
            self.width_reported = True
        if np.isfinite(self.mass) and self.mass > 0.0 and self.mass != self.last_mass:
            print(f'  Mass estimate: {self.mass * 1000:.0f} g', flush=True)
            self.last_mass = self.mass

    def open_now(self, why):
        print(f'\n{why} -- opening the pads.', flush=True)
        self.go('release')

    # ------------------------------------------------------------------ main loop
    def loop(self):
        if self.done:
            return
        key = self.take_key()
        if key == 'q':
            self.quit('Quit.')
            return

        grasping = self.step in ('tare', 'close', 'squeeze', 'lift', 'weigh', 'place')
        if grasping and key == 'o':
            self.open_now('Open requested')
            key = None

        # Arms only move while the servo state is live.
        if self.step not in ('wait_inputs', 'confirm_move') and self.joints() is None:
            self.stop_arms()
            return

        match self.step:
            case 'wait_inputs':
                self.phase = self.PHASE_IDLE
                missing = self.missing_inputs()
                if missing:
                    self.get_logger().info(f'Waiting for: {", ".join(missing)}',
                                           throttle_duration_sec=2.0)
                else:
                    self.start_move(self.pre_grasp_arms, 'ready')

            case 'confirm_move':
                self.stop_arms()
                if key == '':
                    self.go('move')

            case 'move':
                self.phase = self.PHASE_IDLE
                if self.move_command():
                    self.go(self.move_next)
                elif self.elapsed() > self.move_timeout:
                    self.stop_arms()
                    err = np.array(self.move_target) - np.array(self.joints())
                    late = ', '.join(f'{n} {e:+.3f}' for n, e in zip(self.JOINT_NAMES, err)
                                     if abs(e) >= self.move_tol)
                    print(f'\nArm move timed out after {self.move_timeout:.0f} s; still off: '
                          f'{late} rad. Arms stopped. Enter retries the move, q quits.',
                          flush=True)
                    self.go('move_failed')

            case 'move_failed':
                self.stop_arms()
                if key == '':
                    self.start_move(self.move_target, self.move_next, confirm=False)

            case 'ready':
                self.phase = self.PHASE_IDLE
                self.stop_arms()
                self.prompt('Pads OPEN. Put the object on its support between the pads, NOT '
                            'touching them.\n  Enter: tare and close   l: arms to the takeoff '
                            'pose   q: quit')
                if key == '':
                    self.width_reported = False
                    self.last_mass = float('nan')
                    self.go('tare')
                elif key == 'l':
                    self.start_move(self.LAND_ARMS, 'parked')

            case 'parked':
                self.phase = self.PHASE_IDLE
                self.stop_arms()
                self.prompt('Arms at the takeoff pose.  Enter: back to the open pose   q: quit')
                if key == '':
                    self.start_move(self.pre_grasp_arms, 'ready', confirm=False)

            case 'tare':
                # Phase 30 resets the controller's per-grasp state; the pads are still off contact,
                # so this is where the force models and the SSIM references are re-zeroed.
                self.phase = self.PHASE_PRE_GRASP
                self.forward_controller()
                if self.first_loop:
                    self.call_all(self.cli_tare, Trigger.Request())
                    self.call_all(self.cli_ssim, SetBool.Request(data=True))
                    print('\nTaring the force models and setting the SSIM references...',
                          flush=True)
                    self.first_loop = False
                if self.elapsed() > 2.0:
                    self.go('close')

            case 'close':
                self.phase = self.PHASE_CLOSE
                self.forward_controller()
                self.prompt('CLOSING (phase 31) until both pads touch.  o: open now')
                self.report()
                if self.status == self.ST_DONE:
                    self.go('squeeze')
                elif self.elapsed() > self.close_timeout:
                    self.open_now(f'No contact on both pads within {self.close_timeout:.0f} s '
                                  f'(status {self.status})')

            case 'squeeze':
                self.phase = self.PHASE_SQUEEZE
                self.forward_controller()
                self.prompt('Both pads in contact. SQUEEZING (phase 32) to the grip target.')
                self.report()
                if self.fault():
                    self.open_now('Fault during the squeeze')
                elif self.status == self.ST_DONE:
                    self.stable_since = self.stable_since or self.now_s()
                    if self.now_s() - self.stable_since > 0.5:
                        self.go('squeezed')
                else:
                    self.stable_since = None
                    if self.elapsed() > self.squeeze_timeout:
                        self.open_now(f'Grip did not settle within {self.squeeze_timeout:.0f} s '
                                      f'(status {self.status})')

            case 'squeezed':
                # Still phase 32: the grip law runs, the in-flight checks do not, so the load can
                # be put on the pads by hand without tripping S9 while the support is moved.
                self.phase = self.PHASE_SQUEEZE
                self.forward_controller()
                self.prompt('Grip at target. Now take the support away so the pads carry the '
                            'object; the grip target should rise with the load.\n  Enter: LIFT '
                            '(in-flight checks S7/S9/S10 on)   o: open')
                self.get_logger().info(f'load {self.load:.2f} N, grip target '
                                       f'{self.n_target:.2f} N, status {self.status}',
                                       throttle_duration_sec=1.0)
                if self.fault():
                    self.open_now('Fault while squeezed')
                elif key == '':
                    self.go('lift')

            case 'lift':
                self.phase = self.PHASE_LIFT
                self.forward_controller()
                self.prompt('LIFT (phase 33). Pull the object down gently to raise the load; a '
                            'sharp pull should read as a slip.\n  Enter: WEIGH   o: open')
                self.get_logger().info(f'load {self.load:.2f} N, grip target '
                                       f'{self.n_target:.2f} N, status {self.status}',
                                       throttle_duration_sec=1.0)
                if self.fault():
                    self.open_now('Fault in lift (see the controller log for the check)')
                elif key == '':
                    self.go('weigh')

            case 'weigh':
                self.phase = self.PHASE_CARRY
                self.forward_controller()
                self.prompt('WEIGH (phase 34). Hands off, hold still.\n  Enter: PLACE   o: open')
                self.report()
                if self.fault():
                    self.open_now('Fault while weighing (see the controller log for the check)')
                elif key == '':
                    self.go('place')

            case 'place':
                self.phase = self.PHASE_PLACE
                self.forward_controller()
                self.prompt('PLACE (phase 35). Put the support back under the object.\n'
                            '  Enter: RELEASE   o: open')
                if self.load < self.touchdown_load:
                    self.touchdown_since = self.touchdown_since or self.now_s()
                    if self.now_s() - self.touchdown_since > 0.3 and self.elapsed() > 1.0:
                        print(f'  Touchdown: the pads carry {self.load:.2f} N.', flush=True)
                        self.go('release')
                else:
                    self.touchdown_since = None
                if key == '':
                    self.go('release')

            case 'release':
                self.phase = self.PHASE_RELEASE
                self.forward_controller()
                self.prompt('RELEASE (phase 36): opening.')
                if self.status == self.ST_OPEN or self.elapsed() > self.release_timeout:
                    self.start_move(self.pre_grasp_arms, 'ready', confirm=False)

        self.pub_phase.publish(Int32(data=self.phase))

    def quit(self, why):
        self.stop_arms()
        self.pub_phase.publish(Int32(data=self.PHASE_IDLE))
        print(f'\n{why} Arms stopped.', flush=True)
        self.done = True


def main(args=None):
    # No rclpy signal handler: Ctrl-C must reach the finally block while the context is still up,
    # or the zero command could not be sent.
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = PinchDryTest()
    try:
        while not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        # Zero the arms on every exit path: the driver keeps the last velocity it was sent.
        if not node.done:
            node.quit('Stopping.')
        for _ in range(5):
            node.stop_arms()
            time.sleep(0.02)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
