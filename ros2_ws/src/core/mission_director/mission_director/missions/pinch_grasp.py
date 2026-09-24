import datetime
import math

import numpy as np
from std_msgs.msg import Int8, Int32, Float64, Bool
from std_srvs.srv import SetBool, Trigger

from mission_director.base_classes.tactile_mission_director import TactileMissionDirector, run_mission


class PinchGraspMission(TactileMissionDirector):
    """
    The drone hovers directly over the object, lowers two arms down its sides and squeezes
    horizontally:
        set the gripper geometry -> descend -> close until both pads touch (that shoulder angle is
        the width) -> squeeze to the grip the load needs -> lift slowly watching the shear ->
        carry -> tactile touchdown -> release

    The arms are never commanded to a closing angle: 3 mm of indentation is 0.0075 rad of
    shoulder, less than backlash and tube flex. This mission only positions them at the OPEN
    pre-grasp pose and hands the SHOULDERS to DualArmTactileController.

    While grasping, the controller owns all six joints and this mission owns the drone, holding it
    on a POSITION setpoint -> the grasp states do not call state_hover or state_move_uam_to_position

    ARM NAMING: arm 1 is LEFT (body +y, B1), arm 2 is RIGHT (body -y, B2).

    Geometry comes from ats_bringup/config/grasp_geometry.yaml -- change the pose there, not here.
    """

    # Grasp phases on /md/grasp_phase. Must match DualArmTactileController.
    PHASE_IDLE = 0
    PHASE_PRE_GRASP = 30
    PHASE_CLOSE = 31
    PHASE_SQUEEZE = 32
    PHASE_LIFT = 33
    PHASE_CARRY = 34
    PHASE_PLACE = 35
    PHASE_RELEASE = 36

    def __init__(self):
        super().__init__('mission_director')

        # --- Geometry (defaults mirror grasp_geometry.yaml) ------------------------------------
        self.declare_parameter('geom.pad_drop_m', 0.4012)
        self.declare_parameter('geom.object_north_m', 2.0)
        self.declare_parameter('geom.object_east_m', 0.0)
        self.declare_parameter('geom.object_centre_altitude_m', 1.050)
        self.declare_parameter('geom.approach_clearance_m', 0.30)
        self.declare_parameter('geom.dropoff_north_m', 2.0)
        self.declare_parameter('geom.dropoff_east_m', 2.0)
        self.declare_parameter('geom.dropoff_centre_altitude_m', 1.050)
        self.declare_parameter('geom.lift_height_m', 0.25)
        self.declare_parameter('geom.grasp_above_centre_m', 0.015)
        self.declare_parameter('geom.altitude_datum_offset_m', 0.10)
        self.declare_parameter('grasp.shoulder_open_rad', 4.0435)
        self.declare_parameter('grasp.forearm_working_rad', 1.95)
        self.declare_parameter('timeouts.close_s', 25.0)
        self.declare_parameter('timeouts.squeeze_s', 15.0)
        self.declare_parameter('timeouts.weigh_s', 12.0)
        self.declare_parameter('timeouts.release_s', 8.0)
        self.declare_parameter('touchdown_load_n', 0.25)
        # The shoulders' configured travel, per arm, as [min, max]. MUST match `min_angles` /
        # `max_angles` in ats_bringup/config/dxl_ros2_pinch_grasp.yaml -- the driver zeroes the
        # velocity of a joint outside its range, so a pose commanded on the wrong winding would
        # freeze the arm. Used to pick the legal winding nearest the arm's present position.
        self.declare_parameter('grasp.shoulder_limits_rad', [-1.0, 5.0, -5.0, 1.0])
        self.declare_parameter('target_yaw_rad', 0.0)

        g = lambda n: self.get_parameter(n).get_parameter_value()
        pad_drop = g('geom.pad_drop_m').double_value
        obj_n = g('geom.object_north_m').double_value
        obj_e = g('geom.object_east_m').double_value
        obj_alt = g('geom.object_centre_altitude_m').double_value
        clearance = g('geom.approach_clearance_m').double_value
        drop_n = g('geom.dropoff_north_m').double_value
        drop_e = g('geom.dropoff_east_m').double_value
        drop_alt = g('geom.dropoff_centre_altitude_m').double_value
        lift_h = g('geom.lift_height_m').double_value
        above = g('geom.grasp_above_centre_m').double_value
        datum = g('geom.altitude_datum_offset_m').double_value
        self.close_timeout = g('timeouts.close_s').double_value
        self.squeeze_timeout = g('timeouts.squeeze_s').double_value
        self.weigh_timeout = g('timeouts.weigh_s').double_value
        self.release_timeout = g('timeouts.release_s').double_value
        self.touchdown_load = g('touchdown_load_n').double_value
        lim = list(g('grasp.shoulder_limits_rad').double_array_value) or [-1.0, 5.0, -5.0, 1.0]
        self.shoulder_limits = {1: (lim[0], lim[1]), 2: (lim[2], lim[3])}

        
        self.target_yaw = g('target_yaw_rad').double_value
        self.home_position[3] = self.target_yaw
        self.hover_position[3] = self.target_yaw

        # Waypoints, all DERIVED (NED: negative z is up) 
        # GRIP ABOVE THE CENTRE OF MASS. A two-point pinch leaves the object free to rotate about
        # the line joining the two contacts.
        # ALTITUDE DATUM -PX4: mocap?
        self.z_grasp = -(obj_alt + above + pad_drop - datum)
        self.z_over = self.z_grasp - clearance          # hover above before descending
        self.z_lift = self.z_grasp - lift_h
        self.z_drop = -(drop_alt + above + pad_drop - datum)
        self.z_over_drop = self.z_drop - clearance
        self.takeoff_altitude = -self.z_over

        self.engage_pos = [obj_n, obj_e, self.z_grasp, self.target_yaw]
        self.over_object_pos = [obj_n, obj_e, self.z_over, self.target_yaw]
        self.dropoff_pos = [drop_n, drop_e, self.z_drop, self.target_yaw]
        self.over_dropoff_pos = [drop_n, drop_e, self.z_over_drop, self.target_yaw]
        self.home_pos = [0.0, 0.0, self.z_over, self.target_yaw]

        # Arm poses 
        # Takeoff/landing: upper arms out, forearms up, pads 0.27 m ABOVE the body.
    
        self.land_arms = [4.7124, 0.00, 1.5708, -4.7124, 0.00, -1.5708]

        # Pre-grasp: shoulders open wide enough for the widest declared object, forearms at the
        # working angle. This is the last position-commanded arm pose; from here the controller
        # takes over. Arm 1 is +shoulder / -forearm, arm 2 is -shoulder / +forearm -- the two arms
        # run the SAME shoulder locus and mirror through the FOREARM sign, so giving both forearms
        # the same sign puts one arm out to the side instead of under the drone.
        sh_open = g('grasp.shoulder_open_rad').double_value
        q_fore = g('grasp.forearm_working_rad').double_value
        self.pre_grasp_arms = [sh_open, 0.0, -q_fore, -sh_open, 0.0, q_fore]

        # --- Controller interfaces ----------------------------------------------------------------
        self.sub_grasp_status = self.create_subscription(Int8, '/grasp/status',
                                                         self.callback_grasp_status, 10)
        self.sub_estimated_mass = self.create_subscription(Float64, '/grasp/estimated_mass',
                                                           self.callback_estimated_mass, 10)
        self.sub_mass_valid = self.create_subscription(Bool, '/grasp/estimated_mass_valid',
                                                       self.callback_mass_valid, 10)
        self.sub_measured_width = self.create_subscription(Float64, '/grasp/measured_width',
                                                           self.callback_measured_width, 10)
        self.sub_load = self.create_subscription(Float64, '/grasp/tangential_load',
                                                 self.callback_load, 10)
        self.grasp_status = 0
        self.estimated_mass = float('nan')
        self.mass_valid = False
        self.measured_width = float('nan')
        self.width_reported = False
        self.tangential_load = 0.0
        self.grasp_stable_since = None
        self.touchdown_since = None

        # Second SSIM client, for the RIGHT arm's TacTip. The LEFT arm uses the client the base
        # class already makes on 'set_ssim_ref', which the launch remaps to 'set_ssim_ref_left'.
        self.cli_set_ssim_ref_right = self.create_client(SetBool, 'set_ssim_ref_right')
        # Force-tare clients: the model has a small non-zero force bias at no contact and it
        # drifts with temperature, so it is re-zeroed right before closing, off contact.
        self.cli_tare_left = self.create_client(Trigger, 'tare_force_left')
        self.cli_tare_right = self.create_client(Trigger, 'tare_force_right')
        self.tare_requested = False
        if not self.sim:
            while not self.cli_set_ssim_ref_right.wait_for_service(timeout_sec=1.0):
                self.get_logger().info('set_ssim_ref_right service not available, waiting...')

        # Dedicated grasp-phase signal. NOT /md/state: the helper states republish their own state
        # number in the same cycle, so a listener keying off /md/state would toggle at loop rate.
        self.pub_grasp_phase = self.create_publisher(Int32, '/md/grasp_phase', 10)
        self.grasp_phase = self.PHASE_IDLE

        self.get_logger().info(
            f'PinchGraspMission (v2, side grasp) ready. Object at N {obj_n:.2f} E {obj_e:.2f}, '
            f'centre {obj_alt:.3f} m; pads hang {pad_drop:.3f} m below the body, so the grasp '
            f'hover is {-self.z_grasp:.3f} m (PX4 datum, {datum:.3f} m above the floor) '
            f'directly above it. Shoulders open at '
            f'{sh_open:.4f} rad, forearms held at {q_fore:.2f} rad. The arms are '
            'position-commanded only to the OPEN pre-grasp pose; the controller closes them.')

    # callbacks
    def callback_grasp_status(self, msg: Int8):
        self.grasp_status = msg.data

    def callback_estimated_mass(self, msg: Float64):
        self.estimated_mass = msg.data

    def callback_mass_valid(self, msg: Bool):
        self.mass_valid = msg.data

    def callback_measured_width(self, msg: Float64):
        self.measured_width = msg.data

    def callback_load(self, msg: Float64):
        self.tangential_load = msg.data

    # helpers
    def set_new_ssim_ref_both(self, value: bool):
        """Set the SSIM contact reference on both TacTips."""
        self.set_new_ssim_ref(value)                      # left arm (arm 1, B1), inherited
        req = SetBool.Request()
        req.data = value
        self.future_right = self.cli_set_ssim_ref_right.call_async(req)

    def request_force_tare(self):
        """Re-zero both force models. Only valid with the pads off contact."""
        for name, cli in (('left', self.cli_tare_left), ('right', self.cli_tare_right)):
            if cli.service_is_ready():
                cli.call_async(Trigger.Request())
            else:
                self.get_logger().warn(f'tare_force service for the {name} TacTip is not '
                                       'available; using the startup tare.')

    def pinch_outputs(self, position):
        """Hold the drone on a POSITION setpoint and pass the controller's joint velocities
        through to the servos. Used by every grasp state.
        """
        self.publish_trajectory_position_setpoint(position[0], position[1], position[2],
                                                  self.target_yaw)
        ref = getattr(self, 'servo_reference', None)
        if ref is not None and len(ref.velocity) >= 6:
            self.publish_servo_velocity_references(list(ref.velocity))
        else:
            # No controller output yet: hold the arms still rather than guess an angle.
            self.publish_servo_velocity_references([0.0] * 6)
            self.get_logger().warn('No controller servo reference yet -- arms held.',
                                   throttle_duration_sec=2.0)

    def wound_arms(self, q_des):
        """Rewrite the shoulder angles of a commanded pose onto their nearest LEGAL winding.

        Only the shoulders are rewound; the other joints have ranges narrower than a full turn.
        """
        q = list(q_des)
        state = getattr(self, 'servo_state', None)
        if state is None or len(state.position) < 6:
            return q                      # no feedback yet; command it as written
        for arm, idx in ((1, 0), (2, 3)):
            # In sim the driver is not in the loop, so the URDF's +-6.28 is the real constraint.
            lo, hi = (-6.28, 6.28) if self.sim else self.shoulder_limits[arm]
            current = float(state.position[idx])
            best = None
            for k in range(-3, 4):
                cand = q[idx] + 2.0 * math.pi * k
                if lo <= cand <= hi and (best is None
                                         or abs(cand - current) < abs(best - current)):
                    best = cand
            if best is None:
                self.get_logger().error(
                    f'No winding of {q[idx]:+.4f} rad for arm {arm}\'s shoulder lies inside its '
                    f'configured range [{lo}, {hi}] -- commanding it as written, which the driver '
                    'will refuse to move.', throttle_duration_sec=10.0)
            elif abs(best - q[idx]) > 1e-6:
                self.get_logger().info(
                    f'Arm {arm} shoulder: commanding {best:+.4f} rad instead of {q[idx]:+.4f} -- '
                    f'same pose, but {abs(best - current):.2f} rad away instead of '
                    f'{abs(q[idx] - current):.2f} from where the joint is now.',
                    throttle_duration_sec=10.0)
                q[idx] = best
        return q

    def at_position(self, target, tol_horizontal=0.10, tol_vertical=0.07):
        """True when the drone is within tolerance of a NED waypoint."""
        p = self.vehicle_local_position
        return (math.hypot(p.x - target[0], p.y - target[1]) < tol_horizontal
                and abs(p.z - target[2]) < tol_vertical)

    def is_settled(self, speed_limit=0.05):
        p = self.vehicle_local_position
        return math.sqrt(p.vx ** 2 + p.vy ** 2 + p.vz ** 2) < speed_limit

    def elapsed_s(self):
        return (datetime.datetime.now() - self.state_start_time).total_seconds()

    def report_width(self):
        """Log the measured width once, the first time the controller publishes one."""
        if not self.width_reported and np.isfinite(self.measured_width) \
                and self.measured_width > 0.0:
            self.get_logger().info(f'Object width measured: {self.measured_width * 1000:.1f} mm')
            self.width_reported = True

    def check_slip(self):
        """Grasp lost in flight -- set the object down where we are rather than carry on."""
        if self.grasp_status == 3:
            self.get_logger().error('Grasp lost (slip alarm) -- descending to set the object down.')
            self.transition_to_state('emergency_set_down')
            return True
        return False

    # mission
    def execute(self):
        # Default each cycle; only the grasp states below claim a phase, so any other state
        # releases the controller automatically.
        self.grasp_phase = self.PHASE_IDLE

        match self.FSM_state:
            case "entrypoint":
                self.state_entrypoint(next_state="arms_takeoff_position",
                                      target_heading=self.target_yaw)

            case "arms_takeoff_position":
                self.state_move_arms(q_des=self.wound_arms(self.land_arms),
                                     next_state="wait_for_arm_offboard",
                                     target_heading=self.target_yaw)

            case "wait_for_arm_offboard":
                if not self.got_ref and not self.sim:
                    self.set_new_ssim_ref_both(True)
                self.state_wait_for_arming(next_state="takeoff")

            case "takeoff":
                self.state_takeoff(target_altitude=self.takeoff_altitude, next_state="hover",
                                   target_heading=self.target_yaw)

            case "hover":
                self.state_hover(duration_sec=2, next_state="pre_grasp_pose",
                                 target_heading=self.target_yaw)

            # step 1: set the gripper geometry (not a grasp command) 
            case "pre_grasp_pose":
                self.state_move_arms(q_des=self.wound_arms(self.pre_grasp_arms),
                                     next_state="wait_pre_grasp",
                                     target_heading=self.target_yaw)

            case "wait_pre_grasp":
                self.state_hover(duration_sec=2, next_state="approach",
                                 target_heading=self.target_yaw)

            # step 2: fly over the object, then descend onto it
            case "approach":
                self.state_move_uam_to_position(target_position=self.over_object_pos,
                                                next_state="wait_approach")

            case "wait_approach":
                self.state_hover(duration_sec=2, next_state="descend",
                                 target_heading=self.target_yaw)

            case "descend":
                self.state_move_uam_to_position(target_position=self.engage_pos,
                                                next_state="wait_engage")

            # phase 30: settled at grasp height, pads still open 
            case "wait_engage":
                self.handle_state(state_number=self.PHASE_PRE_GRASP)
                self.grasp_phase = self.PHASE_PRE_GRASP
                self.pinch_outputs(self.engage_pos)
                if not self.tare_requested and not self.sim:
                    self.request_force_tare()        # off contact, right before closing
                    self.tare_requested = True
                if self.elapsed_s() > 3.0 and self.is_settled():
                    self.transition_to_state("close_grasp")

            # phase 31: close until both pads report contact; that angle is the width 
            case "close_grasp":
                self.handle_state(state_number=self.PHASE_CLOSE)
                self.grasp_phase = self.PHASE_CLOSE
                self.pinch_outputs(self.engage_pos)
                if self.first_state_loop:
                    self.get_logger().info('Closing on force -- the controller owns the shoulders '
                                           'now. The angle at first contact measures the object.')
                    self.first_state_loop = False
                self.report_width()

                if self.grasp_status == 2:
                    self.get_logger().info('Both pads in contact -- squeezing.')
                    self.transition_to_state("squeeze")
                elif self.elapsed_s() > self.close_timeout:
                    self.get_logger().error(
                        f'No contact on both pads within {self.close_timeout:.0f} s '
                        f'(status {self.grasp_status}) -- opening and backing off.')
                    self.transition_to_state("abort_open")

            # phase 32: squeeze to the grip the object actually needs
            case "squeeze":
                self.handle_state(state_number=self.PHASE_SQUEEZE)
                self.grasp_phase = self.PHASE_SQUEEZE
                self.pinch_outputs(self.engage_pos)
                self.report_width()
                if self.check_slip():
                    return

                if self.grasp_status == 2:
                    self.grasp_stable_since = self.grasp_stable_since or datetime.datetime.now()
                    if (datetime.datetime.now() - self.grasp_stable_since).total_seconds() > 0.5:
                        self.get_logger().info('Grip at target -- lifting.')
                        self.transition_to_state("lift")
                else:
                    self.grasp_stable_since = None

                if self.elapsed_s() > self.squeeze_timeout:
                    self.get_logger().error(
                        f'Grip did not settle within {self.squeeze_timeout:.0f} s '
                        f'(status {self.grasp_status}) -- opening and backing off.')
                    self.transition_to_state("abort_open")

            # phase 33: lift slowly; the grip rises with the load as the object takes its own
            #      weight, and a slip alarm aborts to a set-down 
            case "lift":
                self.handle_state(state_number=self.PHASE_LIFT)
                self.grasp_phase = self.PHASE_LIFT
                self.pinch_outputs([self.engage_pos[0], self.engage_pos[1], self.z_lift,
                                    self.target_yaw])
                if self.check_slip():
                    return
                target = [self.engage_pos[0], self.engage_pos[1], self.z_lift]
                if self.at_position(target) and self.is_settled():
                    self.transition_to_state("weigh")
                elif self.elapsed_s() > 20.0:
                    self.get_logger().warn('Lift timed out; weighing anyway.')
                    self.transition_to_state("weigh")

            # phase 34: weigh (a by-product), then carry 
            case "weigh":
                self.handle_state(state_number=self.PHASE_CARRY)
                self.grasp_phase = self.PHASE_CARRY
                self.pinch_outputs([self.engage_pos[0], self.engage_pos[1], self.z_lift,
                                    self.target_yaw])
                if self.check_slip():
                    return
                if self.first_state_loop:
                    self.get_logger().info('Holding still to weigh the object.')
                    self.first_state_loop = False
                # The controller only publishes a figure once it has a full steady window, and
                # flags whether the pads were in a state where it can be believed. Either way the
                # mission carries on -- the mass is a by-product, not a gate.
                if np.isfinite(self.estimated_mass) and self.estimated_mass > 0.0:
                    tag = '' if self.mass_valid else ' (flagged unreliable -- see the controller log)'
                    self.get_logger().info(
                        f'Estimated payload mass: {self.estimated_mass*1000:.0f} g{tag}')
                    self.transition_to_state("transport")
                elif self.elapsed_s() > self.weigh_timeout:
                    self.get_logger().warn('No mass estimate within the window -- carrying on.')
                    self.transition_to_state("transport")

            case "transport":
                self.handle_state(state_number=self.PHASE_CARRY)
                self.grasp_phase = self.PHASE_CARRY
                self.pinch_outputs([self.over_dropoff_pos[0], self.over_dropoff_pos[1],
                                    self.z_lift, self.target_yaw])
                if self.check_slip():
                    return
                if self.at_position([self.over_dropoff_pos[0], self.over_dropoff_pos[1],
                                     self.z_lift]) and self.is_settled():
                    self.transition_to_state("place_down")

            # phase 35: descend until the table takes the weight 
            # Touchdown is detected tactilely, not by altitude: once the object is supported, the
            # pads stop carrying it and the load along the face collapses. That works whatever the
            # table height actually is. This is its own phase because that collapse looks exactly
            # like a slip, and the controller must not raise the slip alarm on a set-down.
            case "place_down":
                self.handle_state(state_number=self.PHASE_PLACE)
                self.grasp_phase = self.PHASE_PLACE
                self.pinch_outputs(self.dropoff_pos)
                if self.check_slip():
                    return
                landed = (not self.sim) and self.tangential_load < self.touchdown_load \
                    and self.elapsed_s() > 2.0
                if landed:
                    self.touchdown_since = self.touchdown_since or datetime.datetime.now()
                    if (datetime.datetime.now() - self.touchdown_since).total_seconds() > 0.3:
                        self.get_logger().info(
                            f'Touchdown: the pads are carrying {self.tangential_load:.2f} N -- '
                            'the table has the object. Releasing.')
                        self.transition_to_state("release_grasp")
                else:
                    self.touchdown_since = None

                if self.at_position(self.dropoff_pos[:3]) and self.is_settled():
                    self.transition_to_state("release_grasp")
                elif self.elapsed_s() > 20.0:
                    self.get_logger().warn('Descent timed out; releasing here.')
                    self.transition_to_state("release_grasp")

            # phase 36: release 
            case "release_grasp":
                self.handle_state(state_number=self.PHASE_RELEASE)
                self.grasp_phase = self.PHASE_RELEASE
                self.pinch_outputs(self.dropoff_pos)
                if self.grasp_status == 0 or self.elapsed_s() > self.release_timeout:
                    self.get_logger().info('Released.')
                    self.transition_to_state("climb_out")

            # Climb straight up before going anywhere: the arms are still down around the object.
            case "climb_out":
                self.state_move_uam_to_position(target_position=self.over_dropoff_pos,
                                                next_state="wait_disengage")

            case "wait_disengage":
                self.state_hover(duration_sec=2, next_state="return_to_launch",
                                 target_heading=self.target_yaw)

            case "return_to_launch":
                self.state_move_uam_to_position(target_position=self.home_pos,
                                                next_state="land_arms_position")

            case "land_arms_position":
                self.state_move_arms(q_des=self.wound_arms(self.land_arms), next_state="land",
                                     target_heading=self.target_yaw)

            case "land":
                self.state_land(next_state="done", target_heading=self.target_yaw)

            case "done":
                self.get_logger().info('Dual-arm pinch grasp mission complete.',
                                       throttle_duration_sec=5.)

            # abort paths 
            case "abort_open":
                # Closing or squeezing failed: open the pads (phase 36 drives the controller to
                # open) and then climb out. Never leave the arms loaded against an object we could
                # not confirm.
                self.handle_state(state_number=self.PHASE_RELEASE)
                self.grasp_phase = self.PHASE_RELEASE
                self.pinch_outputs(self.engage_pos)
                if self.grasp_status == 0 or self.elapsed_s() > self.release_timeout:
                    self.transition_to_state("retreat")

            case "retreat":
                self.state_move_uam_to_position(target_position=self.over_object_pos,
                                                next_state="return_to_launch")

            case "emergency_set_down":
                # Slip in flight: descend to the grasp height where we are and open. Better to put
                # it down under control than to carry a grasp we no longer trust.
                self.handle_state(state_number=self.PHASE_RELEASE)
                self.grasp_phase = self.PHASE_RELEASE
                p = self.vehicle_local_position
                self.pinch_outputs([p.x, p.y, self.z_grasp, self.target_yaw])
                if self.elapsed_s() > self.release_timeout:
                    self.transition_to_state("return_to_launch")

            case "emergency":
                self.state_emergency()

            case _:
                self.get_logger().error(f'Unknown state: {self.FSM_state}')
                self.transition_to_state(new_state="emergency")

        self.pub_grasp_phase.publish(Int32(data=self.grasp_phase))


def main():
    run_mission(PinchGraspMission)


if __name__ == '__main__':
    main()
