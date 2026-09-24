"""Dual-arm tactile grasp controller: side-to-side pinch, shoulder-driven squeeze.

  * The arms meet the object's VERTICAL faces from the sides and it hangs under the drone's
    centre. The grasp is held by FRICTION, so how hard to squeeze is the question the tactile
    sensing has to answer.
  * The SHOULDER presses; the forearm only aims. At this pose the shoulder moves the pad
    401 mm/rad into the face and 61 along, the forearm 113 into and 244 along -- squeezing with
    the forearm would scrub the pad 6.5 mm per 3 mm of indentation. It also keeps the BELT, which
    drives the forearm, out of the force loop.
  * GRIP LAW, the contribution:  N_target = clip(SF * F_tangential / mu_est, floor, ceiling).
    F_tangential is the part of each pad's force running along the face -- for a hanging object,
    its weight shared between the pads. So the grip follows the MEASURED load: no mass estimate,
    no object model, and it adapts mid-lift. A slip bounds the friction from above, so each one
    measures mu and tightens the grip (see detect_slip).
  * The object's width is an OUTPUT: the shoulder angle at first contact gives it to ~14 mm per
    degree, and it is where each arm's relative backstop is measured from.
  * Loops are proportional only -- velocity -> angle -> indentation -> force already integrates.
  * The squeeze is never commanded as a position: 3 mm of indentation is 0.0075 rad of shoulder,
    less than backlash and tube flex. The backstops are limits, not targets.

ARM NAMING. Arm 1 is LEFT (body +y, TacTip B1, Dynamixel 31/32/33); arm 2 is RIGHT (body -y, B2,
41/42/43). The joint vector is [s1, e1, f1, s2, e2, f2], so anything tied to the hardware is
indexed by arm number and anything about where a pad physically is says left or right.

Forces are handled as vectors (see arm_kinematics): the pads meet the faces ~25 deg off square,
so raw sensor Fz is not the squeeze and magnitudes cannot separate grip from load.

Geometry comes from ats_bringup/config/grasp_geometry.yaml -- change the pose there, not here.
"""

import numpy as np
import rclpy
from rclpy.node import Node

from std_msgs.msg import Int8, Int32, Float64, Bool
from geometry_msgs.msg import TwistStamped, WrenchStamped
from sensor_msgs.msg import JointState
from px4_msgs.msg import VehicleOdometry

from . import arm_kinematics as ak

G = 9.80665


class DualArmTactileController(Node):

    # Grasp phases published by PinchGraspMission on /md/grasp_phase. Keep in sync with the
    # mission -- CLOSE and SQUEEZE are separate because they have different success tests.
    PHASE_IDLE = 0
    PHASE_PRE_GRASP = 30
    PHASE_CLOSE = 31
    PHASE_SQUEEZE = 32
    PHASE_LIFT = 33
    PHASE_CARRY = 34
    PHASE_PLACE = 35
    PHASE_RELEASE = 36

    ACTIVE_PHASES = (PHASE_CLOSE, PHASE_SQUEEZE, PHASE_LIFT, PHASE_CARRY, PHASE_PLACE,
                     PHASE_RELEASE)
    # Phases where the pads are the only thing holding the object up, so a falling load means the
    # grasp is failing. PLACE is deliberately NOT here: during a set-down the load is SUPPOSED to
    # fall away as the table takes it, and that must not read as slip.
    FLIGHT_PHASES = (PHASE_LIFT, PHASE_CARRY)

    # Index of each joint in the 6-joint vector [s1, e1, f1, s2, e2, f2].
    SHOULDER_IDX = {1: 0, 2: 3}
    ELBOW_IDX = {1: 1, 2: 4}
    FOREARM_IDX = {1: 2, 2: 5}

    # The two arms run the SAME shoulder locus and the SAME forearm magnitude; only the signs
    # differ. Arm 1 sits at +shoulder / -forearm, arm 2 at -shoulder / +forearm. Closing always
    # DECREASES the magnitude of the shoulder angle.
    SIGN = {1: +1.0, 2: -1.0}

    # Status codes on /grasp/status.
    ST_OPEN = 0        # released / idle
    ST_WORKING = 1     # in progress
    ST_DONE = 2        # this phase's objective is met
    ST_FAULT = 3       # slip alarm or sensor loss -- the mission aborts on this

    def __init__(self):
        super().__init__('dual_arm_tactile_controller')

        # --- Control parameters ---------------------------------------------------------------
        self.declare_parameter('frequency', 30.0)          # matches the TacTip stream (~29.5 Hz)
        self.declare_parameter('sim', True)
        # PHYSICAL mounting convention, one setting because both arms are mounted the same way:
        # the mark points along the direction the pad travels as the arm folds up. arm_kinematics
        # turns that into the per-arm z sign, which differs between the arms because they are
        # mirrored. Getting it wrong is a 180 deg error: normal near zero, mass negative.
        self.declare_parameter('mark_along_fold_direction', True)
        self.declare_parameter('phase_timeout_s', 0.5)     # S8: stale phase message

        # --- Geometry (from grasp_geometry.yaml) --------------------------------------------------
        self.declare_parameter('grasp.tube_m', 0.228)
        self.declare_parameter('grasp.forearm_working_rad', 1.95)
        self.declare_parameter('grasp.shoulder_open_rad', 4.0435)
        self.declare_parameter('grasp.shoulder_nominal_rad', 3.9556)
        self.declare_parameter('grasp.shoulder_min_rad', 3.88)
        self.declare_parameter('grasp.shoulder_max_rad', 4.08)

        # --- Grip law ------------------------------------------------------------------------------
        self.declare_parameter('grip.contact_threshold_n', 0.15)
        self.declare_parameter('grip.force_deadband_n', 0.05)
        self.declare_parameter('grip.safety_factor', 1.4)
        self.declare_parameter('grip.normal_floor_n', 0.3)
        self.declare_parameter('grip.normal_ceiling_n', 1.5)
        self.declare_parameter('grip.mu_initial', 1.0)
        self.declare_parameter('grip.mu_min', 0.3)
        self.declare_parameter('grip.mu_max', 2.0)
        self.declare_parameter('grip.slip_shear_drop_fraction', 0.4)
        self.declare_parameter('grip.slip_shear_floor_mm', 0.10)
        self.declare_parameter('grip.slip_drop_fraction', 0.3)
        self.declare_parameter('grip.slip_backoff', 0.8)
        self.declare_parameter('grip.slip_holdoff_s', 1.0)
        self.declare_parameter('grip.load_steady_window_s', 0.6)
        self.declare_parameter('grip.load_steady_tol_n', 0.05)
        self.declare_parameter('grip.no_grip_confirm_s', 1.5)
        self.declare_parameter('grip.turning_warn_mm', 0.30)
        self.declare_parameter('grip.pad_stiffness_n_per_m', 0.0)

        # --- Safety ---------------------------------------------------------------------------------
        self.declare_parameter('limits.relative_backstop_rad', 0.0075)
        self.declare_parameter('limits.max_depth_mm', 2.8)
        self.declare_parameter('limits.max_force_n', 2.0)
        self.declare_parameter('limits.max_tilt_deg', 25.0)
        self.declare_parameter('limits.tilt_mismatch_deg', 8.0)
        self.declare_parameter('limits.forearm_clamp_rad', 1.95)
        self.declare_parameter('limits.slip_force_n', 0.2)
        self.declare_parameter('limits.slip_time_s', 0.3)
        self.declare_parameter('sensor_timeout_freeze_s', 0.2)
        self.declare_parameter('sensor_timeout_abort_s', 1.0)

        # --- Rates -----------------------------------------------------------------------------------
        self.declare_parameter('rates.shoulder_approach_rad_s', 0.02)
        self.declare_parameter('rates.shoulder_squeeze_rad_s', 0.002)
        self.declare_parameter('rates.shoulder_open_rad_s', 0.05)
        self.declare_parameter('rates.forearm_hold_kp', 2.0)
        self.declare_parameter('rates.forearm_max_rad_s', 0.3)
        self.declare_parameter('rates.squeeze_kp_rad_s_per_n', 0.004)
        # SIM ONLY 
        self.declare_parameter('rates.sim_grip_hold_rad_s', 0.01)

        # --- Estimator ---------------------------------------------------------------------------------
        self.declare_parameter('estimate.settle_s', 1.0)
        self.declare_parameter('estimate.window_s', 2.0)
        self.declare_parameter('estimate.max_speed_m_s', 0.05)
        self.declare_parameter('estimate.max_tilt_deg', 5.0)

        # --- Topics --------------------------------------------------------------------------------------
        self.declare_parameter('arm1_force_topic', '/tactip_left/force')
        self.declare_parameter('arm2_force_topic', '/tactip_right/force')
        self.declare_parameter('arm1_pose_topic', '/tactip_left/pose')
        self.declare_parameter('arm2_pose_topic', '/tactip_right/pose')
        self.declare_parameter('odometry_topic', '/fmu/in/vehicle_visual_odometry')

        g = lambda n: self.get_parameter(n).get_parameter_value()
        self.frequency = g('frequency').double_value
        self.sim = g('sim').bool_value
        along_fold = g('mark_along_fold_direction').bool_value
        self.mark_plus_z = {a: ak.mark_plus_z(a, along_fold) for a in (1, 2)}
        self.phase_timeout = g('phase_timeout_s').double_value

        self.tube = g('grasp.tube_m').double_value
        self.q_fore = g('grasp.forearm_working_rad').double_value
        self.sh_open = g('grasp.shoulder_open_rad').double_value
        self.sh_nominal = g('grasp.shoulder_nominal_rad').double_value
        self.sh_min = g('grasp.shoulder_min_rad').double_value
        self.sh_max = g('grasp.shoulder_max_rad').double_value

        self.contact_n = g('grip.contact_threshold_n').double_value
        self.deadband = g('grip.force_deadband_n').double_value
        self.sf = g('grip.safety_factor').double_value
        self.n_floor = g('grip.normal_floor_n').double_value
        self.n_ceiling = g('grip.normal_ceiling_n').double_value
        self.mu_est = g('grip.mu_initial').double_value
        self.mu_min = g('grip.mu_min').double_value
        self.mu_max = g('grip.mu_max').double_value
        self.slip_shear_drop = g('grip.slip_shear_drop_fraction').double_value
        self.slip_shear_floor = g('grip.slip_shear_floor_mm').double_value
        self.slip_drop = g('grip.slip_drop_fraction').double_value
        self.slip_backoff = g('grip.slip_backoff').double_value
        self.slip_holdoff = g('grip.slip_holdoff_s').double_value
        self.load_window = g('grip.load_steady_window_s').double_value
        self.load_steady_tol = g('grip.load_steady_tol_n').double_value
        self.no_grip_confirm = g('grip.no_grip_confirm_s').double_value
        self.turning_warn = g('grip.turning_warn_mm').double_value
        self.pad_stiffness = g('grip.pad_stiffness_n_per_m').double_value

        self.rel_backstop = g('limits.relative_backstop_rad').double_value
        self.max_depth = g('limits.max_depth_mm').double_value
        self.max_force = g('limits.max_force_n').double_value
        self.max_tilt = g('limits.max_tilt_deg').double_value
        self.tilt_mismatch = g('limits.tilt_mismatch_deg').double_value
        self.fore_clamp = g('limits.forearm_clamp_rad').double_value
        self.slip_force = g('limits.slip_force_n').double_value
        self.slip_time = g('limits.slip_time_s').double_value
        self.t_freeze = g('sensor_timeout_freeze_s').double_value
        self.t_abort = g('sensor_timeout_abort_s').double_value

        self.v_approach = g('rates.shoulder_approach_rad_s').double_value
        self.v_squeeze = g('rates.shoulder_squeeze_rad_s').double_value
        self.v_open = g('rates.shoulder_open_rad_s').double_value
        self.fore_kp = g('rates.forearm_hold_kp').double_value
        self.fore_max = g('rates.forearm_max_rad_s').double_value
        self.squeeze_kp = g('rates.squeeze_kp_rad_s_per_n').double_value
        self.sim_grip_hold = g('rates.sim_grip_hold_rad_s').double_value

        self.settle_s = g('estimate.settle_s').double_value
        self.window_s = g('estimate.window_s').double_value
        self.est_max_speed = g('estimate.max_speed_m_s').double_value
        self.est_max_tilt = g('estimate.max_tilt_deg').double_value

        # --- State ------------------------------------------------------------------------------------------
        self.phase = self.PHASE_IDLE
        self.phase_stamp = 0.0
        self.servo_state = None
        self.force = {1: None, 2: None}            # sensor-frame force vector, newest
        self.force_stamp = {1: 0.0, 2: 0.0}
        self.pose = {1: None, 2: None}             # depth (mm) and tilt (deg) from the pose model
        self.odom = None
        self.contact_shoulder = {1: None, 2: None}  # |shoulder| at that arm's first contact
        self.low_force_since = {1: None, 2: None}
        self.load_ref = 0.0            # highest steady load the pads have carried this grasp
        self.shear_peak = {1: 0.0, 2: 0.0}     # peak pad shear displacement this grasp
        self.ratio_at_peak = 0.0               # shear/normal when that peak was reached
        self.last_slip_t = -1e9
        self.grip_saturated_since = None
        self.load_hist = []            # (t, load), for telling a rising load from a steady one
        self.mu_measured = float('nan')  # friction from a slip at STEADY load -- the real result
        self.no_grip_since = None      # pads pressed but showing no shear: nothing is being held
        self.measured_width = float('nan')
        self.n_target = self.n_floor
        self.tangential_max = 0.0      # largest per-pad load along the face; drops on touchdown
        self.tilt_unreliable = {1: False, 2: False}   # S4: pad outside the models' trained tilt
        self.estimate_started = None
        self.estimate_buf = []
        self.estimated_mass = float('nan')
        self.mass_valid = False
        self.mass_doubt = ''
        self.last_cmd = np.zeros(6)
        self._sim_hist = {}                        # arm -> (angle, time) of the last real movement
        self._sim_squeezed = {1: False, 2: False}  # sim-only: squeeze has run out of travel

        # --- Subscribers -------------------------------------------------------------------------------------
        self.create_subscription(Int32, '/md/grasp_phase', self.cb_phase, 10)
        self.create_subscription(JointState, '/servo/out/state', self.cb_servo, 10)
        self.create_subscription(WrenchStamped, g('arm1_force_topic').string_value,
                                 lambda m: self.cb_force(1, m), 10)
        self.create_subscription(WrenchStamped, g('arm2_force_topic').string_value,
                                 lambda m: self.cb_force(2, m), 10)
        self.create_subscription(TwistStamped, g('arm1_pose_topic').string_value,
                                 lambda m: self.cb_pose(1, m), 10)
        self.create_subscription(TwistStamped, g('arm2_pose_topic').string_value,
                                 lambda m: self.cb_pose(2, m), 10)
        self.create_subscription(VehicleOdometry, g('odometry_topic').string_value, self.cb_odom, 10)

        # --- Publishers ---------------------------------------------------------------------------------------
        self.pub_servo = self.create_publisher(JointState, '/controller/out/servo_state', 10)
        self.pub_status = self.create_publisher(Int8, '/grasp/status', 10)
        self.pub_mass = self.create_publisher(Float64, '/grasp/estimated_mass', 10)
        self.pub_width = self.create_publisher(Float64, '/grasp/measured_width', 10)
        self.pub_mu = self.create_publisher(Float64, '/grasp/mu_estimate', 10)
        # The friction the control loop is USING, vs friction actually MEASURED from a slip at
        # steady load. Only the second is a result; the first is conservative by design.
        self.pub_mu_meas = self.create_publisher(Float64, '/grasp/mu_measured', 10)
        self.pub_n_target = self.create_publisher(Float64, '/grasp/normal_target', 10)
        # The load the pads are carrying along the face. The mission watches this fall to detect
        # touchdown: once the table takes the weight, the pads stop holding it up.
        self.pub_load = self.create_publisher(Float64, '/grasp/tangential_load', 10)
        # The two pads' shear, split into the object sliding vs the object turning in the grasp.
        self.pub_sliding = self.create_publisher(Float64, '/grasp/shear_sliding', 10)
        self.pub_turning = self.create_publisher(Float64, '/grasp/shear_turning', 10)
        # Whether the mass figure can be trusted. Published alongside the mass rather than
        # withholding it, so every object yields a number AND a note about how much to believe it.
        self.pub_mass_valid = self.create_publisher(Bool, '/grasp/estimated_mass_valid', 10)
        self.pub_world_force = {
            1: self.create_publisher(WrenchStamped, '/grasp/force_arm1_world', 10),
            2: self.create_publisher(WrenchStamped, '/grasp/force_arm2_world', 10),
        }

        self.timer = self.create_timer(1.0 / self.frequency, self.control_loop)
        self.get_logger().info(
            f'DualArmTactileController (v2, side grasp) ready: sim={self.sim}, '
            f'{self.frequency:.0f} Hz. Shoulder squeezes from {self.sh_open:.4f} rad, floor '
            f'{self.sh_min:.4f}; forearm held at {self.q_fore:.2f} rad; backstop '
            f'{self.rel_backstop:.4f} rad ({self.rel_backstop * 401:.1f} mm) past first contact; '
            f'grip = {self.sf:.1f} * F_tan / mu (mu0 {self.mu_est:.2f}), '
            f'{self.n_floor:.2f}-{self.n_ceiling:.2f} N. Contact is expected near '
            f'{self.sh_nominal:.4f} rad for a nominal-width object.')

    # ------------------------------------------------------------------ callbacks
    def cb_phase(self, msg: Int32):
        self.phase_stamp = self.now_s()
        if msg.data != self.phase:
            self.phase = msg.data
            self.on_phase_change()

    def cb_servo(self, msg: JointState):
        self.servo_state = msg

    def cb_force(self, arm: int, msg: WrenchStamped):
        self.force[arm] = np.array([msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z])
        self.force_stamp[arm] = self.now_s()

    def cb_pose(self, arm: int, msg: TwistStamped):
        # linear x/y are the pad's SHEAR displacement in mm -- how far the membrane has been
        # dragged sideways. That is the slip signal: it grows with the tangential load while the
        # pad is gripping, and falls back when the object actually slides.
        shear = np.array([msg.twist.linear.x, msg.twist.linear.y, 0.0], dtype=float)
        self.pose[arm] = {'depth_mm': abs(msg.twist.linear.z),
                          'shear_vec': shear,        # sensor frame, mm, in the contact plane
                          'shear_mm': float(np.linalg.norm(shear)),
                          'tilt_deg': float(np.hypot(msg.twist.angular.x, msg.twist.angular.y))}

    def cb_odom(self, msg: VehicleOdometry):
        self.odom = msg

    def now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_phase_change(self):
        """Reset per-phase state. Both arms stay symmetric throughout -- nothing freezes."""
        if self.phase in (self.PHASE_PRE_GRASP, self.PHASE_CLOSE):
            self.contact_shoulder = {1: None, 2: None}
            self.tilt_unreliable = {1: False, 2: False}
            self.measured_width = float('nan')
            self._sim_hist = {}
            self._sim_squeezed = {1: False, 2: False}
        if self.phase == self.PHASE_CARRY:
            self.estimate_started = self.estimate_started or self.now_s()
        else:
            self.estimate_started = None
            self.estimate_buf = []
        if self.phase in (self.PHASE_PRE_GRASP, self.PHASE_CLOSE):
            self.mass_valid = False
            self.mass_doubt = ''
        if self.phase == self.PHASE_RELEASE:
            self.low_force_since = {1: None, 2: None}
        if self.phase in (self.PHASE_PRE_GRASP, self.PHASE_CLOSE):
            self.load_ref = 0.0
            self.no_grip_since = None
            self.shear_peak = {1: 0.0, 2: 0.0}
            self.ratio_at_peak = 0.0
            self.grip_saturated_since = None

    # ------------------------------------------------------------------ measurement helpers
    def joint(self, idx):
        if self.servo_state is None or len(self.servo_state.position) <= idx:
            return None
        return float(self.servo_state.position[idx])

    def joints(self, arm):
        """(shoulder, elbow, forearm) measured angles for one arm, or None."""
        vals = [self.joint(self.SHOULDER_IDX[arm]), self.joint(self.ELBOW_IDX[arm]),
                self.joint(self.FOREARM_IDX[arm])]
        return None if any(v is None for v in vals) else vals

    def shoulder_mag(self, arm):
        """The shoulder angle as a positive magnitude, so both arms read alike.

        Arm 1 sits near +3.95 and arm 2 near -3.95; closing DECREASES this on both. The reading is
        first folded onto the winding this controller reasons about, because the mission commands
        whichever legal winding is nearest the joint -- and that differs between sim and hardware.
        Without it an arm parked on an equivalent winding reads far below the travel floor and the
        controller refuses to close at all.
        """
        q = self.joint(self.SHOULDER_IDX[arm])
        if q is None:
            return None
        mag = self.SIGN[arm] * q
        return mag + 2.0 * np.pi * round((self.sh_nominal - mag) / (2.0 * np.pi))

    def press_dir_world(self, arm):
        """Unit vector along which this pad pushes the object, in world ENU."""
        q1, q2 = self.joints(1), self.joints(2)
        if q1 is None or q2 is None or self.odom is None:
            return None
        R_wb = ak.R_world_from_body(self.odom.q)
        n_body = ak.press_direction_body(arm, q1, q2, tube_m=self.tube)
        if R_wb is None or n_body is None:
            return None
        return R_wb @ n_body

    def world_force(self, arm):
        """Pad force in world ENU (z up), or None if anything is missing."""
        q = self.joints(arm)
        if q is None or self.force[arm] is None or self.odom is None:
            return None
        return ak.world_force(arm, q[0], q[1], q[2], self.force[arm], self.odom.q,
                              self.mark_plus_z[arm])

    def contact_forces(self, arm):
        """(normal_n, tangential_n) for one pad: into the face, and along it.

        The normal is the squeeze the grip law regulates. The tangential part is what friction has
        to hold -- for a hanging object, its weight shared between the pads.
        """
        f_w = self.world_force(arm)
        n_hat = self.press_dir_world(arm)
        if f_w is None or n_hat is None:
            return None, None
        n, f_tan = ak.decompose_force(f_w, n_hat)
        return float(n), float(np.linalg.norm(f_tan))

    def expected_tilt(self, arm):
        """The contact angle this pose should produce, straight from forward kinematics."""
        q1, q2 = self.joints(1), self.joints(2)
        if q1 is None or q2 is None:
            return None
        return ak.contact_angle_deg(arm, q1, q2, tube_m=self.tube)

    def shear_world(self, arm):
        """This pad's shear displacement as a vector in the world (ENU, mm).

        The membrane is dragged by whatever the object is doing, so putting both pads' shear into
        one frame makes the two failure modes separable -- see `shear_split`.
        """
        p = self.pose[arm]
        q = self.joints(arm)
        if p is None or q is None or self.odom is None or 'shear_vec' not in p:
            return None
        R_wb = ak.R_world_from_body(self.odom.q)
        if R_wb is None:
            return None
        return R_wb @ ak.R_body_from_sensor(arm, q[0], q[1], q[2],
                                            self.mark_plus_z[arm]) @ p['shear_vec']

    def shear_split(self):
        """Split the two pads' shear into "the object is sliding" and "the object is turning".

        Sliding drags both membranes the same way, so the SUM is large. Turning about the line
        joining the contacts moves them oppositely, so the DIFFERENCE is large -- and a two-point
        pinch leaves that rotation essentially unconstrained, so it is worth watching separately.

        Returns (sliding_mm, turning_mm), or (None, None).
        """
        s1, s2 = self.shear_world(1), self.shear_world(2)
        if s1 is None or s2 is None:
            return None, None
        return float(np.linalg.norm(s1 + s2) / 2.0), float(np.linalg.norm(s1 - s2) / 2.0)

    def sensor_age(self, arm):
        return self.now_s() - self.force_stamp[arm] if self.force_stamp[arm] else 1e9

    # grip law
    def update_grip_target(self):
        """N_target = clip(SF * F_tangential / mu_est, floor, ceiling), plus slip bookkeeping.

        The larger of the two pads' tangential readings sets the target, so both arms squeeze
        enough for the worse-off pad. Before the object is lifted the tangential force is near
        zero and the floor applies; during the lift it rises with the load and the grip follows.
        """
        tangential, normals = [], []
        for arm in (1, 2):
            n, t = self.contact_forces(arm)
            if n is None:
                continue
            tangential.append(t)
            normals.append(n)
        if not tangential:
            self.n_target = self.n_floor
            self.tangential_max = 0.0
            return
        self.tangential_max = float(max(tangential))

        self.detect_slip(self.tangential_max, max(normals))

        self.n_target = float(np.clip(self.sf * self.tangential_max / max(self.mu_est, self.mu_min),
                                      self.n_floor, self.n_ceiling))

    def detect_slip(self, load, normal):
        """Spot the object sliding, and use it to measure the friction.

        Force alone cannot do this. In steady state the controller sets the normal FROM the
        measured load, so the shear/normal ratio it reads back is just SF/mu_est -- its own
        command echoed. And a sliding object never reports its weight: friction caps the
        tangential reading at mu*N, so a heavy object that is sliding and a light one that is held
        look identical, and feeding that back into N = SF*F_tan/mu_est is a fixed point the grip
        never climbs out of.

        Motion breaks the tie, and the TacTip measures it: the pad's SHEAR DISPLACEMENT grows as
        the membrane is dragged by the load and falls back when the object breaks away. Sliding
        began at the ratio the pad was working at when the shear peaked, so the true friction is
        BELOW that -- a real measurement. mu_est drops to `slip_backoff` times it, and since
        N_target = SF * F_tan / mu_est, a lower mu means a HARDER squeeze. 
        """
        now = self.now_s()
        if self.phase not in self.FLIGHT_PHASES:
            self.shear_peak = {1: 0.0, 2: 0.0}
            self.ratio_at_peak = 0.0
            self.load_ref = 0.0
            self.load_hist = []
            self.grip_saturated_since = None
            return
        if normal < self.n_floor * 0.5:
            return

        ratio = load / max(normal, 1e-6)
        # A slip while the load is still climbing is expected -- the grip starts at its floor and
        # only learns the weight by feeling it - so it raises the grip but is NOT recorded as a
        # friction measurement, or every grasp would bias mu down. Only steady-load slips count.
        self.load_hist.append((now, load))
        self.load_hist = [(t, v) for t, v in self.load_hist if now - t <= self.load_window]
        rising = (len(self.load_hist) > 2
                  and (load - self.load_hist[0][1]) > self.load_steady_tol)

        slipped, why = False, ''

        # Signal 1, the sensor-native one: the pad's shear displacement falls back.
        for arm in (1, 2):
            p = self.pose[arm]
            if p is None or 'shear_mm' not in p:
                continue
            sh = p['shear_mm']
            if sh > self.shear_peak[arm]:
                self.shear_peak[arm] = sh
                self.ratio_at_peak = max(self.ratio_at_peak, ratio)
            elif (self.shear_peak[arm] > self.slip_shear_floor
                  and sh < (1.0 - self.slip_shear_drop) * self.shear_peak[arm]):
                # RELATIVE The floor keeps noise at near-zero shear from looking like a collapse.
                slipped, why = True, (f'arm {arm} shear fell to {sh:.2f} mm from a peak of '
                                      f'{self.shear_peak[arm]:.2f} mm')

        # Signal 2: the carried load collapses while the pads are still pressed. Same event seen
        # in force; kept because it also catches an outright drop with no shear data.
        if self.load_ref > 0.0 and load < (1.0 - self.slip_drop) * self.load_ref:
            slipped, why = True, (f'the carried load fell from {self.load_ref:.2f} N to '
                                  f'{load:.2f} N')
        self.load_ref = max(self.load_ref, load)

        if slipped and (now - self.last_slip_t) > self.slip_holdoff:
            observed = self.ratio_at_peak if self.ratio_at_peak > 0.0 else ratio
            new_mu = float(np.clip(min(self.mu_est, observed) * self.slip_backoff,
                                   self.mu_min, self.mu_max))
            if rising:
                self.get_logger().warn(
                    f'Slip while the load was still being taken up ({why}, pads at {normal:.2f} N)'
                    f' -- squeezing harder (mu_est {self.mu_est:.2f} -> {new_mu:.2f}). Not '
                    'recorded as a friction measurement: the grip had not caught up with the load '
                    'yet, so this says nothing about the surface.')
            else:
                self.mu_measured = observed
                self.get_logger().warn(
                    f'Slip at steady load: {why}, with the pads pressing at {normal:.2f} N. '
                    f'Sliding began at a shear/normal of {observed:.2f}, so the friction is below '
                    f'that -- FRICTION MEASURED at mu < {observed:.2f}; mu_est {self.mu_est:.2f} '
                    f'-> {new_mu:.2f}, which raises the grip target.')
            self.mu_est = new_mu
            self.last_slip_t = now
            self.load_ref = load
            self.shear_peak = {1: 0.0, 2: 0.0}
            self.ratio_at_peak = 0.0

        # If the grip has been pinned at its ceiling and the object is still going, there is
        # nothing left to try: the pads cannot hold this object at this friction. Say so, so the
        # mission sets it down instead of carrying a grasp that is already failing.
        at_ceiling = normal >= self.n_ceiling - self.deadband
        if at_ceiling and slipped:
            self.grip_saturated_since = self.grip_saturated_since or now
        elif not at_ceiling:
            self.grip_saturated_since = None

    # main loop
    def control_loop(self):
        # S8: not our phase, or the mission has stopped talking -> hold everything still.
        if self.phase not in self.ACTIVE_PHASES:
            self.publish(np.zeros(6), self.ST_OPEN)
            return
        if (self.now_s() - self.phase_stamp) > self.phase_timeout:
            self.get_logger().warn('Grasp phase message is stale -- holding the arms.',
                                   throttle_duration_sec=2.0)
            self.publish(np.zeros(6), self.ST_WORKING)
            return
        if self.servo_state is None or len(self.servo_state.position) < 6:
            self.publish(np.zeros(6), self.ST_WORKING)
            return

        # S6: stale sensors freeze the arms, then abort. Skipped in sim, where the drivers publish
        # fabricated data at their own rate and there is nothing to lose.
        if not self.sim:
            age = max(self.sensor_age(1), self.sensor_age(2))
            if age > self.t_abort:
                self.get_logger().error(f'No TacTip data for {age:.2f} s -- aborting grasp.')
                self.publish(np.zeros(6), self.ST_FAULT)
                return
            if age > self.t_freeze:
                self.get_logger().warn(f'TacTip data stale ({age:.2f} s) -- holding.',
                                       throttle_duration_sec=1.0)
                self.publish(np.zeros(6), self.ST_WORKING)
                return

        self.update_grip_target()

        q_dot = np.zeros(6)
        # The elbows hold zero and the forearms hold the working angle throughout. Velocity mode,
        # so "hold" is a small P loop, not a zero command -- in Gazebo a zero velocity does not
        # hold an arm against gravity (real Dynamixels in velocity mode do).
        for arm in (1, 2):
            q_dot[self.ELBOW_IDX[arm]] = self.hold_command(self.ELBOW_IDX[arm], 0.0)
            q_dot[self.FOREARM_IDX[arm]] = self.forearm_command(arm)
            q_dot[self.SHOULDER_IDX[arm]] = self.shoulder_command(arm)

        self.last_cmd = q_dot
        self.update_estimate()
        self.publish(q_dot, self.status())

    # ------------------------------------------------------------------ per-joint control
    def hold_command(self, idx, target):
        q = self.joint(idx)
        if q is None:
            return 0.0
        return float(np.clip(self.fore_kp * (target - q), -self.fore_max, self.fore_max))

    def forearm_command(self, arm):
        """S5: the forearm is a positioning command now, clamped clear of the belt limit."""
        target = self.SIGN[arm] * -1.0 * min(self.q_fore, self.fore_clamp)
        return self.hold_command(self.FOREARM_IDX[arm], target)

    def shoulder_command(self, arm):
        """Velocity for one shoulder. Positive `v_close` means squeezing further."""
        mag = self.shoulder_mag(arm)
        if mag is None:
            return 0.0

        if self.phase == self.PHASE_RELEASE:
            v_close = -self.v_open if mag < min(self.sh_open, self.sh_max) else 0.0
            return self.SIGN[arm] * -1.0 * v_close

        # In sim the TacTip drivers publish fabricated data, so there is no force to regulate.
        # Close on position until the object stalls the joint -- enough to exercise the mission
        # flow. No mass is estimated in sim
        if self.sim:
            v_close = self.sim_close_rate(arm, mag)
            return self.SIGN[arm] * -1.0 * self.apply_limits(arm, mag, v_close)

        n, _ = self.contact_forces(arm)
        if n is None:
            return 0.0

        # First contact: record this arm's own shoulder angle. Everything downstream -- the
        # backstop and the width measurement -- is referred to it, so the grasp follows the object
        # instead of assuming where the drone ended up.
        if self.contact_shoulder[arm] is None and n > self.contact_n \
                and self.contact_is_plausible(arm, mag):
            self.contact_shoulder[arm] = mag
            self.get_logger().info(f'Arm {arm} first contact at shoulder {mag:.4f} rad ({n:.2f} N).')
            self.record_width()

        if self.contact_shoulder[arm] is None:
            v_close = self.v_approach                       # still closing the air gap
        else:
            err = self.n_target - n
            v_close = 0.0 if abs(err) < self.deadband else float(
                np.clip(self.squeeze_kp * err, -self.v_squeeze, self.v_squeeze))

        return self.SIGN[arm] * -1.0 * self.apply_limits(arm, mag, v_close)

    def sim_close_rate(self, arm, mag):
        """Sim-only: close steadily, and treat a stalled joint as contact.

        BOTH the close and the squeeze have to accept a STALL as success, not just reaching an
        angle. In Gazebo the object physically blocks the shoulder wherever contact happens, which
        can be a little short of wherever the target is 
        """
        if self.contact_shoulder[arm] is not None:
            # Creep the backstop's worth past contact -- the same travel the force loop would use
            # -- then KEEP COMMANDING a small closing rate rather than zero. The joint cannot move
            # with the object in the way, so all the command does is make Gazebo press the pads
            # in, which is the only way a grip force exists there. apply_limits still bounds it.
            if mag <= self.contact_shoulder[arm] - 0.8 * self.rel_backstop:
                self._sim_squeezed[arm] = True
                return self.sim_grip_hold
            if self._stalled(arm, mag):
                if not self._sim_squeezed[arm]:
                    self.get_logger().info(f'Arm {arm} squeeze stalled at shoulder {mag:.4f} rad '
                                           '-- the object is taking the load (sim).')
                self._sim_squeezed[arm] = True
                return self.sim_grip_hold
            return self.v_squeeze
        if self._stalled(arm, mag) and self.contact_is_plausible(arm, mag):
            self.contact_shoulder[arm] = mag                # stalled while being told to close
            self.get_logger().info(f'Arm {arm} stalled at shoulder {mag:.4f} rad -- sim contact.')
            self._sim_hist.pop(arm, None)                   # restart the test for the squeeze
            self.record_width()
        return self.v_approach

    def _stalled(self, arm, mag):
        
        idx = self.SHOULDER_IDX[arm]
        cmd = float(self.last_cmd[idx])
        # Closing drives |shoulder| down, i.e. the joint velocity opposes the joint's own sign.
        if cmd * self.SIGN[arm] > -1e-6:
            self._sim_hist.pop(arm, None)
            return False
        now = self.now_s()
        prev = self._sim_hist.get(arm)
        if prev is None:
            self._sim_hist[arm] = (mag, now)
            return False
        elapsed = now - prev[1]
        if elapsed < 0.5:
            return False
        if abs(mag - prev[0]) > 0.25 * abs(cmd) * elapsed:
            self._sim_hist[arm] = (mag, now)                # still tracking its command
            return False
        return True

    def contact_is_plausible(self, arm, mag):
        """Reject a "contact" that cannot physically be one.

        The arms only ever close from the opening pose, so a contact reported at a shoulder no
        narrower than that means the detector misfired -- nothing had moved inward yet. Left
        unchecked the bogus angle propagates into the width and the backstop, and the whole
        mission proceeds on a grasp that never happened.
        """
        if mag >= self.sh_open - 0.002:
            self.get_logger().error(
                f'Arm {arm} reported contact at shoulder {mag:.4f} rad, which is no narrower than '
                f'the opening pose ({self.sh_open:.4f} rad). It cannot have touched anything '
                'before it started closing -- ignoring.', throttle_duration_sec=5.0)
            return False
        return True

    def record_width(self):
        """Once both arms have touched, the shoulder angles ARE the object's width."""
        if any(self.contact_shoulder[a] is None for a in (1, 2)):
            return
        q1, q2 = self.joints(1), self.joints(2)
        if q1 is None or q2 is None:
            return
        w = ak.grasp_width_m(q1, q2, tube_m=self.tube)
        if w is None:
            return
        # SYSTEMATIC BIAS: contact is only declared once each pad reads `contact_threshold_n`, by
        # which point it is already indented, so the raw figure UNDERSTATES the width by twice
        # that -- about 1 mm. The bench stiffness curve adds it back; 0 leaves it uncorrected.
        if self.pad_stiffness > 0.0:
            w += 2.0 * self.contact_n / self.pad_stiffness
        self.measured_width = w
        angles = [ak.contact_angle_deg(a, q1, q2, tube_m=self.tube) for a in (1, 2)]
        # How far this object is from the nominal one, in shoulder terms: a quick check that the
        # arms stopped where an object of this width should have stopped them.
        off = np.mean([self.contact_shoulder[a] for a in (1, 2)]) - self.sh_nominal
        self.get_logger().info(
            f'Object width measured at {w * 1000:.1f} mm ({off * 401 * 2:+.1f} mm vs nominal); '
            f'contact angles {angles[0]:.1f} / {angles[1]:.1f} deg'
            + ('' if max(angles) <= self.max_tilt else '  -- OUTSIDE the models trained tilt range'))

    def apply_limits(self, arm, mag, v_close):
        """Safety layer S1-S4 on the squeeze axis. `v_close` is positive when closing further."""
        if v_close <= 0.0:
            # Opening is always allowed, up to the shoulders' travel ceiling.
            return 0.0 if mag >= self.sh_max else v_close

        touch = self.contact_shoulder[arm]
        if touch is not None and mag <= touch - self.rel_backstop:              # S1
            self.get_logger().warn(f'Arm {arm} at the relative backstop ({mag:.4f} rad, '
                                   f'{(touch - mag) * 1000:.1f} mm past contact).',
                                   throttle_duration_sec=2.0)
            return 0.0
        if mag <= self.sh_min:                                                  # absolute net
            self.get_logger().warn(f'Arm {arm} at the shoulder travel floor ({mag:.4f} rad) -- '
                                   'no contact was ever detected.', throttle_duration_sec=2.0)
            return 0.0
        # S2 and S4 read the POSE MODEL, which in sim is fabricated -- a CONSTANT 3.0 mm depth.
        # Ungated, S2 replaces every close command with a back-off and the arms never reach the
        # object. The sensor-timeout checks are gated the same way.
        p = None if self.sim else self.pose[arm]
        if p is not None and p['depth_mm'] > self.max_depth:                    # S2
            self.get_logger().warn(f'Arm {arm} depth {p["depth_mm"]:.2f} mm past the limit -- '
                                   'backing off.', throttle_duration_sec=2.0)
            return -self.v_squeeze
        if self.force[arm] is not None and np.linalg.norm(self.force[arm]) > self.max_force:  # S3
            self.get_logger().warn(f'Arm {arm} force overload -- backing off.',
                                   throttle_duration_sec=2.0)
            return -self.v_squeeze
        # S4 FLAGS the reading rather than holding the squeeze, deliberately against the spec: the
        # nominal 10 cm object sits at 24.9 deg, so a holding rule blocks all but the narrowest.
        # Tilt is not a hardware hazard either -- S1, S2 and S3 protect the pads. What it means is
        # that the force reading is outside the models' training, so it gates the mass estimate.
        if p is not None:
            unreliable = p['tilt_deg'] > self.max_tilt
            if unreliable and not self.tilt_unreliable[arm]:
                self.get_logger().warn(
                    f'Arm {arm} tilt {p["tilt_deg"]:.1f} deg is outside the models trained range '
                    f'({self.max_tilt:.0f} deg) -- continuing, but its force reading is flagged '
                    'unreliable and no mass will be reported.')
            self.tilt_unreliable[arm] = unreliable
            # The hard stop is a DISAGREEMENT check: FK already says what tilt this pose should
            # produce, whatever the width, so this catches "the arm is not where it thinks" without
            # caring how wide the object is. An absolute ceiling could not do both.
            expected = self.expected_tilt(arm)
            if expected is not None and abs(p['tilt_deg'] - expected) > self.tilt_mismatch:
                self.get_logger().error(
                    f'Arm {arm} reads {p["tilt_deg"]:.1f} deg of tilt where the kinematics say '
                    f'{expected:.1f} deg -- the pad is not meeting the face the way this pose '
                    'says it should. Stopping the squeeze.', throttle_duration_sec=2.0)
                return 0.0
        return v_close

    # ------------------------------------------------------------------ status
    def status(self):
        """What the mission keys off. ST_DONE means THIS phase's objective is met."""
        if self.phase == self.PHASE_RELEASE:
            if self.sim:
                return self.ST_OPEN if all(
                    (self.shoulder_mag(a) or 0.0) >= self.sh_open - 0.005 for a in (1, 2)
                ) else self.ST_WORKING
            normals = [self.contact_forces(a)[0] for a in (1, 2)]
            if any(n is None for n in normals):
                return self.ST_WORKING
            return self.ST_OPEN if max(normals) < 0.1 else self.ST_WORKING

        if self.phase == self.PHASE_CLOSE:
            return self.ST_DONE if all(self.contact_shoulder[a] is not None
                                       for a in (1, 2)) else self.ST_WORKING

        if self.sim:
            # Nothing to judge from fabricated forces: "squeezed" means both arms have run their
            # backstop's worth past contact and stopped.
            if self.phase == self.PHASE_SQUEEZE:
                return self.ST_DONE if all(self._sim_squeezed[a] for a in (1, 2)) \
                    else self.ST_WORKING
            return self.ST_DONE

        normals = [self.contact_forces(a)[0] for a in (1, 2)]
        if any(n is None for n in normals):
            return self.ST_WORKING

        # NOT ACTUALLY GRIPPING. A pad carrying weight MUST shear, so pressed pads reading no
        # shear in flight mean the object is sliding continuously or was never there. This is the
        # one failure the load-following law cannot see alone: a sliding object reports a small,
        # steady, plausible tangential force (friction caps it at mu*N), so the grip never rises
        # and the mass reads far too low -- the drone would fly off believing it has the object.
        if self.phase in self.FLIGHT_PHASES:
            if self.gripping():
                self.no_grip_since = None
            else:
                self.no_grip_since = self.no_grip_since or self.now_s()
                if (self.now_s() - self.no_grip_since) > self.no_grip_confirm:
                    self.get_logger().error(
                        'The pads are pressed but show no shear: nothing is being held. Either '
                        'the object is sliding continuously or it is not between the pads. '
                        'Setting down rather than carrying a grasp that does not exist.',
                        throttle_duration_sec=2.0)
                    return self.ST_FAULT

        # The grip is maxed out and the object is still sliding: it cannot be held.
        if self.grip_saturated_since is not None \
                and (self.now_s() - self.grip_saturated_since) > self.slip_holdoff:
            self.get_logger().error(
                f'Grip is at its {self.n_ceiling:.2f} N ceiling and the object is still slipping '
                f'(mu_est {self.mu_est:.2f}) -- it cannot be held. Setting it down.')
            return self.ST_FAULT

        # S7: in flight, a pad that goes slack has lost the object.
        now = self.now_s()
        for arm, n in zip((1, 2), normals):
            self.low_force_since[arm] = (self.low_force_since[arm] or now) \
                if n < self.slip_force else None
        if self.phase in self.FLIGHT_PHASES:
            for arm in (1, 2):
                t0 = self.low_force_since[arm]
                if t0 is not None and (now - t0) > self.slip_time:
                    self.get_logger().error(f'Arm {arm} lost contact in flight -- slip alarm.')
                    return self.ST_FAULT

        return self.ST_DONE if min(normals) >= self.n_target - self.deadband else self.ST_WORKING

    def gripping(self):
        """True when the pads show the shear a genuinely held object must produce.

        Both pads have to show it: one pad shearing while the other does not means the object is
        running down one face.
        """
        for arm in (1, 2):
            p = self.pose[arm]
            if p is None or p.get('shear_mm', 0.0) < self.slip_shear_floor:
                return False
        return True

    # ------------------------------------------------------------------ mass, a by-product
    def update_estimate(self):
        """Weigh the object: only in steady hover, then publish the median of a window.

        In a side grasp the squeeze is horizontal and the weight vertical, so the equal-and-
        opposite grip cancels out of the vertical sum and what is left is the load.
        """
        if self.phase != self.PHASE_CARRY or self.sim or self.odom is None:
            return
        if self.estimate_started is None:
            self.estimate_started = self.now_s()
        if (self.now_s() - self.estimate_started) < self.settle_s:
            return                                                        # let the climb settle
        vel = np.asarray(self.odom.velocity, dtype=float)
        if not np.isfinite(vel).all() or np.linalg.norm(vel) > self.est_max_speed:
            return
        R_wb = ak.R_world_from_body(self.odom.q)
        if R_wb is None:
            return
        if np.degrees(np.arccos(np.clip(R_wb[2, 2], -1.0, 1.0))) > self.est_max_tilt:
            return
        # PUBLISHED whatever the pads say, with a flag for how much to believe it. Withholding it
        # would leave every object outside the trained tilt with no number -- and those are the
        # measurements needed to find out whether the tilt matters at all.
        doubts = []
        if any(self.tilt_unreliable[a] for a in (1, 2)):
            doubts.append('pad tilt outside the models trained range')
        if not self.gripping():
            doubts.append('pads show no shear, so the object is not being held')
        self.mass_doubt = '; '.join(doubts)
        f1, f2 = self.world_force(1), self.world_force(2)
        if f1 is None or f2 is None:
            return

        self.estimate_buf.append(float((f1[2] + f2[2]) / G))
        if (self.now_s() - self.estimate_started) > (self.settle_s + self.window_s):
            self.estimated_mass = float(np.median(self.estimate_buf))
            self.mass_valid = not self.mass_doubt
            note = '' if self.mass_valid else f'  NOT TRUSTWORTHY: {self.mass_doubt}'
            self.get_logger().info(
                f'Mass estimate {self.estimated_mass * 1000:.0f} g over '
                f'{len(self.estimate_buf)} samples '
                f'(spread {np.std(self.estimate_buf) * 1000:.0f} g).{note}')
            self.estimate_buf = []
            self.estimate_started = self.now_s()

    # ------------------------------------------------------------------ output
    def publish(self, q_dot, status):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = [f'q{i + 1}' for i in range(6)]
        msg.velocity = [float(v) for v in q_dot]
        self.pub_servo.publish(msg)

        self.pub_status.publish(Int8(data=int(status)))
        self.pub_mass.publish(Float64(data=float(self.estimated_mass)))
        self.pub_width.publish(Float64(data=float(self.measured_width)))
        self.pub_mu.publish(Float64(data=float(self.mu_est)))
        self.pub_mu_meas.publish(Float64(data=float(self.mu_measured)))
        self.pub_n_target.publish(Float64(data=float(self.n_target)))
        self.pub_load.publish(Float64(data=float(self.tangential_max)))
        self.pub_mass_valid.publish(Bool(data=bool(self.mass_valid)))
        sliding, turning = self.shear_split()
        if sliding is not None:
            self.pub_sliding.publish(Float64(data=sliding))
            self.pub_turning.publish(Float64(data=turning))
            if turning > self.turning_warn and self.phase in self.FLIGHT_PHASES:
                self.get_logger().warn(
                    f'The two pads are shearing against each other by {turning:.2f} mm: the '
                    'object is turning in the grasp, not just hanging. A two-point pinch barely '
                    'resists this -- grip higher above its centre of mass if it persists.',
                    throttle_duration_sec=3.0)

        for arm in (1, 2):
            f_w = self.world_force(arm)
            if f_w is None:
                continue
            w = WrenchStamped()
            w.header.stamp = msg.header.stamp
            w.header.frame_id = 'world_enu'
            w.wrench.force.x, w.wrench.force.y, w.wrench.force.z = (float(f_w[0]), float(f_w[1]),
                                                                    float(f_w[2]))
            self.pub_world_force[arm].publish(w)


def main(args=None):
    rclpy.init(args=args)
    node = DualArmTactileController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
