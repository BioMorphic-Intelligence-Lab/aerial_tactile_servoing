import datetime

import rclpy
import tf2_ros
import numpy as np

from std_msgs.msg import Int8
from std_srvs.srv import SetBool
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import JointState
from px4_msgs.msg import TrajectorySetpoint

from mission_director.base_classes.uam_state_machine import UAMStateMachine


class TactileMissionDirector(UAMStateMachine):
    """Shared tactile-servoing layer on top of UAMStateMachine.

    Everything the vbats-family missions had copy-pasted lives here: the tactip
    and controller interfaces, the SSIM contact-reference service, the TF lookup,
    and the per-cycle passthrough of controller outputs to platform and servos.

    Concrete missions subclass this and implement execute() -- the finite state
    machine that defines the mission sequence. Tuned constants come from ROS
    parameters (a per-mission YAML), so a mission can be retuned without code
    changes.

    This is deliberately a separate class from UAMStateMachine: the base is
    mission-type agnostic and is also used by non-tactile missions (ymca,
    uam_control_test, uam_md_test), which must NOT inherit the tactip wiring --
    in particular the blocking wait for the set_ssim_ref service would hang them.
    """

    def __init__(self, node_name: str = 'mission_director'):
        super().__init__(node_name)

        # --- Generic tactile-servoing parameters (shared by all tactile missions) ---
        self.declare_parameter('im.n_tactips', 1)
        self.declare_parameter('im.tactile_servoing_time', 65.0)  # [s] max servoing duration
        self.declare_parameter('im.no_contact_timeout', 10.0)  # [s] lost-contact grace period

        self.n_tactips = self.get_parameter('im.n_tactips').get_parameter_value().integer_value
        self.tactile_servoing_time = self.get_parameter('im.tactile_servoing_time').get_parameter_value().double_value
        no_contact_timeout = self.get_parameter('im.no_contact_timeout').get_parameter_value().double_value
        self.ts_no_contact_counter = 0
        self.ts_no_contact_max_cycles = int(no_contact_timeout * self.frequency)

        # --- Tactip interfaces ---
        self.tactip_pose_subscribers_ = []
        self.tactip_contact_subscribers_ = []

        self.tactip_pose_ = [None] * self.n_tactips
        self.tactip_contact_ = [None] * self.n_tactips
        for n in range(self.n_tactips):
            self.tactip_pose_subscribers_.append(self.create_subscription(TwistStamped, '/tactip_'+str(n)+'/pose', self._make_tactip_pose_callback(n), 10))
            self.tactip_contact_subscribers_.append(self.create_subscription(Int8, '/tactip_'+str(n)+'/contact', self._make_tactip_contact_callback(n), 10))

        # --- Tactile Servoing controller interfaces ---
        self.sub_controller = self.create_subscription(TrajectorySetpoint, '/controller/out/trajectory_setpoint', self.controller_callback, 10)
        self.sub_controller_servo = self.create_subscription(JointState, '/controller/out/servo_state', self.controller_servo_callback, 10)
        self.vehicle_trajectory_setpoint = TrajectorySetpoint()
        self.servo_reference = JointState()

        # --- SSIM contact-reference service ---
        self.cli_set_ssim_ref = self.create_client(SetBool, 'set_ssim_ref')
        self.got_ref = False
        if not self.sim:  # in sim there is no tactip node providing this service
            while not self.cli_set_ssim_ref.wait_for_service(timeout_sec=1.0):
                self.get_logger().info('set_ssim_ref service not available, waiting...')

        # --- Kinematics (TF) ---
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # --- Timer (always last) ---
        self.counter = 0
        self.timer = self.create_timer(self.timer_period, self.execute)

    # ------------------------------------------------------------------
    # FSM entry point -- concrete missions must override
    # ------------------------------------------------------------------
    def execute(self):
        raise NotImplementedError('Concrete missions must implement execute().')

    # ------------------------------------------------------------------
    # Shared tactile-servoing helpers
    # ------------------------------------------------------------------
    def publish_tactile_servoing_outputs(self):
        """Pass the controller outputs through to platform + servos and update the
        no-contact counter. Call once per cycle from the tactile-servoing state."""
        if not self.contact:
            self.ts_no_contact_counter += 1
            self.get_logger().info(
                f'No contact detected. Counter: {self.ts_no_contact_counter}/{self.ts_no_contact_max_cycles}',
                throttle_duration_sec=1)
        else:
            self.ts_no_contact_counter = 0

        if self.first_state_loop:
            self.get_logger().info('Tactile servoing initiated.')
            self.first_state_loop = False

        self.publish_trajectory_velocity_setpoint(
            self.vehicle_trajectory_setpoint.velocity[0],
            self.vehicle_trajectory_setpoint.velocity[1],
            self.vehicle_trajectory_setpoint.velocity[2],
            self.vehicle_trajectory_setpoint.yawspeed,
        )

        for n in range(self.n_arms):
            self.publish[n].publish(self.servo_reference.velocity[n*3:n*3+3])

    def lost_contact(self):
        """True when contact has been absent longer than the grace period."""
        return self.ts_no_contact_counter > self.ts_no_contact_max_cycles

    def servoing_time_elapsed(self):
        """True when the mission has been servoing longer than tactile_servoing_time."""
        return (datetime.datetime.now() - self.state_start_time).seconds > self.tactile_servoing_time

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def controller_callback(self, msg):
        self.vehicle_trajectory_setpoint = msg

    def controller_servo_callback(self, msg):
        self.servo_reference = msg

    def _make_tactip_pose_callback(self, index):
        def callback(msg: TwistStamped):
            self.tactip_pose_[index] = msg.twist
        return callback

    def _make_tactip_contact_callback(self, index):
        def callback(msg: Int8):
            self.tactip_contact_[index] = msg.data
        return callback

    def set_new_ssim_ref(self, value: bool):
        """Set the current tactip image as the SSIM contact-detection reference."""
        req = SetBool.Request()
        req.data = value
        self.future = self.cli_set_ssim_ref.call_async(req)
        self.got_ref = True

    def get_transform(self):
        try:
            return self.tf_buffer.lookup_transform(
                'present_sensor_frame',   # target frame
                'world',                  # source frame
                rclpy.time.Time(),
            )
        except Exception as e:
            self.get_logger().warn(str(e))


def run_mission(node_cls):
    """Standard rclpy entry point shared by all tactile missions."""
    rclpy.init(args=None)
    node = node_cls()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
