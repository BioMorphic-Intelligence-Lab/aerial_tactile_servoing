import datetime

import rclpy
import tf2_ros

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
        self.declare_parameter('im.tactile_servoing_time', 65.0)  # [s] max servoing duration
        self.tactile_servoing_time = self.get_parameter('im.tactile_servoing_time').get_parameter_value().double_value

        self.declare_parameter('im.no_contact_timeout', 10.0)  # [s] lost-contact grace period
        no_contact_timeout = self.get_parameter('im.no_contact_timeout').get_parameter_value().double_value
        self.ts_no_contact_counter = 0
        self.ts_no_contact_max_cycles = int(no_contact_timeout * self.frequency)

        # --- Tactip interfaces ---
        self.sub_tactip = self.create_subscription(TwistStamped, '/tactip/pose', self.tactip_callback, 10)
        self.sub_tactip_contact = self.create_subscription(Int8, '/tactip/contact', self.tactip_contact_callback, 10)
        self.tactip_data = None
        self.contact = False

        # --- Controller interfaces ---
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
        self.publish_servo_velocity_references(self.servo_reference.velocity)

    def lost_contact(self):
        """True when contact has been absent longer than the grace period."""
        return self.ts_no_contact_counter > self.ts_no_contact_max_cycles

    def servoing_time_elapsed(self):
        """True when the mission has been servoing longer than tactile_servoing_time."""
        return (datetime.datetime.now() - self.state_start_time).seconds > self.tactile_servoing_time

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def tactip_callback(self, msg):
        self.tactip_data = msg

    def tactip_contact_callback(self, msg):
        self.contact = bool(msg.data)

    def controller_callback(self, msg):
        self.vehicle_trajectory_setpoint = msg

    def controller_servo_callback(self, msg):
        self.servo_reference = msg

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
