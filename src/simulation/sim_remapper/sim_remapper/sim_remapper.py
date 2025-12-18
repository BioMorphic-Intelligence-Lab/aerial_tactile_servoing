import rclpy
from rclpy.node import Node

from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from std_msgs.msg import Float64
from sensor_msgs.msg import JointState

class SimRemapper(Node):
    def __init__(self):
        super().__init__('sim_remapper')

        self.declare_parameter('frequency', 25.)
        self.declare_parameter('verbose', False)
        self.declare_parameter('operating_mode', 'position')  # position or velocity
        self.operating_mode = self.get_parameter('operating_mode').get_parameter_value().string_value

        # Arm subscribers
        self.subscriber_servo_state = self.create_subscription(JointState, '/servo/in/state', self.servo_refs_callback, 10)
        self.subscriber_joint_states = self.create_subscription(JointState, '/servo/out/state', self.joint_states_callback, 10)
        # Arms publishers
        self.ros2_publishers = [
            self.create_publisher(Float64, '/shoulder_1_vel_cmd', 10),
            self.create_publisher(Float64, '/elbow_1_vel_cmd', 10),
            self.create_publisher(Float64, '/forearm_1_vel_cmd', 10),
            self.create_publisher(Float64, '/shoulder_2_vel_cmd', 10),
            self.create_publisher(Float64, '/elbow_2_vel_cmd', 10),
            self.create_publisher(Float64, '/forearm_2_vel_cmd', 10)
        ]
        # self.direcions = [1., -1., -1., ]

        self.q_cmd = []
        self.arm_positions = []
        self.arm_velocities = []
        self.kp = 2.
        self.kd = 0.3

        self.feedback_length = 0
        self.ref_length = 0

        self.verbose = self.get_parameter('verbose').get_parameter_value().bool_value

        # Timer -- always last
        self.counter = 0
        self.frequency = self.get_parameter('frequency').get_parameter_value().double_value
        self.timer_period = 1.0 / self.frequency
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
    
    def timer_callback(self):
        if self.operating_mode == 'position':
            self.move_arm_to_position(self.q_cmd)
        elif self.operating_mode == 'velocity':
            self.publish_arm_velocity_commands(self.q_cmd)

        if self.feedback_length != self.ref_length:
            self.get_logger().warn(f"Feedback length {self.feedback_length} does not match reference length {self.ref_length}", throttle_duration_sec=5.0)

    def joint_states_callback(self, msg):
        self.feedback_length = len(msg.position)
        self.arm_positions = msg.position
        self.arm_velocities = msg.velocity

    def servo_refs_callback(self, msg):
        self.ref_length = len(msg.position)
        self.q_cmd = msg.position

    def publish_arm_velocity_commands(self, q_dot_cmd : list):
        for i, q_dot in enumerate(q_dot_cmd):
            msg = Float64()
            msg.data = q_dot
            self.ros2_publishers[i].publish(msg)

    def move_arm_to_position(self, q_cmd :list):
        if len(q_cmd) == 0 or len(self.arm_positions) == 0 or len(self.arm_velocities) == 0:
            return False

        epsilon = 0.01
        error = 0
        q_d_cmd_list = []
        for i, q in enumerate(q_cmd):
            q_d_cmd_list.append(self.kp*(q-self.arm_positions[i]) - self.kd*self.arm_velocities[i])
            error += abs(q-self.arm_positions[i])

        if self.verbose:
            self.get_logger().info(f"Error: {error/len(q_cmd)}, Cmds: {q_d_cmd_list}", throttle_duration_sec=2.0)
        self.publish_arm_velocity_commands(q_d_cmd_list)

        if(error/len(q_cmd) < epsilon):
            self.publish_arm_velocity_commands([0.0 for x in q_cmd])
            #self.get_logger().debug(f"Done moving arm to {pos1}, {pos2}, {pos3}")
            return True
        else:
            return False


def main(args=None):
    rclpy.init(args=args)
    node = SimRemapper()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()