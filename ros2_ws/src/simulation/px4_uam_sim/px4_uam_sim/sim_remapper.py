import rclpy
from rclpy.node import Node

from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from std_msgs.msg import Float64
from sensor_msgs.msg import JointState

import numpy as np

class SimRemapper(Node):
    def __init__(self):
        super().__init__('sim_remapper')

        self.declare_parameter('frequency', 25.)

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Arm subscribers
        self.subscriber_servo_state = self.create_subscription(JointState, '/servo/in/state', self.servo_state_callback, 10)
        self.subscriber_joint_states = self.create_subscription(JointState, '/servo/out/state', self.joint_states_callback, 10)
        # Arms publishers
        self.publisher_pivot_vel = self.create_publisher(Float64, '/shoulder_1_vel_cmd', 10) # TODO: Change to interface with Feetech ROS2 driver 
        self.publisher_shoulder_vel = self.create_publisher(Float64, '/elbow_1_vel_cmd', 10)
        self.publisher_elbow_vel = self.create_publisher(Float64, '/forearm_1_vel_cmd', 10)

        self.q1_cmd = 0.0
        self.q2_cmd = 0.0
        self.q3_cmd = 0.0

        self.arm_positions = [0.0, 0.0, 0.0]
        self.arm_velocities = [0.0, 0.0, 0.0]
        self.kp = 2.
        self.kd = 0.3

        # Timer -- always last
        self.counter = 0
        self.frequency = self.get_parameter('frequency').get_parameter_value().double_value
        self.timer_period = 1.0 / self.frequency
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
    
    def timer_callback(self):
        self.move_arm_to_position(self.q1_cmd, self.q2_cmd, self.q3_cmd)

    def joint_states_callback(self, msg):
        self.arm_positions[0] = msg.position[0] # Pivot
        self.arm_positions[1] = msg.position[1] # Shoulder
        self.arm_positions[2] = msg.position[2] # Elbow

        self.arm_velocities[0] = msg.velocity[0] # Pivot
        self.arm_velocities[1] = msg.velocity[1] # Shoulder
        self.arm_velocities[2] = msg.velocity[2] # Elbow

    def servo_state_callback(self, msg):
        self.q1_cmd = msg.position[0]
        self.q2_cmd = msg.position[1]
        self.q3_cmd = msg.position[2]
        
    def publish_arm_velocity_commands(self, q1, q2, q3):
        msg = Float64()
        msg.data = q1
        self.publisher_pivot_vel.publish(msg)
        msg.data = q2
        self.publisher_shoulder_vel.publish(msg)
        msg.data = q3
        self.publisher_elbow_vel.publish(msg)

    def move_arm_to_position(self, pos1, pos2, pos3):
        epsilon = 0.01

        error1 = pos1-self.arm_positions[0]
        error2 = pos2-self.arm_positions[1]
        error3 = pos3-self.arm_positions[2]
        vel1 = self.kp*(error1) - self.kd*self.arm_velocities[0]
        vel2 = self.kp*(error2) - self.kd*self.arm_velocities[1]
        vel3 = self.kp*(error3) - self.kd*self.arm_velocities[2]
        self.publish_arm_velocity_commands(vel1, vel2, vel3)

        if(abs(error1) < epsilon and abs(error2) < epsilon and abs(error3) < epsilon):
            self.publish_arm_velocity_commands(0.0, 0.0, 0.0)
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