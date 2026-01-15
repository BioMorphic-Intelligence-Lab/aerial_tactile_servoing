import rclpy
from rclpy.node import Node
import numpy as np

from geometry_msgs.msg import TwistStamped
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from px4_msgs.msg import RcChannels # normalized values

class ATSPlanner(Node):
    '''
    Aerial Tactile Servoing Planner node. This node publishes pose references for the tactile sensor to follow.
    The references are defined in the contact frame, with Z perpendicular to the interaction surface. THe surface_3d 
    Tactip feedback only allows tracking contact depth (z position), roll (around x axis), and pitch (around y axis).
    Setting the other references nonzero will induce a movement in that direction as the feedback is always zero.

    Channel map
    [0]: Yaw
    [1]: Throttle
    [2]: Pitch
    [3]: Roll
    [4]: Left silver 3 position switch - not used
    [5]: Left green 3 position switch - Arm: 1 is armed, -1 is disarmed
    [6]: Right yellow 3 position switch - Flight mode: 1 is position, 0 is stabilized, -1 is altitude
    [7]: Right white 3 position switch - Offboard mode: 1 is offboard, -1 is manual
    [8]: Left black switch - not used
    [9]: Left red 2 position switch - Kill: 1 is killed, -1 is alive
    [10]: Right blue 3 position switch - not used
    [11]: Right pink 2 position button switch - not used
    [12]: S1 switch - not used
    [13]: S2 switch - not used
    [14]: Left side dial - not used
    [15]: Right side dial - not used
    '''
    def __init__(self):
        super().__init__('ats_planner')

        # Parameters
        self.declare_parameter('frequency', 10.)
        self.declare_parameter('default_depth', 3.0) # default contact depth in mm
        self.declare_parameter('verbose', True)
        self.verbose = self.get_parameter('verbose').get_parameter_value().bool_value

        # Initialization log message
        self.get_logger().info("Planner node initialized")

        # Publishers
        self.ee_velocity_publisher_ = self.create_publisher(TwistStamped, '/references/sensor_in_contact', 10)

        # Flight controller interfaces
        px4_qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.sub_rc_channels = self.create_subscription(RcChannels, '/fmu/out/rc_channels', self.rc_callback, px4_qos_profile)
        self.rc_input = RcChannels()

        self.reference_msg = TwistStamped()
        self.reference_msg.twist.linear.x = 0.0
        self.reference_msg.twist.linear.y = 0.0
        self.reference_msg.twist.linear.z = self.get_parameter('default_depth').get_parameter_value().double_value/1000. # m
        self.reference_msg.twist.angular.x = 0.0
        self.reference_msg.twist.angular.y = 0.0
        self.reference_msg.twist.angular.z = 0.0

        # Class data
        self.enable_reference_manipulation = False
        self.offboard = False

        # Timer
        self.period = 1.0/float(self.get_parameter('frequency').get_parameter_value().double_value) # seconds
        self.timer = self.create_timer(self.period, self.timer_callback)

    '''
    Publish the reference end effector velocity
    '''
    def timer_callback(self):
        self.reference_msg.header.stamp = self.get_clock().now().to_msg()
        if self.enable_reference_manipulation and self.offboard: # If the right blue switch is on and we are in offboard mode
            self.reference_msg.twist.linear.z = self.get_parameter('default_depth').get_parameter_value().double_value/1000. + (self.rc_input.channels[0])*0.005 # m
            self.reference_msg.twist.angular.x = (self.rc_input.channels[2])*0.5 # rad
            self.reference_msg.twist.angular.y = (self.rc_input.channels[3])*0.5 # rad
            if self.verbose:
                self.get_logger().info(f"Feeding RC to references: Depth: {(self.reference_msg.twist.linear.z*1000.):.3f} mm, "
                                    f"Pitch: {np.rad2deg(self.reference_msg.twist.angular.x):.3f} deg, "
                                    f"Roll: {np.rad2deg(self.reference_msg.twist.angular.y):.3f} deg", throttle_duration_sec=1.0)
        else:
            self.reference_msg.twist.linear.z = self.get_parameter('default_depth').get_parameter_value().double_value/1000. # m
            self.reference_msg.twist.angular.x = 0.0
            self.reference_msg.twist.angular.y = 0.0
        self.ee_velocity_publisher_.publish(self.reference_msg)

    def rc_callback(self, msg):
        self.rc_input = msg
        if msg.channels[10] > 0.5: # If right blue switch is on -> towards you
            self.enable_reference_manipulation = True
        elif msg.channels[10] < -0.5: # If right blue switch is off -> away from you
            self.enable_reference_manipulation = False
        
        if msg.channels[7] > 0.5: # If right white switch is on -> offboard
            self.offboard = True
        elif msg.channels[7] < -0.5: # If right white switch is off -> manual
            self.offboard = False

def main(args=None):
    rclpy.init(args=args)
    node = ATSPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()