import rclpy
from rclpy.node import Node
import numpy as np

from std_msgs.msg import Int32
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
    [12]: S1 dial - not used
    [13]: S2 dial - not used
    [14]: Left side dial - not used
    [15]: Right side dial - not used
    '''
    def __init__(self):
        super().__init__('ats_planner')

        # Parameters
        self.declare_parameter('frequency', 10.)
        self.declare_parameter('default_depth', 3.0) # default contact depth in mm
        self.declare_parameter('varying_refs', False) # RC channel for depth manipulation   
        self.declare_parameter('verbose', True)
        self.verbose = self.get_parameter('verbose').get_parameter_value().bool_value

        # Initialization log message
        self.get_logger().info("Planner node initialized")

        # MD sub
        self.sub_md = self.create_subscription(Int32, '/md/state', self.md_callback, 10)

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

        # Class data
        self.enable_reference_manipulation = False
        self.offboard = False
        self.tactile_servoing_active = False
        self.ts_time_elapsed = 0.0

        # Timer
        self.period = 1.0/float(self.get_parameter('frequency').get_parameter_value().double_value) # seconds
        self.timer = self.create_timer(self.period, self.timer_callback)

    '''
    Publish the reference end effector velocity
    '''
    def timer_callback(self):
        # Clean message
        reference_msg = TwistStamped()
        reference_msg.header.stamp = self.get_clock().now().to_msg()

        if self.enable_reference_manipulation and self.offboard: # If the right blue switch is on and we are in offboard mode
            reference_msg.twist.linear.x = (self.rc_input.channels[12])*3. # mm
            reference_msg.twist.linear.y = (self.rc_input.channels[13])*3. # mm
            reference_msg.twist.linear.z = self.get_parameter('default_depth').get_parameter_value().double_value + (self.rc_input.channels[0])*5. # mm
            reference_msg.twist.angular.x = (self.rc_input.channels[2])*25. # deg
            reference_msg.twist.angular.y = (self.rc_input.channels[3])*25. # deg
            reference_msg.twist.angular.z = 0.0 # deg
            if self.verbose:
                self.get_logger().info(f"Feeding RC to references: Depth: {(reference_msg.twist.linear.z):.3f} mm"
                                    f", Shear X: {reference_msg.twist.linear.x:.2f} mm, "
                                    f"Shear Y: {reference_msg.twist.linear.y:.2f} mm, "
                                    f"Pitch: {reference_msg.twist.angular.x:.2f} deg, "
                                    f"Roll: {reference_msg.twist.angular.y:.2f} deg", throttle_duration_sec=1.0)
        else:
            reference_msg.twist.linear.x = 0.0
            reference_msg.twist.linear.y = 0.0
            reference_msg.twist.linear.z = self.get_parameter('default_depth').get_parameter_value().double_value # mm
            reference_msg.twist.angular.x = 0.0
            reference_msg.twist.angular.y = 0.0
            reference_msg.twist.angular.z = 0.0

        # Modify the reference msg with time-based references
        if self.tactile_servoing_active and self.get_parameter('varying_refs').get_parameter_value().bool_value:
            self.ts_time_elapsed += self.period
            if self.ts_time_elapsed > 15.0 and self.ts_time_elapsed < 30.0:
                self.get_logger().info(f"Changing x angular reference to 15 deg from 0 deg after {self.ts_time_elapsed:.1f} seconds", once=True)
                reference_msg.twist.angular.x += 15.0 # deg
            elif self.ts_time_elapsed >= 30.0 and self.ts_time_elapsed < 45.0:
                self.get_logger().info(f"Changing y angular reference to 15 deg from 0 deg after {self.ts_time_elapsed:.1f} seconds", once=True)
                reference_msg.twist.angular.y += 15.0 # deg
            elif self.ts_time_elapsed >= 45.0 and self.ts_time_elapsed < 60.0:
                self.get_logger().info(f"Changing x angular reference to -15 deg from 0 deg after {self.ts_time_elapsed:.1f} seconds", once=True)
                reference_msg.twist.angular.x += 15.0
                reference_msg.twist.angular.y += 15.0
            elif self.ts_time_elapsed >= 60.0:
                self.get_logger().info(f"Returning angular references to 0 deg after {self.ts_time_elapsed:.1f} seconds", once=True)
                reference_msg.twist.angular.x = 0.0
                reference_msg.twist.angular.y = 0.0
                self.ts_time_elapsed = 0.0
        
        
        reference_msg.twist.linear.x = np.clip(reference_msg.twist.linear.x, -5.0, 5.0)
        reference_msg.twist.linear.y = np.clip(reference_msg.twist.linear.y, -5.0, 5.0)
        reference_msg.twist.linear.z = np.clip(reference_msg.twist.linear.z, -8.0, 8.0)
        reference_msg.twist.angular.x = np.clip(reference_msg.twist.angular.x, -25.0, 25.0)
        reference_msg.twist.angular.y = np.clip(reference_msg.twist.angular.y, -25.0, 25.0)
        reference_msg.twist.angular.z = np.clip(reference_msg.twist.angular.z, -25.0, 25.0)
        self.ee_velocity_publisher_.publish(reference_msg)

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

    def md_callback(self, msg):
        if msg.data == 30: # If in MD state tactile_servoing
            self.tactile_servoing_active = True
        else:
            self.tactile_servoing_active = False
            self.ts_time_elapsed = 0.0

def main(args=None):
    rclpy.init(args=args)
    node = ATSPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()