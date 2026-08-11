import rclpy
from rclpy.node import Node
import datetime

from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from std_msgs.msg import Int32

from geometry_msgs.msg import TwistStamped

from sensor_msgs.msg import JointState

from std_srvs.srv import SetBool

from feetech_ros2.srv import SetMode

from px4_msgs.msg import VehicleStatus
from px4_msgs.msg import VehicleLocalPosition
from px4_msgs.msg import TrajectorySetpoint
from px4_msgs.msg import OffboardControlMode


class MissionDirectorPy(Node):

    def __init__(self):
        super().__init__('mission_director_')

        # Parameters
        self.declare_parameter('frequency', 10.)

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Subscribe to manual input
        self.subscriber_input_state = self.create_subscription(
            Int32, 
            '/md/input', 
            self.input_state_callback, 
            10)
        self.subscriber_tactip = self.create_subscription(
            TwistStamped,
            '/tactip/pose',
            self.tactip_callback,
            10)
        # Initialize tactip data to zero
        self.tactip_data = TwistStamped()
        self.tactip_data.twist.linear.x = 0.0
        self.tactip_data.twist.linear.y = 0.0
        self.tactip_data.twist.linear.z = 0.0
        self.tactip_data.twist.angular.x = 0.0
        self.tactip_data.twist.angular.y = 0.0
        self.tactip_data.twist.angular.z = 0.0
        

        # Arms publishers
        self.publisher_arm = self.create_publisher(JointState, '/servo/in/references/joint_references', 10)
        self.declare_parameter('initial_joint_states', [0.0, 0.0, 0.0])

        # State transition times
        self.declare_parameter('entrypoint_time', 5.0)
        self.declare_parameter('position_arm_time', 5.0)
        self.declare_parameter('tactile_servoing_time', 5.0)

        # IK activation server
        self.ik_client = self.create_client(SetBool, 'activate_inverse_kinematics')
        self.ik_req = SetBool.Request()
        self.ik_active = False

        # Set mode server
        self.mode_client = self.create_client(SetMode, 'set_servo_mode')
        self.mode_req = SetMode.Request()
        self.operating_mode = 4

        while not self.ik_client.wait_for_service(timeout_sec=1.0): # and not self.mode_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('services not available, waiting again...')

        # set initial state
        self.FSM_state = 'entrypoint'
        self.input_state = 0
        self.first_state_loop = True
        self.state_start_time = datetime.datetime.now()

        # Timer -- always last
        self.timer_period = 1.0 / self.get_parameter('frequency').get_parameter_value().double_value
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

    def timer_callback(self):
        match self.FSM_state:
            case('entrypoint'):
                if self.first_state_loop:
                    self.get_logger().info('Starting FSM')
                    self.first_state_loop = False
                    self.publish_arm_position_commands(0.0, 0.0, 0.0) # Arm to home position



                # TODO How to know when servo driver is ready? Now we just wait -- implement going to position as action
                # Transition
                if datetime.datetime.now() - self.state_start_time > datetime.timedelta(seconds=self.get_parameter('entrypoint_time').get_parameter_value().double_value):
                    self.FSM_state = 'position_arm'
                    self.get_logger().info(f"Waited for 5 seconds -- switching to {self.FSM_state}")
                    self.state_start_time = datetime.datetime.now() # Reset state start time
                    self.first_state_loop = True # Reset first state loop flag

            case('position_arm'):
                if self.first_state_loop:
                    self.get_logger().info('Positioning arm')
                    self.first_state_loop = False
                    self.publish_arm_position_commands(1.3, 0.0, 2.8)

                # Transition
                if datetime.datetime.now() - self.state_start_time > datetime.timedelta(seconds=self.get_parameter('position_arm_time').get_parameter_value().double_value):
                    self.FSM_state = 'set_velocity_mode'
                    self.get_logger().info(f"Arm positioned -- switching to {self.FSM_state}")
                    self.state_start_time = datetime.datetime.now() # Reset state start time
                    self.first_state_loop = True # Reset first state loop flag
            
            case("set_velocity_mode"):
                if self.first_state_loop:
                    self.get_logger().info('Setting velocity mode')
                    self.first_state_loop = False
                    self.request_set_mode(1)
                    self.publish_arm_velocity_commands(0.0, 0.0, 0.0)
                
                # Transition
                if self.future.done():
                    self.FSM_state = 'wait_for_contact'
                    self.get_logger().info(f"Servo driver mode set to velocity mode -- switching to {self.FSM_state}")
                    self.state_start_time = datetime.datetime.now()
                    self.first_state_loop = True # Reset first state loop fla



            case('wait_for_contact'):
                if self.first_state_loop:
                    self.get_logger().info('Waiting for contact')
                    self.first_state_loop = False

                self.get_logger().info(f'Tactip depth: {self.tactip_data.twist.linear.z}')
                #self.get_logger().info(f'Tactip conditions: {self.tactip.data.twist.linear.z*1000 < -2.0}')
                # Transition
                # If contact depth is greater than 1.0 mm, we assume contact.
                if (self.tactip_data.twist.linear.z) < -2.0:
                    self.get_logger().info(f"Contact detected -- activating IK")
                    self.activate_ik()
                    self.request_set_mode(1) # 1 is velocity mode
                    self.get_logger().info('Contact detected, IK activated, servo velocity mode set -- switching to tactile servoing')
                    self.FSM_state = 'tactile_servoing'
                    self.state_start_time = datetime.datetime.now()
                    self.first_state_loop = True # Reset first state loop flag

            case('tactile_servoing'):
                if self.first_state_loop:
                    self.get_logger().info('Tactile servoing')
                    self.first_state_loop = False

                # Transition    
                if datetime.datetime.now() - self.state_start_time > datetime.timedelta(seconds=self.get_parameter('tactile_servoing_time').get_parameter_value().double_value):
                    self.deactivate_ik()
                    self.get_logger().info(f"Waited for 5 seconds and IK deactivated -- switching to end")
                    self.FSM_state = 'end'
                    self.state_start_time = datetime.datetime.now()
                    self.first_state_loop = True # Reset first state loop flag
            
            case('end'):
                if self.first_state_loop:
                    self.get_logger().info('End')
                    self.first_state_loop = False
                    self.request_set_mode(4) # 4 is continuous position mode
                
                
                self.publish_arm_position_commands(0.0, 0.0, 0.0) # Arm to home position

                if self.future.done():
                    self.get_logger().info('Mission completed')
                    self.get_logger().info('Shutting down')
                    rclpy.shutdown()
    
    def publish_arm_position_commands(self, q1, q2, q3):
        self.get_logger().debug(f'Publishing position references: ({q1}, {q2}, {q3})')
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ['q1', 'q2', 'q3']
        msg.position = [q1, q2, q3]
        msg.velocity = [0.0, 0.0, 0.0]
        msg.effort = [0.0, 0.0, 0.0]

        self.publisher_arm.publish(msg)

    def publish_arm_velocity_commands(self, q1, q2, q3):
        self.get_logger().debug(f'Publishing velocity references: ({q1}, {q2}, {q3})')
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ['q1', 'q2', 'q3']
        msg.position = [0.0, 0.0, 0.0]
        msg.velocity = [q1, q2, q3]
        msg.effort = [0.0, 0.0, 0.0]

        self.publisher_arm.publish(msg)
    
    def input_state_callback(self, msg):
        self.get_logger().info(f'Got input state: {msg.data}')
        self.input_state = msg.data

    def tactip_callback(self, msg):
        #self.get_logger().info(f'Tactip data: {msg}')
        self.tactip_data = msg

    def activate_ik(self):
        self.ik_req.data = True
        self.future = self.ik_client.call_async(self.ik_req)
        self.future.add_done_callback(self.activate_ik_callback)

        self.get_logger().info('Waiting for IK service call to complete')

    def activate_ik_callback(self, future):
        response = future.result()
        self.get_logger().info('Service call completed -- IK active')
        self.ik_active = True

    def deactivate_ik(self):
        self.ik_req.data = False
        self.future = self.ik_client.call_async(self.ik_req)
        self.future.add_done_callback(self.deactivate_ik_callback)

        self.get_logger().info('Waiting for IK service call to complete')

    def deactivate_ik_callback(self, future):
        response = future.result()
        self.get_logger().info('Service call completed -- IK deactivated')
        self.ik_active = False

    '''
    Request to set the servo driver mode. Mode 0 is default position mode, mode 1 is velocity mode, mode 4 is continuous position mode.
    '''
    def request_set_mode(self, mode):
        self.mode_req.operating_mode = mode
        self.future = self.mode_client.call_async(self.mode_req)
        self.future.add_done_callback(self.set_mode_callback)

        self.get_logger().info('Waiting for set mode service call to complete')

    def set_mode_callback(self, future):
        try:
            response = future.result()
            self.get_logger().info('Service call completed')
        except Exception as e:
            self.get_logger().info(f'Service call failed {e}')

def main():
    rclpy.init(args=None)

    mission_director_py = MissionDirectorPy()

    rclpy.spin(mission_director_py)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    mission_director_py.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()