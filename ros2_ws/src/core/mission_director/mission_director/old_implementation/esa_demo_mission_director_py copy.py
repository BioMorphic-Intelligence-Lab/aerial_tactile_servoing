import rclpy
from rclpy.node import Node
import datetime

from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from std_msgs.msg import Int32

from geometry_msgs.msg import TwistStamped

from px4_msgs.msg import VehicleStatus
from px4_msgs.msg import VehicleLocalPosition
from px4_msgs.msg import TrajectorySetpoint
from px4_msgs.msg import OffboardControlMode


class MissionDirectorPy(Node):

    def __init__(self):
        super().__init__('mission_director_py')

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.subscriber_vehicle_status = self.create_subscription(
            VehicleStatus, 
            '/fmu/out/vehicle_status', 
            self.vehicle_status_callback, 
            qos_profile)
        self.subscriber_vehicle_local_position = self.create_subscription(
            VehicleLocalPosition, 
            '/fmu/out/vehicle_local_position', 
            self.vehicle_local_position_callback, 
            qos_profile)
        
        self.subscriber_input_state = self.create_subscription(
            Int32, 
            '/md/input_state', 
            self.input_state_callback, 
            10)
        
        # Position setpoint publishers
        self.publisher_vehicle_trajectory_setpoint = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', 10)
        self.publisher_offboard_control_mode = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode',10)

        # Arms publishers
        self.publisher_arm = self.create_publisher(TwistStamped, '/servo/in/reference_position', 10)

        self.declare_parameter('frequency', 10.)
        self.timer_period = 1.0 / self.get_parameter('frequency').get_parameter_value().double_value
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

        # set initial state
        self.FSM_state = 'entrypoint'

        # Initialize vehicle data
        self.vehicle_status = VehicleStatus()
        self.vehicle_local_position = VehicleLocalPosition()

        # Disarmed data
        self.armed = False
        self.offboard = False

        # Takeoff data
        self.takeoff_altitude = -1.0

        # Hover data
        self.state_start_time = None
        self.declare_parameter('hover_duration', 10.)
        self.hover_time = self.get_parameter('hover_duration').get_parameter_value().double_value

        self.declare_parameter('landing_velocity', 0.1)
        self.landing_velocity = self.get_parameter('landing_velocity').get_parameter_value().double_value
        self.previous_next_landing_altitude = 1.1

        self.x_setpoint = 0.0
        self.y_setpoint = 0.0

        self.input_state = 0

    def timer_callback(self):
        match self.FSM_state:
            case('entrypoint'):
                self.x_setpoint = self.vehicle_local_position.x
                self.y_setpoint = self.vehicle_local_position.y
                self.get_logger().info("Waiting for position fix")
                if self.x_setpoint != 0.0 and self.y_setpoint != 0.0:
                    self.get_logger().info(f"Got position fix X: {self.x_setpoint} Y: {self.y_setpoint}")
                    self.FSM_state = 'disarmed'

            case('disarmed'):
                self.publishOffboardControlMode()
                self.publishTrajectorySetpoint(0.0, 0.0, 0.0, 0.0) # Start publishing messages to px4

                if self.armed and self.offboard:
                    self.get_logger().info('Armed and offboard -- switching to takeoff')
                    self.FSM_state = 'takeoff'
                    self.state_start_time = datetime.datetime.now()
                elif self.armed and not self.offboard:
                    self.get_logger().info('Armed but not offboard -- waiting')
                elif not self.armed and self.offboard:
                    self.get_logger().info('Not armed but offboard -- waiting')
                elif self.input_state == 3:
                    self.get_logger().info('Input state is 3 -- manually transitioning to start_hover')
                    self.FSM_state = 'start_hover'
                    self.state_start_time = datetime.datetime.now()
                else:
                    self.get_logger().info('Waiting for arming and offboard')

                

            case('takeoff'):
                # get current vehicle altitude
                current_altitude = self.vehicle_local_position.z

                # create and publish setpoint message
                self.publishOffboardControlMode()
                # send takeoff command
                self.publishTrajectorySetpoint(self.x_setpoint, self.y_setpoint, self.takeoff_altitude, 0.0)

                # Raise arms slightly on takeoff
                self.publishPositionReferencesArm(0.2, 0.0, 0.0, 0.2, 0.0, 0.0)
                
                # check if vehicle has reached takeoff altitude
                if abs(current_altitude)+0.1 > abs(self.takeoff_altitude) and not self.input_state==1:
                    self.get_logger().info('Reached takeoff altitude -- switching to hover')
                    self.FSM_state = 'start_hover'
                    self.state_start_time = datetime.datetime.now()
                if (self.input_state==5):
                    self.get_logger().info('Manually triggered -- switching to hover')
                    self.FSM_state = 'start_hover'
                    self.state_start_time = datetime.datetime.now()                    
                elif (self.input_state == 1):
                    self.get_logger().info(f'Input state is 1 -- switching to end hover')
                    self.FSM_state = 'end_hover'
                    self.state_start_time = datetime.datetime.now()
                elif (self.input_state == 3):
                    self.get_logger().info(f'Input state is 3 -- manually transitioning to start_hover')
                    self.FSM_state = 'start_hover'
                    self.state_start_time = datetime.datetime.now()

            case('start_hover'):
                # create and publish setpoint message
                self.publishOffboardControlMode()
                self.publishTrajectorySetpoint(self.x_setpoint, self.y_setpoint, self.takeoff_altitude, 0.0)

                if (datetime.datetime.now() - self.state_start_time).seconds > self.hover_time and not self.input_state==1:
                    self.get_logger().info(f'Hovered for {self.hover_time} seconds -- switching to Y')
                    self.FSM_state = 'Y'
                    self.state_start_time = datetime.datetime.now()
                elif (self.input_state == 1):
                    self.get_logger().info(f'Input state is 1 -- switching to end hover')
                    self.FSM_state = 'end_hover'
                    self.state_start_time = datetime.datetime.now()

            case('Y'):
                # create and publish setpoint message
                self.publishOffboardControlMode()
                self.publishTrajectorySetpoint(self.x_setpoint, self.y_setpoint, self.takeoff_altitude, 0.0)

                # Angles associated with Y
                self.publishPositionReferencesArm(
                    1.0, 0.0, 1.6,
                    1.0, 0.0, -1.6)

                if (datetime.datetime.now() - self.state_start_time).seconds > self.hover_time and not self.input_state==1:
                    self.get_logger().info(f'Y for {self.hover_time} seconds -- switching to M')
                    self.FSM_state = 'M'
                    self.state_start_time = datetime.datetime.now()
                elif (self.input_state == 1):
                    self.get_logger().info(f'Input state is 1 -- switching to end hover')
                    self.FSM_state = 'end_hover'
                    self.state_start_time = datetime.datetime.now()

            case('M'):
                # create and publish setpoint message
                self.publishOffboardControlMode()
                self.publishTrajectorySetpoint(self.x_setpoint, self.y_setpoint, self.takeoff_altitude, 0.0)

                # Angles associated with M
                self.publishPositionReferencesArm(
                    0.7, 0.0, 3.6,
                    0.7, 0.0, -3.6)

                if (datetime.datetime.now() - self.state_start_time).seconds > self.hover_time and not self.input_state==1:
                    self.get_logger().info(f'M for {self.hover_time} seconds -- switching to C')
                    self.FSM_state = 'C'
                    self.state_start_time = datetime.datetime.now()
                elif (self.input_state == 1):
                    self.get_logger().info(f'Input state is 1 -- switching to end hover')
                    self.FSM_state = 'end_hover'
                    self.state_start_time = datetime.datetime.now()

            case('C'):
                # create and publish setpoint message
                self.publishOffboardControlMode()
                self.publishTrajectorySetpoint(self.x_setpoint, self.y_setpoint, self.takeoff_altitude, 0.0)

                # Angles associated with C
                self.publishPositionReferencesArm(
                    1.3, 0.0, 3.0, # 0.7, 0.0, -0.8
                    -1.6, 0.0, -3.0) 

                if (datetime.datetime.now() - self.state_start_time).seconds > self.hover_time and not self.input_state==1:
                    self.get_logger().info(f'C for {self.hover_time} seconds -- switching to A')
                    self.FSM_state = 'A'
                    self.state_start_time = datetime.datetime.now()
                elif (self.input_state == 1):
                    self.get_logger().info(f'Input state is 1 -- switching to end hover')
                    self.FSM_state = 'end_hover'
                    self.state_start_time = datetime.datetime.now()

            case('A'):
                # create and publish setpoint message
                self.publishOffboardControlMode()
                self.publishTrajectorySetpoint(self.x_setpoint, self.y_setpoint, self.takeoff_altitude, 0.0)

                # Angles associated with A
                self.publishPositionReferencesArm(
                    0.8, 0.0, 0.1,
                    0.8, 0.0, -0.1)

                if (datetime.datetime.now() - self.state_start_time).seconds > self.hover_time and self.input_state==0:
                    self.get_logger().info(f'A for {self.hover_time} seconds -- switching to circle_start')
                    self.FSM_state = 'circle_start'
                    self.state_start_time = datetime.datetime.now()
                
                elif (self.input_state == 1):
                    self.get_logger().info(f'Input state is 1 -- switching to end hover')
                    self.FSM_state = 'end_hover'
                    self.state_start_time = datetime.datetime.now()
                
                elif (self.input_state == 2):
                    self.get_logger().info(f'Input state is 2 -- repeating')
                    self.FSM_state = 'Y'
                    self.state_start_time = datetime.datetime.now()
                

            case('circle_start'):
                # create and publish setpoint message
                self.publishOffboardControlMode()
                self.publishTrajectorySetpoint(self.x_setpoint, self.y_setpoint, self.takeoff_altitude, 0.0)

                # Back to home position
                self.publishPositionReferencesArm(
                    1.5, 0.0, 0.2, 
                    -1.3, 0.0, -3.3)
                if (datetime.datetime.now() - self.state_start_time).seconds > self.hover_time:
                    self.get_logger().info(f'A for {self.hover_time} seconds -- switching to executing circle manoeuvre')
                    self.FSM_state = 'circle_exec'
                    self.state_start_time = datetime.datetime.now()
                elif (self.input_state == 1):
                    self.get_logger().info(f'Input state is 1 -- switching to end hover')
                    self.FSM_state = 'end_hover'
                    self.state_start_time = datetime.datetime.now()     

            case('circle_exec'):
                # create and publish setpoint message
                self.publishOffboardControlMode()
                self.publishTrajectorySetpoint(self.x_setpoint, self.y_setpoint, self.takeoff_altitude, 0.0)

                # Back to home position
                self.publishPositionReferencesArm(
                    -1.3, 0.0, 0.2, 
                    1.5, 0.0, -3.3)
                if (datetime.datetime.now() - self.state_start_time).seconds > self.hover_time:
                    self.get_logger().info(f'A for {self.hover_time} seconds -- switching to end_hover')
                    self.FSM_state = 'end_hover'
                    self.state_start_time = datetime.datetime.now()    

            case('end_hover'):
                # create and publish setpoint message
                self.publishOffboardControlMode()
                self.publishTrajectorySetpoint(self.x_setpoint, self.y_setpoint, self.takeoff_altitude, 0.0)

                # Back to home position
                self.publishPositionReferencesArm(
                    0.2, 0.0, 0.0, 
                    0.2, 0.0, 0.0)


                if (datetime.datetime.now() - self.state_start_time).seconds > self.hover_time:
                    self.get_logger().info(f'End hover for {self.hover_time} seconds -- switching to land')
                    self.FSM_state = 'land'

            case('land'):
                self.get_logger().debug('Landing')
                # Calculate next landing altitude (plus because z down positive)
                next_landing_altitude = self.vehicle_local_position.z + self.landing_velocity*self.timer_period
                # create and publish setpoint message
                self.publishOffboardControlMode()
                self.publishTrajectorySetpoint(self.x_setpoint, self.y_setpoint, next_landing_altitude, 0.0)
                if abs(self.previous_next_landing_altitude - next_landing_altitude) < 0.001:
                    self.get_logger().info('Going to landed state')
                    self.FSM_state = 'landed'

                self.previous_next_landing_altitude = next_landing_altitude
            
            case('landed'):
                self.get_logger().info('Landed -- please disarm')


    def publishOffboardControlMode(self):
        msg = OffboardControlMode()
        msg.timestamp = int(self.get_clock().now().nanoseconds/1000)
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        self.publisher_offboard_control_mode.publish(msg)
    
    def publishTrajectorySetpoint(self, x, y, z, yaw):
        msg = TrajectorySetpoint()
        msg.position[0] = x
        msg.position[1] = y
        msg.position[2] = z
        msg.yaw = yaw
        msg.timestamp = int(self.get_clock().now().nanoseconds/1000)
        self.publisher_vehicle_trajectory_setpoint.publish(msg)
    
    def publishPositionReferencesArm(self, q11, q12, q13, q21, q22, q23):
        self.get_logger().debug(f'Publishing position references: ({q11}, {q12}, {q13}) ({q21}, {q22}, {q23})')
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.twist.linear.x = q11
        msg.twist.linear.y = q12
        msg.twist.linear.z = q13

        msg.twist.angular.x = q21
        msg.twist.angular.y = q22
        msg.twist.angular.z = q23

        self.publisher_arm.publish(msg)

    def vehicle_status_callback(self, msg):
        self.get_logger().debug(f'Received vehicle_status')
        if (msg.arming_state == msg.ARMING_STATE_DISARMED):
            self.armed = False
        elif (msg.arming_state == msg.ARMING_STATE_ARMED):
            self.armed = True
        
        if (msg.nav_state == msg.NAVIGATION_STATE_OFFBOARD):
            self.offboard=True
        else:
            self.offboard=False
    
    def vehicle_local_position_callback(self, msg):
        self.get_logger().debug(f'Received vehicle_local_position: {msg.z}')
        self.vehicle_local_position = msg
    
    def input_state_callback(self, msg):
        self.get_logger().info(f'Got input state: {msg.data}')
        self.input_state = msg.data


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