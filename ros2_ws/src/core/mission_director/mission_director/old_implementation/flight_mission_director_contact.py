import rclpy
from rclpy.node import Node
import datetime

from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from std_msgs.msg import Int32, Int8
from std_msgs.msg import Float64

from numpy import pi, clip

from geometry_msgs.msg import TwistStamped

from sensor_msgs.msg import JointState

from px4_msgs.msg import VehicleStatus
from px4_msgs.msg import VehicleOdometry
from px4_msgs.msg import VehicleLocalPosition
from px4_msgs.msg import TrajectorySetpoint
from px4_msgs.msg import OffboardControlMode
from px4_msgs.msg import VehicleCommand


class MissionDirectorPy(Node):

    def __init__(self):
        super().__init__('mission_director_py')

        # Parameters
        self.declare_parameter('frequency', 10.)
        self.declare_parameter('takeoff_altitude', 2.0)
        self.declare_parameter('landing_velocity', -0.5)
        self.declare_parameter('search_velocity', -0.3)
        self.declare_parameter('hover_time', 10.0)
        self.declare_parameter('position_clip', 0.0)

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Position setpoint publishers
        self.publisher_vehicle_trajectory_setpoint = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', 10)
        self.publisher_offboard_control_mode = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode',10)
        self.publisher_vehicle_command = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', 10)

        self.publisher_md_state = self.create_publisher(Int32, '/md/state', 10)

        # PX4 subscribers
        self.subscriber_vehicle_status = self.create_subscription(VehicleStatus, '/fmu/out/vehicle_status_v1', self.vehicle_status_callback, qos_profile)
        self.subscriber_vehicle_odometry = self.create_subscription(VehicleOdometry, '/fmu/out/vehicle_odometry', self.vehicle_odometry_callback, qos_profile)
        self.subscriber_vehicle_local_position = self.create_subscription(VehicleLocalPosition, '/fmu/out/vehicle_local_position', self.vehicle_local_position_callback, qos_profile)

        # Controller subscriber
        self.subscriber_controller = self.create_subscription(TrajectorySetpoint, '/controller/out/trajectory_setpoint', self.controller_callback, 10)
        self.subscriber_controller_servo = self.create_subscription(JointState, '/controller/out/servo_positions', self.controller_servo_callback, 10)
        
        # GZ subscribers
        self.subscriber_joint_states = self.create_subscription(JointState, '/servo/out/state', self.joint_states_callback, 10)
        
        # Subscribe to manual input
        self.subscriber_input_state = self.create_subscription(Int32, '/md/input', self.input_state_callback, 10)
        self.subscriber_tactip = self.create_subscription(TwistStamped, '/tactip/pose', self.tactip_callback, 10)
        self.subscriber_tactip_contact = self.create_subscription(Int8, '/tactip/contact', self.tactip_contact_callback, 10)
        
        # Initialize tactip data to zero
        self.tactip_data = TwistStamped()
        self.tactip_data.twist.linear.x = 0.0
        self.tactip_data.twist.linear.y = 0.0
        self.tactip_data.twist.linear.z = 0.0
        self.tactip_data.twist.angular.x = 0.0
        self.tactip_data.twist.angular.y = 0.0
        self.tactip_data.twist.angular.z = 0.0

        # Arms publishers
        self.publisher_servo_state = self.create_publisher(JointState, '/servo/in/state', 10)

        # set initial state
        self.FSM_state = 'entrypoint'
        self.input_state = 0
        self.first_state_loop = True
        self.state_start_time = datetime.datetime.now()
        self.armed = False
        self.offboard = False
        self.killed = False
        self.contact = False
        self.hover_time = self.get_parameter('hover_time').get_parameter_value().double_value
        self.takeoff_altitude = self.get_parameter('takeoff_altitude').get_parameter_value().double_value
        self.landing_velocity = self.get_parameter('landing_velocity').get_parameter_value().double_value
        self.search_velocity = self.get_parameter('search_velocity').get_parameter_value().double_value
        self.previous_next_landing_altitude = abs(self.takeoff_altitude)+0.1
        self.position_clip = self.get_parameter('position_clip').get_parameter_value().double_value
        self.x_setpoint = 0.0
        self.y_setpoint = 0.0

        self.arm_positions = [0.0, 0.0, 0.0]
        self.arm_velocities = [0.0, 0.0, 0.0]

        # Initialize vehicle data
        self.vehicle_status = VehicleStatus()
        self.vehicle_odometry = VehicleOdometry()
        self.vehicle_local_position = VehicleLocalPosition()
        self.vehicle_trajectory_setpoint = TrajectorySetpoint()

        self.servo_reference = JointState()

        # Simulation stuff
        self.servo_position = True

        # Timer -- always last
        self.counter = 0
        self.frequency = self.get_parameter('frequency').get_parameter_value().double_value
        self.timer_period = 1.0 / self.frequency
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

    def __del__(self):
        self.move_arm_to_position(0., 0., 0.,)

    def timer_callback(self):
        match self.FSM_state:
            case('entrypoint'): # Entry point - wait for position fix
                if self.first_state_loop:
                    self.get_logger().info("Waiting for position fix")
                    self.first_state_loop = False

                self.publishMDState(0)
                self.x_setpoint = self.vehicle_local_position.x
                self.y_setpoint = self.vehicle_local_position.y
                self.publishOffboardPositionMode()
                if (self.x_setpoint != 0.0 and self.y_setpoint != 0.0) or self.input_state == 1:
                    self.get_logger().info(f'Got position fix! \t x: {self.x_setpoint} \t y: {self.y_setpoint}')
                    self.transition_state(new_state='wait_for_servo_driver')

            case('wait_for_servo_driver'):
                if (datetime.datetime.now() - self.state_start_time).seconds > 2 or self.input_state == 1:
                    self.transition_state('move_arm_landed')

            case('move_arm_landed'):
                self.move_arm_to_position(pi/2, 0.0, -1.85)
                self.publishMDState(1)
                 # Wait 5 seconds until the arm is in position
                if (datetime.datetime.now() - self.state_start_time).seconds > 2 or self.input_state == 1:
                    self.transition_state(new_state='extend_arm')
                    
            case('extend_arm'):
                self.move_arm_to_position(pi/2, 0.0, 0.0)
                self.publishMDState(1)
                 # Wait 5 seconds until the arm is in position
                if (datetime.datetime.now() - self.state_start_time).seconds > 3 or self.input_state == 1:
                    self.transition_state(new_state='move_arm_landed2')

            case('move_arm_landed2'):
                self.move_arm_to_position(pi/2, 0.0, -1.85)
                self.publishMDState(1)
                 # Wait 5 seconds until the arm is in position
                if (datetime.datetime.now() - self.state_start_time).seconds > 2 or self.input_state == 1:
                    self.transition_state(new_state='wait_for_arm_offboard')

            case('wait_for_arm_offboard'):
                self.move_arm_to_position(pi/2, 0.0, -1.85)
                self.publishOffboardPositionMode()
                self.publishMDState(3)
                if self.armed and not self.offboard:
                    self.get_logger().info('Armed but not offboard -- waiting')
                elif not self.armed and self.offboard:
                    self.get_logger().info('Not armed but offboard -- waiting')
                elif self.armed and self.offboard:
                    self.transition_state('takeoff')

            case('sim_arm_offboard'):
                if not self.offboard and self.counter%self.frequency==0:
                    self.get_logger().info("Sending offboard command")
                    self.engage_offboard_mode()
                    self.counter = 0
                if not self.armed and self.offboard and self.counter%self.frequency==0:
                    self.get_logger().info("Sending arm command")
                    self.armVehicle()
                    self.counter = 0

                self.publishOffboardPositionMode()
                self.publishMDState(3)
                if self.armed and self.offboard:
                    self.transition_state('takeoff')

            case('takeoff'): # Takeoff - wait for takeoff altitude
                self.publishMDState(4)
                self.move_arm_to_position(pi/2, 0.0, -1.85)
                # get current vehicle altitude
                current_altitude = self.vehicle_local_position.z

                self.publishOffboardPositionMode()
                self.publishTrajectoryPositionSetpoint(self.x_setpoint, self.y_setpoint, self.takeoff_altitude, self.vehicle_local_position.heading)
                if self.first_state_loop:
                    self.get_logger().info(f'Vehicle local position heading: {self.vehicle_local_position.heading}')
                    self.get_logger().info(f'Takeoff altitude: {self.takeoff_altitude}')
                    self.first_state_loop = False
                
                # check if vehicle has reached takeoff altitude
                if abs(current_altitude)+0.1 > abs(self.takeoff_altitude) or self.input_state==1:
                    self.transition_state('hover')

            case('hover'):
                self.move_arm_to_position(pi/2, 0.0, -1.85)
                self.publishMDState(5)
                # create and publish setpoint message
                self.publishOffboardPositionMode()
                self.publishTrajectoryPositionSetpoint(self.x_setpoint, self.y_setpoint, self.takeoff_altitude, 0.0)

                if (datetime.datetime.now() - self.state_start_time).seconds > self.hover_time or self.input_state==1:
                    self.transition_state('move_arm_for_ats')
                elif not self.offboard or self.killed or self.input_state == 2:
                    self.transition_state('emergency')

            case('move_arm_for_ats'):
                self.publishMDState(6)
                # Stay in place
                self.publishOffboardPositionMode()
                self.publishTrajectoryPositionSetpoint(self.x_setpoint, self.y_setpoint, self.takeoff_altitude, 0.0)
                self.move_arm_to_position(pi/3, 0.0, pi/6)
                self.y_forward_setpoint = self.y_setpoint

                # Transition
                if (datetime.datetime.now() - self.state_start_time).seconds > 2.5 or self.input_state==1:
                    self.transition_state('look_for_contact')
                elif not self.offboard or self.killed or self.input_state == 2:
                    self.transition_state('emergency')

            case('look_for_contact'):
                self.publishMDState(7)
                self.publishOffboardPositionMode()
                self.move_arm_to_position(pi/3, 0.0, pi/6)
                self.y_forward_setpoint = self.y_forward_setpoint + self.search_velocity*self.timer_period # fly forward
                self.publishTrajectoryPositionSetpoint(self.x_setpoint, self.y_forward_setpoint, self.takeoff_altitude, 0.0)
                
                if self.counter % self.frequency:
                    self.get_logger().info(f'Tactip depth: {self.tactip_data.twist.linear.z} mm')
                
                if self.contact: 
                    self.transition_state('contact')
                elif not self.offboard or self.killed or self.input_state == 2:
                    self.transition_state('emergency')

            case('contact'):
                self.publishMDState(8)
                self.publishOffboardPositionMode()
                self.publishTrajectoryPositionSetpoint(
                    self.vehicle_trajectory_setpoint.position[0], 
                    self.vehicle_trajectory_setpoint.position[1], 
                    self.vehicle_trajectory_setpoint.position[2], 
                    self.vehicle_trajectory_setpoint.yaw,
                    self.vehicle_trajectory_setpoint.yawspeed)
                self.move_arm_to_position(
                    self.servo_reference.position[0],
                    self.servo_reference.position[1],
                    self.servo_reference.position[2]
                )
                if self.first_state_loop:
                    self.get_logger().info('Tactile servoing')
                    self.first_state_loop = False

                # Transition 1: Lost contact
                #if self.tactip_data.twist.linear.z > self.contact_depth_threshold:
                #    self.transition_state('look_for_contact')
                elif (datetime.datetime.now() - self.state_start_time).seconds > 150. or self.input_state==1:
                    self.transition_state('land')
                elif not self.offboard or self.killed or self.input_state == 2:
                    self.transition_state('emergency')

            case('land'):
                self.move_arm_to_position(0.0, 0.0, 0.0)
                self.publishMDState(9)
                self.get_logger().debug('Landing')
                next_landing_altitude = self.vehicle_local_position.z - self.landing_velocity*self.timer_period
                self.publishOffboardPositionMode()
                self.land()
                if abs(self.previous_next_landing_altitude - next_landing_altitude) < 0.001:
                    self.transition_state(new_state='landed')

                self.previous_next_landing_altitude = next_landing_altitude

            case('landed'):
                self.publishMDState(10)
                self.get_logger().info('Done')
                self.disarmVehicle()

            case('emergency'):
                self.move_arm_to_position(1.578, 0.0, -1.85)
                self.publishMDState(-1)
                if self.counter% (2*self.frequency) == 0: # Publish message every 2 seconds
                    self.get_logger().warn("Emergency state - no offboard mode")
                    self.counter = 0
                self.counter +=1

    def publish_arm_position_commands(self, q1, q2, q3):
        msg = JointState()
        msg.position = [q1, q2, q3]
        msg.velocity = [0., 0., 0.]
        msg.name = ['q1', 'q2', 'q3']
        msg.header.stamp = self.get_clock().now().to_msg()
        self.publisher_servo_state.publish(msg)

    def publish_arm_velocity_commands(self, q1, q2, q3):
        msg = JointState()
        msg.position = [0., 0., 0.]
        msg.velocity = [q1, q2, q3]
        msg.name = ['q1', 'q2', 'q3']
        msg.header.stamp = self.get_clock().now().to_msg()
        self.publisher_servo_state.publish(msg)

    def input_state_callback(self, msg):
        self.get_logger().info(f'Got input state: {msg.data}')
        self.input_state = msg.data

    def tactip_callback(self, msg):
        self.tactip_data = msg

    def controller_callback(self, msg):
        self.vehicle_trajectory_setpoint = msg

    def controller_servo_callback(self, msg):
        self.servo_reference = msg
    
    def tactip_contact_callback(self, msg):
        self.contact = msg.data

    def move_arm_to_position(self, q1, q2, q3):
        self.publish_arm_position_commands(q1, q2, q3)

    def publishMDState(self, state):
        msg = Int32()
        msg.data = state
        self.publisher_md_state.publish(msg)

    def publishOffboardPositionMode(self):
        msg = OffboardControlMode()
        msg.timestamp = int(self.get_clock().now().nanoseconds/1000)
        msg.position = True
        msg.velocity = True
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        self.publisher_offboard_control_mode.publish(msg)

    def publishTrajectoryPositionSetpoint(self, x, y, z, yaw, yawspeed=0.):
        # If clipping is not zero, clip the position
        if self.position_clip > 0.1:
            x_clipped = clip(x, -self.position_clip, self.position_clip)
            y_clipped = clip(y, -self.position_clip, self.position_clip)
            z_clipped = clip(z, -self.position_clip, 0.0) # Negative up
        msg = TrajectorySetpoint()
        msg.position[0] = x_clipped
        msg.position[1] = y_clipped
        msg.position[2] = z_clipped
        msg.yaw = yaw
        msg.yawspeed=yawspeed
        msg.timestamp = int(self.get_clock().now().nanoseconds/1000)
        self.publisher_vehicle_trajectory_setpoint.publish(msg)

    def armVehicle(self):
        """Send an arm command to the vehicle."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
        self.get_logger().info('Arm command sent')

    def disarmVehicle(self):
        """Send a disarm command to the vehicle."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=0.0)
        self.get_logger().info('Disarm command sent')

    def engage_offboard_mode(self):
        """Switch to offboard mode."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
        self.get_logger().info("Switching to offboard mode")

    def land(self):
        """Switch to land mode."""
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        self.get_logger().info("Switching to land mode")

    def publish_vehicle_command(self, command, **params) -> None:
        """Publish a vehicle command."""
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = params.get("param1", 0.0)
        msg.param2 = params.get("param2", 0.0)
        msg.param3 = params.get("param3", 0.0)
        msg.param4 = params.get("param4", 0.0)
        msg.param5 = params.get("param5", 0.0)
        msg.param6 = params.get("param6", 0.0)
        msg.param7 = params.get("param7", 0.0)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.publisher_vehicle_command.publish(msg)

    # Callbacks
    def vehicle_status_callback(self, msg):
        if msg.arming_state == VehicleStatus.ARMING_STATE_ARMED:
            self.armed = True
        else:
            self.armed = False

        if msg.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            self.offboard = True
        else:
            self.offboard = False
        
        if msg.latest_disarming_reason == VehicleStatus.ARM_DISARM_REASON_KILL_SWITCH:
            self.killed = True
        else:
            self.killed = False

        self.vehicle_status = msg

    def vehicle_odometry_callback(self, msg):
        self.get_logger().debug(f'Received vehicle_odometry: {msg.position[2]}')
        self.vehicle_odometry = msg

    def vehicle_local_position_callback(self, msg):
        self.vehicle_local_position = msg

    def joint_states_callback(self, msg):
        self.arm_positions[0] = msg.position[0] # Pivot
        self.arm_positions[1] = msg.position[1] # Shoulder
        self.arm_positions[2] = msg.position[2] # Elbow

        self.arm_velocities[0] = msg.velocity[0] # Pivot
        self.arm_velocities[1] = msg.velocity[1] # Shoulder
        self.arm_velocities[2] = msg.velocity[2] # Elbow

    def transition_state(self, new_state='end'):
        if self.input_state != 0:
            self.get_logger().info('Manually triggered state transition')
            self.input_state = 0
        self.state_start_time = datetime.datetime.now() # Reset start time for next state
        self.first_state_loop = False # Reset first loop flag
        self.get_logger().info(f"Transition from {self.FSM_state} to {new_state}")
        self.FSM_state = new_state
        self.counter = 0

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