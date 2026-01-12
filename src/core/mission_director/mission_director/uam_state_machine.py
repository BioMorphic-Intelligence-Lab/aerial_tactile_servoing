import datetime

import numpy as np

from rclpy.node import Node
from std_msgs.msg import Int32
from sensor_msgs.msg import JointState

from px4_msgs.msg import TrajectorySetpoint
from px4_msgs.msg import OffboardControlMode
from px4_msgs.msg import VehicleCommand
from px4_msgs.msg import VehicleStatus
from px4_msgs.msg import VehicleOdometry
from px4_msgs.msg import VehicleLocalPosition

from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

class UAMStateMachine(Node):
    def __init__(self, node_name: str):
        super().__init__(node_name)

        # Parameters
        self.declare_parameter('sm.frequency', 10.0)
        self.frequency = self.get_parameter('sm.frequency').get_parameter_value().double_value
        self.timer_period = 1.0 / self.frequency
        self.declare_parameter('sm.position_clip', 0.0)
        self.position_clip = self.get_parameter('sm.position_clip').get_parameter_value().double_value
        self.declare_parameter('sm.fcu_on', True)
        self.fcu_on = self.get_parameter('sm.fcu_on').get_parameter_value().bool_value
        self.declare_parameter('sm.sim', False)
        self.sim = self.get_parameter('sm.sim').get_parameter_value().bool_value
        self.declare_parameter('sm.platform_mode', 'position')
        self.platform_mode = self.get_parameter('sm.platform_mode').get_parameter_value().string_value
        self.declare_parameter('sm.manipulator_mode', 'position')
        self.manipulator_mode = self.get_parameter('sm.manipulator_mode').get_parameter_value().string_value
        if self.sim:
            self.fcu_on = False  # In sim, FCU is always off

        self.get_logger().info(f"StateMachine node '{node_name}' initialized with params: \n"
                                f"frequency={self.frequency}, \n position_clip={self.position_clip}, \n fcu_on={self.fcu_on}, \n sim={self.sim}, \n platform_mode={self.platform_mode}, \n manipulator_mode={self.manipulator_mode}")

        # State machine publishers and subscribers
        self.pub_sm_state = self.create_publisher(Int32, '/md/state', 10)
        self.sub_sm_input = self.create_subscription(Int32, '/md/input', self.input_callback, 10)

        # Flight controller interfaces
        px4_qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.pub_vehicle_trajectory_setpoint = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', 10)
        self.pub_offboard_control_mode = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode',10)
        self.pub_vehicle_command = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', 10)

        self.sub_vehicle_status = self.create_subscription(VehicleStatus, '/fmu/out/vehicle_status_v1', self.vehicle_status_callback, px4_qos_profile)
        self.sub_vehicle_odometry = self.create_subscription(VehicleOdometry, '/fmu/out/vehicle_odometry', self.vehicle_odometry_callback, px4_qos_profile)
        self.sub_vehicle_local_position = self.create_subscription(VehicleLocalPosition, '/fmu/out/vehicle_local_position', self.vehicle_local_position_callback, px4_qos_profile)

        # Manipulator interfaces
        self.pub_servo_references = self.create_publisher(JointState, '/servo/in/state', 10)
        self.sub_servo_states = self.create_subscription(JointState, '/servo/out/state', self.servo_state_callback, 10)

        # Attributes
        self.FSM_state = "entrypoint"
        self.input_state = 0
        self.first_state_loop = True
        self.flying = False
        self.armed = False
        self.offboard = False
        self.takeoff_altitude = 1.5  # Default takeoff altitude
        self.vehicle_local_position = VehicleLocalPosition()
        self.vehicle_odometry = VehicleOdometry()
        self.vehicle_status = VehicleStatus()
        self.state_start_time = datetime.datetime.now()
        self.servo_state = JointState()
        self.home_position = np.zeros(4)  # x, y, z, heading
        self.hover_position = np.zeros(4)  # x, y, z, heading
        self.land_position = np.zeros(4)  # x, y, z, heading
        self.running_position = np.zeros(4)  # x, y, z, heading
        self.max_speed = 3.0 # Maximum platform speed for velocity setpoints

        self.kp = 2.0 # Arm position control proportional gain
        self.kd = 0.3 # Arm position control derivative gain

    def publish_offboard_position_mode(self):
        msg = OffboardControlMode()
        msg.timestamp = int(self.get_clock().now().nanoseconds/1000)
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        self.pub_offboard_control_mode.publish(msg)

    def publish_trajectory_position_setpoint(self, x: float, y: float, z: float, yaw: float, yawspeed: float = 0.):
        # If clipping is not zero, clip the position
        if self.position_clip > 0.1:
            x_clipped = np.clip(x, -self.position_clip, self.position_clip)
            y_clipped = np.clip(y, -self.position_clip, self.position_clip)
            z_clipped = np.clip(z, -self.position_clip, 0.0) # Negative up
            if (x != x_clipped) or (y != y_clipped) or (z != z_clipped):
                self.get_logger().warn(f"Position setpoint clipped to ({x_clipped}, {y_clipped}, {z_clipped}) from ({x}, {y}, {z})")
        else:
            x_clipped = x
            y_clipped = y
            z_clipped = z
        msg = TrajectorySetpoint()
        msg.position[0] = x_clipped
        msg.position[1] = y_clipped
        msg.position[2] = z_clipped
        msg.yaw = yaw
        msg.yawspeed=yawspeed
        msg.timestamp = int(self.get_clock().now().nanoseconds/1000)
        self.pub_vehicle_trajectory_setpoint.publish(msg)

    def publish_trajectory_velocity_setpoint(self, vx: float, vy: float, vz: float, yawspeed: float):
        # If clipping is not zero, clip the position
        vx_max = min(self.max_speed, max(0.0, self.position_clip - abs(self.vehicle_odometry.position[0])))
        vy_max = min(self.max_speed, max(0.0, self.position_clip - abs(self.vehicle_odometry.position[1])))
        vz_max = min(self.max_speed, max(0.0, self.position_clip + self.vehicle_odometry.position[2]))

        msg = TrajectorySetpoint()
        msg.position = [np.nan, np.nan, np.nan]
        msg.velocity[0] = np.clip(vx, -vx_max, vx_max)
        msg.velocity[1] = np.clip(vy, -vy_max, vy_max)
        msg.velocity[2] = np.clip(vz, -vz_max, vz_max)
        msg.yaw = np.nan
        msg.yawspeed = yawspeed
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.pub_vehicle_trajectory_setpoint.publish(msg)

    def transition_to_state(self, new_state='emergency'):
        if self.input_state != 0:
            self.get_logger().info('Manually triggered state transition')
            self.input_state = 0
        self.state_start_time = datetime.datetime.now() # Reset start time for next state
        self.first_state_loop = True # Reset first loop flag
        self.get_logger().info(f"[TRANSITION] {self.FSM_state} -> {new_state}")
        self.FSM_state = new_state
        self.counter = 0

    def handle_state(self, state_number: int):
        self.publish_md_state(state_number)
        self.publish_offboard_position_mode()
        self.counter += 1

    def publish_md_state(self, state):
        msg = Int32()
        msg.data = state
        self.pub_sm_state.publish(msg)

    def publish_servo_position_references(self, q_list: list):
        msg = JointState()
        for q in q_list:
            msg.name.append('q'+str(len(msg.position)+1))
            msg.position.append(q)
        msg.header.stamp = self.get_clock().now().to_msg()
        self.pub_servo_references.publish(msg)

    def publish_servo_velocity_references(self, q_dot_list: list):
        msg = JointState()
        for q_dot in q_dot_list:
            msg.name.append('q'+str(len(msg.velocity)+1))
            msg.velocity.append(q_dot)
        msg.header.stamp = self.get_clock().now().to_msg()
        self.pub_servo_references.publish(msg)

    def sim_engage_offboard_mode(self):
        """Switch to offboard mode."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
        self.get_logger().info("Switching to offboard mode")

    def sim_arm_vehicle(self):
        """Send an arm command to the vehicle."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
        self.get_logger().info('Arm command sent')

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
        self.pub_vehicle_command.publish(msg)

    #--------------------------------------------------------------------------
    # Callbacks
    #--------------------------------------------------------------------------
    def input_callback(self, msg: Int32):
        self.input_state = msg.data

    def vehicle_status_callback(self, msg: VehicleStatus):
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
    
    def vehicle_odometry_callback(self, msg: VehicleOdometry):
        self.vehicle_odometry = msg

    def vehicle_local_position_callback(self, msg: VehicleLocalPosition):
        self.vehicle_local_position = msg
    
    def servo_state_callback(self, msg: JointState):
        self.servo_state = msg

    #--------------------------------------------------------------------------
    # State Machine state executors
    #--------------------------------------------------------------------------
    def state_entrypoint(self, next_state='emergency'):
        self.handle_state(state_number=0)
        self.home_position[0] = self.vehicle_local_position.x
        self.home_position[1] = self.vehicle_local_position.y
        self.home_position[2] = self.vehicle_local_position.z
        self.home_position[3] = self.vehicle_local_position.heading        
        
        if self.first_state_loop:
            self.get_logger().info("Waiting for position fix")
            self.first_state_loop = False

        # State transition
        if (self.home_position[0] != 0.0 and self.home_position[1] != 0.0):
            self.get_logger().info(f'Got position fix! \t x: {self.home_position[0]:.3f} [m] \t y: {self.home_position[1]:.3f} [m] \t {np.rad2deg(self.home_position[3]):.2f} [deg]')
            self.transition_to_state(new_state=next_state)
        elif self.input_state == 1:
            self.get_logger().info('Bypassing position fix wait')
            self.transition_to_state(new_state=next_state)

    def state_do_nothing(self, next_state='emergency'):
        self.handle_state(state_number=1)
        self.get_logger().info('[1] Doing nothing... Publish 1 to continue \n ros2 topic pub --once /md/input std_msgs/msg/Int32 "{data: 1}"', throttle_duration_sec=1)

        # State transition
        if self.input_state == 1:
            self.transition_to_state(new_state=next_state)

    def state_wait_for_arming(self, next_state='emergency'):
        self.handle_state(state_number=3)

        self.get_logger().info('[3] Waiting for arming and offboard mode', throttle_duration_sec=1)

        if self.sim:
            if not self.offboard and self.counter%self.frequency==0:
                self.get_logger().info("[3] Sending sim offboard command")
                self.sim_engage_offboard_mode()
                self.counter = 0
            if not self.armed and self.offboard and self.counter%self.frequency==0:
                self.get_logger().info("[3] Sending sim arm command")
                self.sim_arm_vehicle()
                self.counter = 0

        # State transition
        if self.armed and not self.offboard:
            self.get_logger().info('[3] Armed but not offboard -- waiting', throttle_duration_sec=1)
        elif not self.armed and self.offboard:
            self.get_logger().info('[3] Not armed but offboard -- waiting', throttle_duration_sec=1)
        elif (self.armed and self.offboard) or self.input_state == 1:
            self.transition_to_state(new_state=next_state)

    def state_takeoff(self, target_altitude = 1.5, next_state='emergency'):
        self.handle_state(state_number=4)

        # First state loop
        if self.first_state_loop:
            self.takeoff_altitude = -abs(target_altitude)
            self.home_position[2] = self.takeoff_altitude
            self.get_logger().info(f'[4] Vehicle local position heading: {self.home_position[3]} rad')
            self.get_logger().info(f'[4] Takeoff z-coord: {self.home_position[2]} m')
            self.first_state_loop = False

        self.publish_trajectory_position_setpoint(*self.home_position)
        current_altitude = self.vehicle_local_position.z

        # State transition
        if not self.offboard and self.fcu_on:
            self.transition_to_state('emergency')
        elif abs(current_altitude)+0.1 > abs(self.home_position[2]) or self.input_state==1:
            self.transition_to_state(new_state=next_state)
            self.flying = True

    def state_land(self, landing_speed=0.5, next_state='emergency'):
        self.handle_state(state_number=5)
        self.publish_trajectory_position_setpoint(*self.land_position)
        self.land_position[2] += landing_speed / self.frequency  # Increase z (down) at landing speed

        # First state loop
        if self.first_state_loop:
            self.land_position[0] = self.vehicle_local_position.x
            self.land_position[1] = self.vehicle_local_position.y
            self.land_position[2] = self.vehicle_local_position.z
            self.land_position[3] = self.vehicle_local_position.heading
            self.get_logger().info(f'[5] Landing from altitude: {self.land_position[2]} m at speed {landing_speed} m/s')
            self.first_state_loop = False

        # State transition
        if not self.offboard and self.fcu_on:
            self.transition_to_state('emergency')
        elif (self.vehicle_local_position.z >= -0.1 and self.land_position[2] > 0.5) or self.input_state==1:
            self.transition_to_state(new_state=next_state)
            self.flying = False

    def state_hover(self, duration_sec: float, next_state='emergency'):
        self.handle_state(state_number=10)

        # First state loop
        if self.first_state_loop:
            self.hover_position[0] = self.vehicle_local_position.x
            self.hover_position[1] = self.vehicle_local_position.y
            self.hover_position[2] = self.vehicle_local_position.z
            self.hover_position[3] = self.vehicle_local_position.heading
            self.get_logger().info(f'[5] Hovering at altitude: {self.home_position[2]:.2f} m for {duration_sec} seconds')
            self.first_state_loop = False

        self.get_logger().info(f'Hovering... {(datetime.datetime.now()-self.state_start_time).seconds:.1f}/{duration_sec} sec', throttle_duration_sec=1)

        # State transition
        if not self.offboard and self.fcu_on:
            self.transition_to_state('emergency')
        elif (datetime.datetime.now()-self.state_start_time).seconds > duration_sec or self.input_state==1:
            self.transition_to_state(new_state=next_state)

    """ Move the arms to the specified joint positions 'q' using either servo position or velocity control mode. Always goes to a specified position
    using either servo position or servo velocity as the controlled variables.
    Args:
        q (list): Target joint positions for the arms.
        mode (str, optional): Control mode, either 'position' or 'velocity'. Defaults to 'position'.
        next_state (str, optional): Next state to transition to after completion. Defaults to 'emergency'.
        epsilon (float, optional): Position error threshold for completion. Defaults to 0.1.
    """
    def state_move_arms(self, q: list, next_state='emergency', epsilon=0.1):
        self.handle_state(state_number=11)
        error = 0.0
        # First state loop
        if self.first_state_loop:
            self.hover_position[0] = self.vehicle_local_position.x
            self.hover_position[1] = self.vehicle_local_position.y
            self.hover_position[2] = self.vehicle_local_position.z
            self.hover_position[3] = self.vehicle_local_position.heading
            self.get_logger().info(f'[11] Hovering at altitude: {self.home_position[2]:.2f} m while moving arms to states {q} in mode {self.manipulator_mode}')
            self.first_state_loop = False
        
        if self.manipulator_mode=='position': # If the servos are controlled in position mode
            self.publish_servo_position_references(q)
        elif self.manipulator_mode=='velocity': # If the servos are controlled in velocity mode
            q_dot_cmd = []
            for i in range(len(q)):
                qd = (self.kp*(q[i] - self.servo_state.position[i]) - self.kd*self.servo_state.velocity[i]) # Simple P controller to reach target position
                q_dot_cmd.append(qd)
                error += abs(q[i]-self.servo_state.position[i])
            error = np.sqrt(error)
            # self.get_logger().info(f'Arm position error: {error:.4f} rad')
            if error > epsilon:
                self.publish_servo_velocity_references(q_dot_cmd)
            else:
                self.publish_servo_velocity_references([0.0 for x in q])  # Stop the servos if within epsilon

        # State transition
        if not self.offboard and self.fcu_on and self.flying:
            self.transition_to_state('emergency')
        elif (error < epsilon) or self.input_state == 1: # If error is small enough or input state is 1
            self.transition_to_state(new_state=next_state)

    def state_move_uam_to_position(self, target_position: list, next_state='emergency'):
        self.handle_state(state_number=12)

        # First state loop
        if self.first_state_loop:
            self.get_logger().info(f'[12] Moving UAM to position: {target_position}')
            self.first_state_loop = False
        if len(target_position) == 3:
            target_position.append(self.vehicle_local_position.heading)  # Keep current heading if not provided
        self.publish_trajectory_position_setpoint(*target_position)
        # Calculate euclidean distance between current and target positions
        current_position = np.array([self.vehicle_local_position.x, self.vehicle_local_position.y, self.vehicle_local_position.z])
        target_pos = np.array(target_position[:3])  # x, y, z
        error = np.linalg.norm(current_position - target_pos)
        self.get_logger().info(f'Current position: {current_position}, Target position: {target_pos}, Error: {error:.4f}', throttle_duration_sec=1)

        # State transition
        if not self.offboard and self.fcu_on and self.flying:
            self.transition_to_state('emergency')
        elif (error < 0.2) or self.input_state==1: # If error is small enough or input state is 1
            self.transition_to_state(new_state=next_state)

    def state_emergency(self):
        self.handle_state(state_number=-1)
        self.get_logger().warn("EMERGENCY STATE! No offboard mode.", throttle_duration_sec=1)
        error = 0.0
        epsilon = 0.1
        q_emergency = [1.578, 0.0, -1.85]
        if self.manipulator_mode=='position': # If the servos are controlled in position mode
            self.publish_servo_position_references(q_emergency)
        elif self.manipulator_mode=='velocity': # If the servos are controlled in velocity mode
            q_dot_cmd = []
            for i in range(len(q_emergency)):
                qd = (self.kp*(q_emergency[i] - self.servo_state.position[i]) - self.kd*self.servo_state.velocity[i]) # Simple P controller to reach target position
                q_dot_cmd.append(qd)
                error += abs(q_emergency[i]-self.servo_state.position[i])
            error = np.sqrt(error)
            if error > epsilon:
                self.publish_servo_velocity_references(q_dot_cmd)
            else:
                self.publish_servo_velocity_references([0.0 for x in q_emergency])  # Stop the servos if within epsilon
