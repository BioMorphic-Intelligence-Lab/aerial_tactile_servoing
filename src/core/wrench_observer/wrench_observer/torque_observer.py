import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rcl_interfaces.msg import SetParametersResult
from rclpy.parameter import Parameter

import numpy as np
import datetime

from std_msgs.msg import Float64
from geometry_msgs.msg import Vector3Stamped
from sensor_msgs.msg import JointState
from px4_msgs.msg import SensorCombined, ActuatorMotors

L_1 = 0.118
L_2 = 0.326 
L_3 = 0.273 # 0.330 with tactip

class TorqueObserver(Node):
    def __init__(self):
        super().__init__('external_torque_observer')

        # Parameters
        self.declare_parameter('frequency', 100.0)
        self.declare_parameter('gain_torque', 1.0)
        self.declare_parameter('alpha_torque', 1.0)
        self.declare_parameter('alpha_angular_velocity', 0.7)
        self.declare_parameter('alpha_accelerometer', 0.7)
        self.declare_parameter('alpha_motor_inputs', 0.7)

        self.frequency = self.get_parameter('frequency').get_parameter_value().double_value
        self.gain_torque = self.get_parameter('gain_torque').get_parameter_value().double_value * np.eye(3)
        self.alpha_torque = self.get_parameter('alpha_torque').get_parameter_value().double_value
        self.alpha_angular_velocity = self.get_parameter('alpha_angular_velocity').get_parameter_value().double_value
        self.alpha_accelerometer = self.get_parameter('alpha_accelerometer').get_parameter_value().double_value
        self.alpha_motor_inputs = self.get_parameter('alpha_motor_inputs').get_parameter_value().double_value

        # Register parameter change callback
        self.add_on_set_parameters_callback(self.param_callback)

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # IMU
        self.subscriber_accel = self.create_subscription(SensorCombined, '/fmu/out/sensor_combined', self.sensor_callback, qos_profile)
        self.sensor_acceleration = np.array([0., 0., 0.])
        self.sensor_angular_velocity = np.array([0., 0., 0.])

        # Actuators
        self.subscriber_actuators = self.create_subscription(ActuatorMotors, '/fmu/out/actuator_motors', self.actuator_callback, qos_profile)
        self.actuator_thrust = None

        self.subscriber_servos = self.create_subscription(JointState, '/servo/out/state', self.servo_callback, 10)
        self.servo_state = None

        # Publisher
        self.publisher_estimator_torque = self.create_publisher(Vector3Stamped, '/wrench_observer/out/estimated_torque', 10)
        self.publisher_estimator_torque_magnitude = self.create_publisher(Float64, '/wrench_observer/out/torque_magnitude', 10)

        # Logging publishers
        self.publisher_acclero_smoothed = self.create_publisher(Vector3Stamped, '/wrench_observer/log/accelero_smoothed', 10)
        self.publisher_gyro_smoothed = self.create_publisher(Vector3Stamped, '/wrench_observer/log/gyro_smoothed', 10)
        self.publisher_torque_unsmoothed = self.create_publisher(Vector3Stamped, '/wrench_observer/log/torque_unsmoothed', 10)
        
        # Model parameters
        self.thrust_coefficient = 21.0 # Obtained through experimental data previous value 19.468 18.538 21.0 with new batteries
        self.propeller_incline_angle = 5. # [deg] propeller incline in degreess
        self.model_mass = 4.239 # [kg] with 6000 mAh batteries
        self.acceleration_gravity = np.array([0., 0., 9.81])
        self.linear_allocation_matrix = np.array([[-np.sin(np.deg2rad(60.))*np.sin(np.deg2rad(self.propeller_incline_angle)), np.sin(np.deg2rad(60.))*np.sin(np.deg2rad(self.propeller_incline_angle)), -np.sin(np.deg2rad(60.))*np.sin(np.deg2rad(self.propeller_incline_angle)), np.sin(np.deg2rad(60.))*np.sin(np.deg2rad(self.propeller_incline_angle))],
                                                 [-np.sin(np.deg2rad(30.))*np.sin(np.deg2rad(self.propeller_incline_angle)), np.sin(np.deg2rad(30.))*np.sin(np.deg2rad(self.propeller_incline_angle)), np.sin(np.deg2rad(30.))*np.sin(np.deg2rad(self.propeller_incline_angle)), -np.sin(np.deg2rad(30.))*np.sin(np.deg2rad(self.propeller_incline_angle))],
                                                 [-np.cos(np.deg2rad(self.propeller_incline_angle)), -np.cos(np.deg2rad(self.propeller_incline_angle)), -np.cos(np.deg2rad(self.propeller_incline_angle)), -np.cos(np.deg2rad(self.propeller_incline_angle))]])

        self.inertia = np.array([[0.072, -1.111e-5, -7.294e-6],
                                 [-1.111e-5, 0.067, -1.552e-5],
                                 [-7.294e-6, -1.552e-5, 0.128]])

        arm_x = 0.184 # [m] Moment arm along the body x-axis
        arm_y = 0.231 # [m] Moment arm along the body y-axis
        drag_coeff = 0.1e-5 # Somewhat arbitrary
        yaw_moment_arm = 0.33911 # [m] Moment arm of yaw contribution of thrust
        self.rotational_allocation_matrix = np.array([[-arm_y*np.cos(np.deg2rad(self.propeller_incline_angle)), arm_y*np.cos(np.deg2rad(self.propeller_incline_angle)), arm_y*np.cos(np.deg2rad(self.propeller_incline_angle)), -arm_y*np.cos(np.deg2rad(self.propeller_incline_angle))], # Roll moment
                                                     [-arm_x*np.cos(np.deg2rad(self.propeller_incline_angle)), arm_x*np.cos(np.deg2rad(self.propeller_incline_angle)), -arm_x*np.cos(np.deg2rad(self.propeller_incline_angle)), arm_x*np.cos(np.deg2rad(self.propeller_incline_angle))], # Pitch moment
                                                     [np.sin(np.deg2rad(self.propeller_incline_angle))*yaw_moment_arm+drag_coeff, np.sin(np.deg2rad(self.propeller_incline_angle))*yaw_moment_arm+drag_coeff, -np.sin(np.deg2rad(self.propeller_incline_angle))*yaw_moment_arm+drag_coeff, -np.sin(np.deg2rad(self.propeller_incline_angle))*yaw_moment_arm+drag_coeff]]) # Yaw moment

        self.R_accelerometer = np.array([[-1., 0., 0.],
                                         [0., -1., 0.],
                                         [0., 0., -1.]])
        
        # Initial values for updating member variables
        self.most_recent_force_estimate = np.array([0., 0., 0.])
        self.most_recent_torque_estimate = np.array([0., 0., 0.])
        self.momentum_integral = np.array([0., 0., 0.])
        self.torque_bias = np.array([0.25, 0.06, 0.13])

        # Timer -- always last
        self.previous_time = datetime.datetime.now()
        self.counter = 0
        self.timer_period = 1.0 / self.frequency
        while(self.sensor_acceleration is None or self.sensor_angular_velocity is None or self.actuator_thrust is None):
            self.get_logger().info('Waiting for initial sensor and actuator data...', throttle_duration_sec=2.)
            rclpy.spin_once(self, timeout_sec=0.1)
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

    def timer_callback(self):
        self.torque_observer()

    def torque_observer(self):
        dt = (datetime.datetime.now() - self.previous_time).total_seconds() # Get time difference since last
        # self.get_logger().info(f'dt [sec]: {dt}')
        self.previous_time = datetime.datetime.now()

        # Torque observer
        # Calculate observed angular momentum
        current_angular_momentum = self.inertia @ self.sensor_angular_velocity * dt

        # Update angular momentum model
        self.momentum_integral = self.momentum_integral + (-np.cross(self.sensor_angular_velocity, current_angular_momentum) + 
                                                           self.rotational_allocation_matrix @ self.actuator_thrust + self.most_recent_torque_estimate) * dt
        # Torque estimate is the difference between measured and model, with a gain
        current_torque_estimate = self.gain_torque @ (current_angular_momentum - self.momentum_integral) + self.torque_bias # Z torque bias term
        self.publish_unsmoothed_torque(current_torque_estimate)
        
        # EWMA smoothing
        self.most_recent_torque_estimate = (self.alpha_torque * current_torque_estimate +
            (1. - self.alpha_torque) * self.most_recent_torque_estimate)
        self.publish_estimated_torque(self.most_recent_torque_estimate)
        self.publish_estimated_torque_magnitude(np.linalg.norm(self.most_recent_torque_estimate))
        self.get_logger().info(f'Torque estimate [{self.most_recent_torque_estimate[0]:.2f}, {self.most_recent_torque_estimate[1]:.2f}, {self.most_recent_torque_estimate[2]:.2f}] [Nm]', throttle_duration_sec=1)

    def publish_estimated_torque(self, torque):
        msg = Vector3Stamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.vector.x = torque[0]
        msg.vector.y = torque[1]
        msg.vector.z = torque[2]
        self.publisher_estimator_torque.publish(msg)

    def publish_estimated_torque_magnitude(self, torque_magnitude:float):
        msg = Float64()
        msg.data = torque_magnitude
        self.publisher_estimator_torque_magnitude.publish(msg)

    def publish_unsmoothed_torque(self, torque):
        msg = Vector3Stamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.vector.x = torque[0]
        msg.vector.y = torque[1]
        msg.vector.z = torque[2]
        self.publisher_torque_unsmoothed.publish(msg)

    def publish_smoothed_accelero(self, accelero):
        msg = Vector3Stamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.vector.x = accelero[0]
        msg.vector.y = accelero[1]
        msg.vector.z = accelero[2]
        self.publisher_acclero_smoothed.publish(msg)

    def publish_smoothed_gyro(self, gyro):
        msg = Vector3Stamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.vector.x = gyro[0]
        msg.vector.y = gyro[1]
        msg.vector.z = gyro[2]
        self.publisher_gyro_smoothed.publish(msg)

    # CALLBACKS
    def sensor_callback(self, msg):
        self.sensor_acceleration = self.alpha_accelerometer * np.array(msg.accelerometer_m_s2) + (1.-self.alpha_accelerometer)*self.sensor_acceleration
        self.sensor_angular_velocity = self.alpha_angular_velocity * np.array(msg.gyro_rad) + (1.-self.alpha_angular_velocity)*self.sensor_angular_velocity
        # Republish the smoothed signals for checking
        self.publish_smoothed_accelero(self.sensor_acceleration)
        self.publish_smoothed_gyro(self.sensor_angular_velocity)

    def actuator_callback(self, msg):
        if self.actuator_thrust is None:
            self.actuator_thrust = np.array([0., 0., 0., 0.])
        smoothed_0 = self.alpha_motor_inputs*msg.control[0]*self.thrust_coefficient + (1.-self.alpha_motor_inputs)*self.actuator_thrust[0]
        smoothed_1 = self.alpha_motor_inputs*msg.control[1]*self.thrust_coefficient + (1.-self.alpha_motor_inputs)*self.actuator_thrust[1]
        smoothed_2 = self.alpha_motor_inputs*msg.control[2]*self.thrust_coefficient + (1.-self.alpha_motor_inputs)*self.actuator_thrust[2]
        smoothed_3 = self.alpha_motor_inputs*msg.control[3]*self.thrust_coefficient + (1.-self.alpha_motor_inputs)*self.actuator_thrust[3]
        self.actuator_thrust = np.array([smoothed_0, smoothed_1, smoothed_2, smoothed_3])

    def servo_callback(self, msg):
        self.servo_state = msg
    
    def param_callback(self, params):
        """
        Called whenever one or more parameters are set via `ros2 param set`
        """
        for param in params:
            if param.name == 'gain_torque' and param.type_ == Parameter.Type.DOUBLE:
                self.get_logger().info(f"Param updated from {self.gain_torque} to {param.value* np.eye(3)}")
                self.gain_torque = param.value * np.eye(3)
            elif param.name == 'alpha_torque' and param.type_ == Parameter.Type.DOUBLE:
                self.get_logger().info(f"Param updated from {self.alpha_torque} to {param.value}")
                self.alpha_torque = param.value
            elif param.name == 'alpha_angular_velocity' and param.type_ == Parameter.Type.DOUBLE:
                self.get_logger().info(f"Param updated from {self.alpha_angular_velocity} to {param.value}")
            elif param.name == 'alpha_accelerometer' and param.type_ == Parameter.Type.DOUBLE:
                self.get_logger().info(f"Param updated from {self.alpha_accelerometer} to {param.value}")
                self.alpha_accelerometer = param.value
            elif param.name == 'alpha_motor_inputs' and param.type_ == Parameter.Type.DOUBLE:
                self.get_logger().info(f"Param updated from {self.alpha_motor_inputs} to {param.value}")
                self.alpha_motor_inputs = param.value
        # Returning successful result allows the change
        return SetParametersResult(successful=True)

    def position_forward_kinematics(self, q_1, q_2, q_3):
        x_BS = -L_2*np.sin(q_2) - L_3*np.sin(q_2)*np.cos(q_3)
        y_BS = L_1*np.sin(q_1) + L_2*np.sin(q_1)*np.cos(q_2) + L_3*np.sin(q_1)*np.cos(q_2)*np.cos(q_3) + L_3*np.sin(q_3)*np.cos(q_1)
        z_BS = -L_1*np.cos(q_1) - L_2*np.cos(q_1)*np.cos(q_2) + L_3*np.sin(q_1)*np.sin(q_3) - L_3*np.cos(q_1)*np.cos(q_2)*np.cos(q_3)
        return [x_BS, y_BS, z_BS]

def main():
    rclpy.init(args=None)
    torque_observer = TorqueObserver()
    rclpy.spin(torque_observer)

    # Destroy the node explicitly
    torque_observer.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()