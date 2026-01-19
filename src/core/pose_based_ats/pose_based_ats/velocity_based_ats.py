import rclpy
from rclpy.node import Node

import numpy as np

from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R

from tf2_ros import TransformBroadcaster
from std_msgs.msg import Int8, Int32, Float64
from geometry_msgs.msg import TwistStamped, TransformStamped
from sensor_msgs.msg import JointState
from px4_msgs.msg import VehicleOdometry, TrajectorySetpoint

L_1 = 0.12025  # Distance from body frame to first servo axis
L_2 = 0.336 # Distance from first servo axis to second servo axis
L_3 = 0.327 # Distance from second servo axis to TacTip sensor

class VelocityBasedATS(Node):
    def __init__(self):
        super().__init__('velocity_based_ats')

        # Parameters
        self.declare_parameter('frequency', 10.)
        self.declare_parameter('Kp_linear', 3.0)
        self.declare_parameter('Kp_angular', 3.0)
        self.declare_parameter('Ki_linear', 0.1)
        self.declare_parameter('Ki_angular', 0.1)
        self.declare_parameter('windup_clip', 10.)
        self.declare_parameter('publish_log', True)
        self.declare_parameter('regularization_weight', 0.001)
        self.declare_parameter('test_execution_time', False)
        self.integrator = np.zeros(6)
        self.windup = self.get_parameter('windup_clip').get_parameter_value().double_value
        self.reg_weight = self.get_parameter('regularization_weight').get_parameter_value().double_value

        # Subscribers
        self.subscription_tactip = self.create_subscription(TwistStamped, '/tactip/pose', self.callback_tactip, 10)
        self.subscription_tactip_contact = self.create_subscription(Int8, '/tactip/contact', self.callback_tactip_contact, 10)
        self.subscription_servos = self.create_subscription(JointState, '/servo/out/state', self.callback_servo_feedback, 10)
        self.subscription_fmu = self.create_subscription(VehicleOdometry, '/fmu/in/vehicle_visual_odometry', self.callback_fmu, 10)
        self.subscription_md = self.create_subscription(Int32, '/md/state', self.md_callback, 10)
        self.sub_reference = self.create_subscription(TwistStamped, '/references/sensor_in_contact', self.callback_reference, 10)

        # Publishers (necessary)
        self.publisher_servo_state = self.create_publisher(JointState, '/controller/out/servo_state', 10)
        self.publisher_drone_ref = self.create_publisher(TrajectorySetpoint, '/controller/out/trajectory_setpoint', 10)

        # Publishers (for logging and debugging of IK)
        self.pub_u_s = self.create_publisher(TwistStamped, '/controller/log/us', 10)
        self.pub_u_ss = self.create_publisher(TwistStamped, '/controller/log/uss', 10)
        self.pub_proportional = self.create_publisher(TwistStamped, '/controller/log/proportional', 10)
        self.pub_integrator = self.create_publisher(TwistStamped, '/controller/log/integrator', 10)

        # Broadcasters
        self.broadcaster_tf2 = TransformBroadcaster(self)

        # Gains
        self.Kp = np.eye(6)
        self.Kp[0,0] = self.get_parameter('Kp_linear').get_parameter_value().double_value
        self.Kp[1,1] = self.get_parameter('Kp_linear').get_parameter_value().double_value
        self.Kp[2,2] = self.get_parameter('Kp_linear').get_parameter_value().double_value
        self.Kp[3,3] = self.get_parameter('Kp_angular').get_parameter_value().double_value
        self.Kp[4,4] = self.get_parameter('Kp_angular').get_parameter_value().double_value
        self.Kp[5,5] = self.get_parameter('Kp_angular').get_parameter_value().double_value

        self.Ki = np.eye(6)
        self.Ki[0,0] = self.get_parameter('Ki_linear').get_parameter_value().double_value
        self.Ki[1,1] = self.get_parameter('Ki_linear').get_parameter_value().double_value
        self.Ki[2,2] = self.get_parameter('Ki_linear').get_parameter_value().double_value
        self.Ki[3,3] = self.get_parameter('Ki_angular').get_parameter_value().double_value
        self.Ki[4,4] = self.get_parameter('Ki_angular').get_parameter_value().double_value
        self.Ki[5,5] = self.get_parameter('Ki_angular').get_parameter_value().double_value

        self.P_Cref = self.evaluate_P_CS(0., 0., 0.) # Initial contact frame at zero angles and zero depth

        self.tactip = TwistStamped()
        self.tactip.twist.linear.x = 0.0
        self.tactip.twist.linear.y = 0.0
        self.tactip.twist.linear.z = 0.0
        self.tactip.twist.angular.x = 0.0
        self.tactip.twist.angular.y = 0.0
        self.tactip.twist.angular.z = 0.0
        self.servo_state = JointState()
        self.servo_state.position = [0., 0., 0.]
        self.vehicle_odometry = VehicleOdometry()
        self.vehicle_odometry.q = [1., 0., 0., 0.]  # w, x, y, z
        self.contact = False
        self.accumulate_integrator = False
        self.md_state = 0

        self.nominal_state = np.array([0, 0, 0, 0, 0, 0, np.pi/3, 0, np.pi/6])

        self.previous_yaw_cmd = 0.0

        # Timer
        self.get_logger().info("Starting pose-based ATS controller...")
        self.period = 1./self.get_parameter('frequency').get_parameter_value().double_value
        self.timer = self.create_timer(self.period, self.callback_timer)

    def callback_timer(self):
        # Get state
        state = self.get_state()
        jacobian_full = self.evaluate_JG(roll=state[3], pitch=state[4], yaw=state[5], q_1=state[6], q_2=state[7], q_3=state[8])
        J_controlled = jacobian_full[:,[0,1,2,5,6,7,8]] # X, Y, Z, yaw, q1, q2, q3 are controlled DOFs
        J_uncontrolled = jacobian_full[:,[3,4]] # Roll and pitch are uncontrolled DOFs
        J_controlled_pinv = np.linalg.pinv(J_controlled) # Pseudo-inverse of controlled jacobian
        J_null = np.eye(J_controlled.shape[1]) - J_controlled_pinv @ J_controlled # Null space projector of controlled jacobian

        # Broadcast the current drone pose world -> body
        self.broadcast_tf2(self.evaluate_P_B(state), "world", "present_body_frame")

        # Evaluate the error
        P_SC = self.evaluate_P_SC(np.deg2rad(self.tactip.twist.angular.x), np.deg2rad(self.tactip.twist.angular.y),
                                  self.tactip.twist.linear.x/1000., self.tactip.twist.linear.y/1000., self.tactip.twist.linear.z/1000.)
        E_Sref = P_SC @ self.P_Cref
        e_sr = self.transformation_to_vector(E_Sref)

        # Check for contact through SSIM
        if self.accumulate_integrator: # If contact, accumulate integrator
            self.integrator += self.Ki @ e_sr
        else: # If not contact, reset integrator    
            self.integrator = 0.

        u_ss = -self.Kp@e_sr - np.clip(self.integrator,-self.windup, self.windup) # u_ss is in sensor frame, transform to inertial frame
        self.publish_twist(u_ss, self.pub_u_ss) # Publish u_ss for logging
        self.publish_twist(-self.Kp@e_sr, self.pub_proportional) # Publish proportional term for logging
        self.publish_twist(-np.clip(self.integrator, -self.windup, self.windup), self.pub_integrator) # Publish integrator for logging

        # Rotate u_ss from sensor frame to inertial frame
        R_S = self.evaluate_P_S(state)[0:3, 0:3]
        u_s = np.concatenate((R_S @ u_ss[0:3], R_S @ u_ss[3:]), axis=0)
        self.publish_twist(u_s, self.pub_u_s) # Publish u_s for log

        # Secondary objective: move servos to nominal position and away from the singularity
        q_secondary = np.zeros(7) 
        q_secondary[4] = self.nominal_state[6] - state[6]  # q1 nominal position
        q_secondary[5] = self.nominal_state[7] - state[7]  # q2 nominal position
        q_secondary[6] = self.nominal_state[8] - state[8]  # q3 nominal position

        # Transform angular velocity from FCU to euler angle rates
        euler_rate_inertial = self.T_euler_rate_to_angular_velocity_inv(state[4], state[5]) @ \
            R.from_euler('ZYX', np.array([state[5], state[4], state[3]])).as_matrix().T @ \
            np.array([self.vehicle_odometry.angular_velocity[0], self.vehicle_odometry.angular_velocity[1], self.vehicle_odometry.angular_velocity[2]])

        # Inverse kinematics - controlled states [x, y, z, yaw, q1, q2, q3]
        controlled_state_reference = J_controlled_pinv @ u_s - \
            J_controlled_pinv @ J_uncontrolled @ np.array([euler_rate_inertial[0], euler_rate_inertial[1]]) \
            + J_null @ q_secondary # Secondary objective velocities
        
        # TODO: convert controlled state reference from euler angle rates to angular velocities in the body frame

        # Broadcast the sensor frame in the body frame
        P_BS = self.evaluate_P_BS(state[6], state[7], state[8])
        self.broadcast_tf2(P_BS, "present_body_frame", "present_sensor_frame")
        self.broadcast_tf2(P_SC, "present_sensor_frame", "present_contact_frame")

        # Publish velocity commands
        servo_cmd = JointState()
        servo_cmd.name = ['q1', 'q2', 'q3']
        servo_cmd.velocity = [controlled_state_reference[4], controlled_state_reference[5], controlled_state_reference[6]]
        servo_cmd.header.stamp = self.get_clock().now().to_msg()
        self.publisher_servo_state.publish(servo_cmd)

        drone_cmd = TrajectorySetpoint()
        drone_cmd.position = [np.nan, np.nan, np.nan]  # Position is not controlled
        drone_cmd.velocity = [float(controlled_state_reference[0]), float(controlled_state_reference[1]), float(controlled_state_reference[2])]
        drone_cmd.yaw = np.nan  # Yaw position is not controlled
        drone_cmd.yawspeed = float(controlled_state_reference[3])
        drone_cmd.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.publisher_drone_ref.publish(drone_cmd)

    def callback_tactip(self, msg):
        self.tactip = msg

    def callback_reference(self, msg):
        self.P_Cref = self.evaluate_P_CS(
            np.deg2rad(msg.twist.angular.x), # received in deg
            np.deg2rad(msg.twist.angular.y), # received in deg
            msg.twist.linear.z/1000.) # received in mm # TODO Invert to publish transform (sensor in contact to contact in sensor frames)
    
        

    def callback_tactip_contact(self, msg):
        self.contact = msg.data

    def callback_servo_feedback(self, msg):
        self.servo_state = msg

    def callback_fmu(self, msg):
        self.vehicle_odometry = msg

    def md_callback(self, msg):
        self.md_state = msg.data

    ''' Evaluate transformation matrix of body frame in inertial frame
    '''
    def evaluate_P_B(self,state):
        roll = state[3]
        pitch = state[4]
        yaw = state[5]
        x_B = state[0]
        y_B = state[1]
        z_B = state[2]
        P_B = np.zeros((4,4))
        P_B[0,0] = np.cos(pitch)*np.cos(yaw)
        P_B[0,1] = np.sin(pitch)*np.sin(roll)*np.cos(yaw) - np.sin(yaw)*np.cos(roll)
        P_B[0,2] = np.sin(pitch)*np.cos(roll)*np.cos(yaw) + np.sin(roll)*np.sin(yaw)
        P_B[0,3] = x_B
        P_B[1,0] = np.sin(yaw)*np.cos(pitch)
        P_B[1,1] = np.sin(pitch)*np.sin(roll)*np.sin(yaw) + np.cos(roll)*np.cos(yaw)
        P_B[1,2] = np.sin(pitch)*np.sin(yaw)*np.cos(roll) - np.sin(roll)*np.cos(yaw)
        P_B[1,3] = y_B
        P_B[2,0] = -np.sin(pitch)
        P_B[2,1] = np.sin(roll)*np.cos(pitch)
        P_B[2,2] = np.cos(pitch)*np.cos(roll)
        P_B[2,3] = z_B
        P_B[3,0] = 0
        P_B[3,1] = 0
        P_B[3,2] = 0
        P_B[3,3] = 1
        return P_B

    ''' Evaluate transformation matrix of contact frame in sensor frame
    '''
    def evaluate_P_SC(self, alpha, beta, x_CS, y_CS, z_CS):
        P_SC = np.zeros((4,4))
        P_SC[0,0] = np.cos(beta)
        P_SC[0,1] = 0
        P_SC[0,2] = -np.sin(beta)
        P_SC[0,3] = -x_CS*np.cos(beta) + z_CS*np.sin(beta)
        P_SC[1,0] = np.sin(alpha)*np.sin(beta)
        P_SC[1,1] = np.cos(alpha)
        P_SC[1,2] = np.sin(alpha)*np.cos(beta)
        P_SC[1,3] = -x_CS*np.sin(alpha)*np.sin(beta) - y_CS*np.cos(alpha) - z_CS*np.sin(alpha)*np.cos(beta)
        P_SC[2,0] = np.sin(beta)*np.cos(alpha)
        P_SC[2,1] = -np.sin(alpha)
        P_SC[2,2] = np.cos(alpha)*np.cos(beta)
        P_SC[2,3] = -x_CS*np.sin(beta)*np.cos(alpha) + y_CS*np.sin(alpha) - z_CS*np.cos(alpha)*np.cos(beta)
        P_SC[3,0] = 0
        P_SC[3,1] = 0
        P_SC[3,2] = 0
        P_SC[3,3] = 1

        return P_SC

    ''' Evaluate transformation matrix of sensor frame in contact frame
    This is the alpha and beta that is the output of the TacTip
    '''
    def evaluate_P_CS(self, alpha, beta, d):
        P_CS = np.zeros((4,4))
        P_CS[0,0] = np.cos(beta)
        P_CS[0,1] = np.sin(alpha)*np.sin(beta)
        P_CS[0,2] = np.sin(beta)*np.cos(alpha)
        P_CS[0,3] = 0
        P_CS[1,0] = 0
        P_CS[1,1] = np.cos(alpha)
        P_CS[1,2] = -np.sin(alpha)
        P_CS[1,3] = 0
        P_CS[2,0] = -np.sin(beta)
        P_CS[2,1] = np.sin(alpha)*np.cos(beta)
        P_CS[2,2] = np.cos(alpha)*np.cos(beta)
        P_CS[2,3] = d
        P_CS[3,0] = 0
        P_CS[3,1] = 0
        P_CS[3,2] = 0
        P_CS[3,3] = 1
        return P_CS

    ''' Evaluate transformation matrix of sensor frame in body frame
    '''
    def evaluate_P_BS(self, q_1, q_2, q_3):
        P_BS = np.zeros((4,4))
        P_BS[0,0] = np.cos(q_2)
        P_BS[0,1] = np.sin(q_2)*np.sin(q_3)
        P_BS[0,2] = np.sin(q_2)*np.cos(q_3)
        P_BS[0,3] = -(L_2 + L_3*np.cos(q_3))*np.sin(q_2)
        P_BS[1,0] = np.sin(q_1)*np.sin(q_2)
        P_BS[1,1] = -np.sin(q_1)*np.sin(q_3)*np.cos(q_2) + np.cos(q_1)*np.cos(q_3)
        P_BS[1,2] = -np.sin(q_1)*np.cos(q_2)*np.cos(q_3) - np.sin(q_3)*np.cos(q_1)
        P_BS[1,3] = L_1*np.sin(q_1) + L_2*np.sin(q_1)*np.cos(q_2) + L_3*np.sin(q_1)*np.cos(q_2)*np.cos(q_3) + L_3*np.sin(q_3)*np.cos(q_1)
        P_BS[2,0] = -np.sin(q_2)*np.cos(q_1)
        P_BS[2,1] = np.sin(q_1)*np.cos(q_3) + np.sin(q_3)*np.cos(q_1)*np.cos(q_2)
        P_BS[2,2] = -np.sin(q_1)*np.sin(q_3) + np.cos(q_1)*np.cos(q_2)*np.cos(q_3)
        P_BS[2,3] = -L_1*np.cos(q_1) - L_2*np.cos(q_1)*np.cos(q_2) + L_3*np.sin(q_1)*np.sin(q_3) - L_3*np.cos(q_1)*np.cos(q_2)*np.cos(q_3)
        P_BS[3,0] = 0
        P_BS[3,1] = 0
        P_BS[3,2] = 0
        P_BS[3,3] = 1
        return P_BS

    ''' Evaluate transformation matrix of contact frame in inertial frame
    '''
    def evaluate_P_C(self, state, alpha, beta, d):
        x_B = state[0]
        y_B = state[1]
        z_B = state[2]
        roll = state[3]
        pitch =state[4]
        yaw = state[5]
        q_1 = state[6]
        q_2 = state[7]
        q_3 = state[8]
        P_C = np.zeros((4,4))
        P_C[0,0] = np.sin(beta)*np.sin(pitch)*np.sin(alpha - q_3)*np.sin(q_1 + roll)*np.cos(yaw) + np.sin(beta)*np.sin(pitch)*np.cos(q_2)*np.cos(yaw)*np.cos(alpha - q_3)*np.cos(q_1 + roll) + np.sin(beta)*np.sin(q_2)*np.cos(pitch)*np.cos(yaw)*np.cos(alpha - q_3) - np.sin(beta)*np.sin(yaw)*np.sin(alpha - q_3)*np.cos(q_1 + roll) + np.sin(beta)*np.sin(yaw)*np.sin(q_1 + roll)*np.cos(q_2)*np.cos(alpha - q_3) - np.sin(pitch)*np.sin(q_2)*np.cos(beta)*np.cos(yaw)*np.cos(q_1 + roll) - np.sin(q_2)*np.sin(yaw)*np.sin(q_1 + roll)*np.cos(beta) + np.cos(beta)*np.cos(pitch)*np.cos(q_2)*np.cos(yaw)
        P_C[0,1] = -np.sin(pitch)*np.sin(alpha - q_3)*np.cos(q_2)*np.cos(yaw)*np.cos(q_1 + roll) + np.sin(pitch)*np.sin(q_1 + roll)*np.cos(yaw)*np.cos(alpha - q_3) - np.sin(q_2)*np.sin(alpha - q_3)*np.cos(pitch)*np.cos(yaw) - np.sin(yaw)*np.sin(alpha - q_3)*np.sin(q_1 + roll)*np.cos(q_2) - np.sin(yaw)*np.cos(alpha - q_3)*np.cos(q_1 + roll)
        P_C[0,2] = np.sin(beta)*np.sin(pitch)*np.sin(q_2)*np.cos(yaw)*np.cos(q_1 + roll) + np.sin(beta)*np.sin(q_2)*np.sin(yaw)*np.sin(q_1 + roll) - np.sin(beta)*np.cos(pitch)*np.cos(q_2)*np.cos(yaw) + np.sin(pitch)*np.sin(alpha - q_3)*np.sin(q_1 + roll)*np.cos(beta)*np.cos(yaw) + np.sin(pitch)*np.cos(beta)*np.cos(q_2)*np.cos(yaw)*np.cos(alpha - q_3)*np.cos(q_1 + roll) + np.sin(q_2)*np.cos(beta)*np.cos(pitch)*np.cos(yaw)*np.cos(alpha - q_3) - np.sin(yaw)*np.sin(alpha - q_3)*np.cos(beta)*np.cos(q_1 + roll) + np.sin(yaw)*np.sin(q_1 + roll)*np.cos(beta)*np.cos(q_2)*np.cos(alpha - q_3)
        P_C[0,3] = -0.11*np.sin(pitch)*np.cos(yaw)*np.cos(q_1 + roll) - 0.11*np.sin(yaw)*np.sin(q_1 + roll) - L_2*np.sin(pitch)*np.cos(q_2)*np.cos(yaw)*np.cos(q_1 + roll) - L_2*np.sin(q_2)*np.cos(pitch)*np.cos(yaw) - L_2*np.sin(yaw)*np.sin(q_1 + roll)*np.cos(q_2) + L_3*np.sin(pitch)*np.sin(q_3)*np.sin(q_1 + roll)*np.cos(yaw) - L_3*np.sin(pitch)*np.cos(q_2)*np.cos(q_3)*np.cos(yaw)*np.cos(q_1 + roll) - L_3*np.sin(q_2)*np.cos(pitch)*np.cos(q_3)*np.cos(yaw) - L_3*np.sin(q_3)*np.sin(yaw)*np.cos(q_1 + roll) - L_3*np.sin(yaw)*np.sin(q_1 + roll)*np.cos(q_2)*np.cos(q_3) + x_B - d*np.sin(beta)*np.sin(pitch)*np.sin(q_2)*np.cos(yaw)*np.cos(q_1 + roll) - d*np.sin(beta)*np.sin(q_2)*np.sin(yaw)*np.sin(q_1 + roll) + d*np.sin(beta)*np.cos(pitch)*np.cos(q_2)*np.cos(yaw) - d*np.sin(pitch)*np.sin(alpha - q_3)*np.sin(q_1 + roll)*np.cos(beta)*np.cos(yaw) - d*np.sin(pitch)*np.cos(beta)*np.cos(q_2)*np.cos(yaw)*np.cos(alpha - q_3)*np.cos(q_1 + roll) - d*np.sin(q_2)*np.cos(beta)*np.cos(pitch)*np.cos(yaw)*np.cos(alpha - q_3) + d*np.sin(yaw)*np.sin(alpha - q_3)*np.cos(beta)*np.cos(q_1 + roll) - d*np.sin(yaw)*np.sin(q_1 + roll)*np.cos(beta)*np.cos(q_2)*np.cos(alpha - q_3)
        P_C[1,0] = np.sin(beta)*np.sin(pitch)*np.sin(yaw)*np.sin(alpha - q_3)*np.sin(q_1 + roll) + np.sin(beta)*np.sin(pitch)*np.sin(yaw)*np.cos(q_2)*np.cos(alpha - q_3)*np.cos(q_1 + roll) + np.sin(beta)*np.sin(q_2)*np.sin(yaw)*np.cos(pitch)*np.cos(alpha - q_3) + np.sin(beta)*np.sin(alpha - q_3)*np.cos(yaw)*np.cos(q_1 + roll) - np.sin(beta)*np.sin(q_1 + roll)*np.cos(q_2)*np.cos(yaw)*np.cos(alpha - q_3) - np.sin(pitch)*np.sin(q_2)*np.sin(yaw)*np.cos(beta)*np.cos(q_1 + roll) + np.sin(q_2)*np.sin(q_1 + roll)*np.cos(beta)*np.cos(yaw) + np.sin(yaw)*np.cos(beta)*np.cos(pitch)*np.cos(q_2)
        P_C[1,1] = -np.sin(pitch)*np.sin(yaw)*np.sin(alpha - q_3)*np.cos(q_2)*np.cos(q_1 + roll) + np.sin(pitch)*np.sin(yaw)*np.sin(q_1 + roll)*np.cos(alpha - q_3) - np.sin(q_2)*np.sin(yaw)*np.sin(alpha - q_3)*np.cos(pitch) + np.sin(alpha - q_3)*np.sin(q_1 + roll)*np.cos(q_2)*np.cos(yaw) + np.cos(yaw)*np.cos(alpha - q_3)*np.cos(q_1 + roll)
        P_C[1,2] = np.sin(beta)*np.sin(pitch)*np.sin(q_2)*np.sin(yaw)*np.cos(q_1 + roll) - np.sin(beta)*np.sin(q_2)*np.sin(q_1 + roll)*np.cos(yaw) - np.sin(beta)*np.sin(yaw)*np.cos(pitch)*np.cos(q_2) + np.sin(pitch)*np.sin(yaw)*np.sin(alpha - q_3)*np.sin(q_1 + roll)*np.cos(beta) + np.sin(pitch)*np.sin(yaw)*np.cos(beta)*np.cos(q_2)*np.cos(alpha - q_3)*np.cos(q_1 + roll) + np.sin(q_2)*np.sin(yaw)*np.cos(beta)*np.cos(pitch)*np.cos(alpha - q_3) + np.sin(alpha - q_3)*np.cos(beta)*np.cos(yaw)*np.cos(q_1 + roll) - np.sin(q_1 + roll)*np.cos(beta)*np.cos(q_2)*np.cos(yaw)*np.cos(alpha - q_3)
        P_C[1,3] = -0.11*np.sin(pitch)*np.sin(yaw)*np.cos(q_1 + roll) + 0.11*np.sin(q_1 + roll)*np.cos(yaw) - L_2*np.sin(pitch)*np.sin(yaw)*np.cos(q_2)*np.cos(q_1 + roll) - L_2*np.sin(q_2)*np.sin(yaw)*np.cos(pitch) + L_2*np.sin(q_1 + roll)*np.cos(q_2)*np.cos(yaw) + L_3*np.sin(pitch)*np.sin(q_3)*np.sin(yaw)*np.sin(q_1 + roll) - L_3*np.sin(pitch)*np.sin(yaw)*np.cos(q_2)*np.cos(q_3)*np.cos(q_1 + roll) - L_3*np.sin(q_2)*np.sin(yaw)*np.cos(pitch)*np.cos(q_3) + L_3*np.sin(q_3)*np.cos(yaw)*np.cos(q_1 + roll) + L_3*np.sin(q_1 + roll)*np.cos(q_2)*np.cos(q_3)*np.cos(yaw) + y_B - d*np.sin(beta)*np.sin(pitch)*np.sin(q_2)*np.sin(yaw)*np.cos(q_1 + roll) + d*np.sin(beta)*np.sin(q_2)*np.sin(q_1 + roll)*np.cos(yaw) + d*np.sin(beta)*np.sin(yaw)*np.cos(pitch)*np.cos(q_2) - d*np.sin(pitch)*np.sin(yaw)*np.sin(alpha - q_3)*np.sin(q_1 + roll)*np.cos(beta) - d*np.sin(pitch)*np.sin(yaw)*np.cos(beta)*np.cos(q_2)*np.cos(alpha - q_3)*np.cos(q_1 + roll) - d*np.sin(q_2)*np.sin(yaw)*np.cos(beta)*np.cos(pitch)*np.cos(alpha - q_3) - d*np.sin(alpha - q_3)*np.cos(beta)*np.cos(yaw)*np.cos(q_1 + roll) + d*np.sin(q_1 + roll)*np.cos(beta)*np.cos(q_2)*np.cos(yaw)*np.cos(alpha - q_3)
        P_C[2,0] = -np.sin(beta)*np.sin(pitch)*np.sin(q_2)*np.cos(alpha - q_3) + np.sin(beta)*np.sin(alpha - q_3)*np.sin(q_1 + roll)*np.cos(pitch) + np.sin(beta)*np.cos(pitch)*np.cos(q_2)*np.cos(alpha - q_3)*np.cos(q_1 + roll) - np.sin(pitch)*np.cos(beta)*np.cos(q_2) - np.sin(q_2)*np.cos(beta)*np.cos(pitch)*np.cos(q_1 + roll)
        P_C[2,1] = np.sin(pitch)*np.sin(q_2)*np.sin(alpha - q_3) - np.sin(alpha - q_3)*np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll) + np.sin(q_1 + roll)*np.cos(pitch)*np.cos(alpha - q_3)
        P_C[2,2] = np.sin(beta)*np.sin(pitch)*np.cos(q_2) + np.sin(beta)*np.sin(q_2)*np.cos(pitch)*np.cos(q_1 + roll) - np.sin(pitch)*np.sin(q_2)*np.cos(beta)*np.cos(alpha - q_3) + np.sin(alpha - q_3)*np.sin(q_1 + roll)*np.cos(beta)*np.cos(pitch) + np.cos(beta)*np.cos(pitch)*np.cos(q_2)*np.cos(alpha - q_3)*np.cos(q_1 + roll)
        P_C[2,3] = -0.11*np.cos(pitch)*np.cos(q_1 + roll) + L_2*np.sin(pitch)*np.sin(q_2) - L_2*np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll) + L_3*np.sin(pitch)*np.sin(q_2)*np.cos(q_3) + L_3*np.sin(q_3)*np.sin(q_1 + roll)*np.cos(pitch) - L_3*np.cos(pitch)*np.cos(q_2)*np.cos(q_3)*np.cos(q_1 + roll) + z_B - d*np.sin(beta)*np.sin(pitch)*np.cos(q_2) - d*np.sin(beta)*np.sin(q_2)*np.cos(pitch)*np.cos(q_1 + roll) + d*np.sin(pitch)*np.sin(q_2)*np.cos(beta)*np.cos(alpha - q_3) - d*np.sin(alpha - q_3)*np.sin(q_1 + roll)*np.cos(beta)*np.cos(pitch) - d*np.cos(beta)*np.cos(pitch)*np.cos(q_2)*np.cos(alpha - q_3)*np.cos(q_1 + roll)
        P_C[3,0] = 0
        P_C[3,1] = 0
        P_C[3,2] = 0
        P_C[3,3] = 1

        return P_C

    ''' Get HTM describing end-effector (sensor) pose in inertial frame -> P_S, evaluated at latest state
    '''
    def evaluate_P_S(self, state):
        x_B = state[0]
        y_B = state[1]
        z_B = state[2]
        roll = state[3]
        pitch = state[4]
        yaw = state[5]
        q_1 = state[6]
        q_2 = state[7]
        q_3 = state[8]

        P_S = np.zeros((4,4))
        P_S[0,0] = -(np.sin(pitch)*np.cos(yaw)*np.cos(q_1 + roll) + np.sin(yaw)*np.sin(q_1 + roll))*np.sin(q_2) + np.cos(pitch)*np.cos(q_2)*np.cos(yaw)
        P_S[0,1] = -(-(np.sin(pitch)*np.cos(yaw)*np.cos(q_1 + roll) + np.sin(yaw)*np.sin(q_1 + roll))*np.cos(q_2) - np.sin(q_2)*np.cos(pitch)*np.cos(yaw))*np.sin(q_3) + (np.sin(pitch)*np.sin(q_1 + roll)*np.cos(yaw) - np.sin(yaw)*np.cos(q_1 + roll))*np.cos(q_3)
        P_S[0,2] = -(-(np.sin(pitch)*np.cos(yaw)*np.cos(q_1 + roll) + np.sin(yaw)*np.sin(q_1 + roll))*np.cos(q_2) - np.sin(q_2)*np.cos(pitch)*np.cos(yaw))*np.cos(q_3) - (np.sin(pitch)*np.sin(q_1 + roll)*np.cos(yaw) - np.sin(yaw)*np.cos(q_1 + roll))*np.sin(q_3)
        P_S[0,3] = -0.11*np.sin(pitch)*np.cos(yaw)*np.cos(q_1 + roll) - 0.11*np.sin(yaw)*np.sin(q_1 + roll) - L_2*np.sin(pitch)*np.cos(q_2)*np.cos(yaw)*np.cos(q_1 + roll) - L_2*np.sin(q_2)*np.cos(pitch)*np.cos(yaw) - L_2*np.sin(yaw)*np.sin(q_1 + roll)*np.cos(q_2) + L_3*np.sin(pitch)*np.sin(q_3)*np.sin(q_1 + roll)*np.cos(yaw) - L_3*np.sin(pitch)*np.cos(q_2)*np.cos(q_3)*np.cos(yaw)*np.cos(q_1 + roll) - L_3*np.sin(q_2)*np.cos(pitch)*np.cos(q_3)*np.cos(yaw) - L_3*np.sin(q_3)*np.sin(yaw)*np.cos(q_1 + roll) - L_3*np.sin(yaw)*np.sin(q_1 + roll)*np.cos(q_2)*np.cos(q_3) + x_B
        P_S[1,0] = (-np.sin(pitch)*np.sin(yaw)*np.cos(q_1 + roll) + np.sin(q_1 + roll)*np.cos(yaw))*np.sin(q_2) + np.sin(yaw)*np.cos(pitch)*np.cos(q_2)
        P_S[1,1] = -((-np.sin(pitch)*np.sin(yaw)*np.cos(q_1 + roll) + np.sin(q_1 + roll)*np.cos(yaw))*np.cos(q_2) - np.sin(q_2)*np.sin(yaw)*np.cos(pitch))*np.sin(q_3) + (np.sin(pitch)*np.sin(yaw)*np.sin(q_1 + roll) + np.cos(yaw)*np.cos(q_1 + roll))*np.cos(q_3)
        P_S[1,2] = -((-np.sin(pitch)*np.sin(yaw)*np.cos(q_1 + roll) + np.sin(q_1 + roll)*np.cos(yaw))*np.cos(q_2) - np.sin(q_2)*np.sin(yaw)*np.cos(pitch))*np.cos(q_3) - (np.sin(pitch)*np.sin(yaw)*np.sin(q_1 + roll) + np.cos(yaw)*np.cos(q_1 + roll))*np.sin(q_3)
        P_S[1,3] = -0.11*np.sin(pitch)*np.sin(yaw)*np.cos(q_1 + roll) + 0.11*np.sin(q_1 + roll)*np.cos(yaw) - L_2*np.sin(pitch)*np.sin(yaw)*np.cos(q_2)*np.cos(q_1 + roll) - L_2*np.sin(q_2)*np.sin(yaw)*np.cos(pitch) + L_2*np.sin(q_1 + roll)*np.cos(q_2)*np.cos(yaw) + L_3*np.sin(pitch)*np.sin(q_3)*np.sin(yaw)*np.sin(q_1 + roll) - L_3*np.sin(pitch)*np.sin(yaw)*np.cos(q_2)*np.cos(q_3)*np.cos(q_1 + roll) - L_3*np.sin(q_2)*np.sin(yaw)*np.cos(pitch)*np.cos(q_3) + L_3*np.sin(q_3)*np.cos(yaw)*np.cos(q_1 + roll) + L_3*np.sin(q_1 + roll)*np.cos(q_2)*np.cos(q_3)*np.cos(yaw) + y_B
        P_S[2,0] = -np.sin(pitch)*np.cos(q_2) - np.sin(q_2)*np.cos(pitch)*np.cos(q_1 + roll)
        P_S[2,1] = -(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.sin(q_3) + np.sin(q_1 + roll)*np.cos(pitch)*np.cos(q_3)
        P_S[2,2] = -(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.cos(q_3) - np.sin(q_3)*np.sin(q_1 + roll)*np.cos(pitch)
        P_S[2,3] = -0.11*np.cos(pitch)*np.cos(q_1 + roll) + L_2*np.sin(pitch)*np.sin(q_2) - L_2*np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll) + L_3*np.sin(pitch)*np.sin(q_2)*np.cos(q_3) + L_3*np.sin(q_3)*np.sin(q_1 + roll)*np.cos(pitch) - L_3*np.cos(pitch)*np.cos(q_2)*np.cos(q_3)*np.cos(q_1 + roll) + z_B
        P_S[3,0] = 0
        P_S[3,1] = 0
        P_S[3,2] = 0
        P_S[3,3] = 1

        return P_S

    def evaluate_Jc(self, roll, pitch, yaw, q_1, q_2, q_3):
        Jc = np.zeros((6,9))
        Jc[0,0]=1
        Jc[0,3]=L_1*np.sin(pitch)*np.sin(q_1 + roll)*np.cos(yaw) - L_1*np.sin(yaw)*np.cos(q_1 + roll) + L_2*np.sin(pitch)*np.sin(q_1 + roll)*np.cos(q_2)*np.cos(yaw) - L_2*np.sin(yaw)*np.cos(q_2)*np.cos(q_1 + roll) + L_3*np.sin(pitch)*np.sin(q_3)*np.cos(yaw)*np.cos(q_1 + roll) + L_3*np.sin(pitch)*np.sin(q_1 + roll)*np.cos(q_2)*np.cos(q_3)*np.cos(yaw) + L_3*np.sin(q_3)*np.sin(yaw)*np.sin(q_1 + roll) - L_3*np.sin(yaw)*np.cos(q_2)*np.cos(q_3)*np.cos(q_1 + roll)
        Jc[0,4]=-L_1*np.cos(pitch)*np.cos(yaw)*np.cos(q_1 + roll) + L_2*np.sin(pitch)*np.sin(q_2)*np.cos(yaw) - L_2*np.cos(pitch)*np.cos(q_2)*np.cos(yaw)*np.cos(q_1 + roll) + L_3*np.sin(pitch)*np.sin(q_2)*np.cos(q_3)*np.cos(yaw) + L_3*np.sin(q_3)*np.sin(q_1 + roll)*np.cos(pitch)*np.cos(yaw) - L_3*np.cos(pitch)*np.cos(q_2)*np.cos(q_3)*np.cos(yaw)*np.cos(q_1 + roll)
        Jc[0,5]=L_1*np.sin(pitch)*np.sin(yaw)*np.cos(q_1 + roll) - L_1*np.sin(q_1 + roll)*np.cos(yaw) + L_2*np.sin(pitch)*np.sin(yaw)*np.cos(q_2)*np.cos(q_1 + roll) + L_2*np.sin(q_2)*np.sin(yaw)*np.cos(pitch) - L_2*np.sin(q_1 + roll)*np.cos(q_2)*np.cos(yaw) - L_3*np.sin(pitch)*np.sin(q_3)*np.sin(yaw)*np.sin(q_1 + roll) + L_3*np.sin(pitch)*np.sin(yaw)*np.cos(q_2)*np.cos(q_3)*np.cos(q_1 + roll) + L_3*np.sin(q_2)*np.sin(yaw)*np.cos(pitch)*np.cos(q_3) - L_3*np.sin(q_3)*np.cos(yaw)*np.cos(q_1 + roll) - L_3*np.sin(q_1 + roll)*np.cos(q_2)*np.cos(q_3)*np.cos(yaw)
        Jc[0,6]=L_1*np.sin(pitch)*np.sin(q_1 + roll)*np.cos(yaw) - L_1*np.sin(yaw)*np.cos(q_1 + roll) + L_2*np.sin(pitch)*np.sin(q_1 + roll)*np.cos(q_2)*np.cos(yaw) - L_2*np.sin(yaw)*np.cos(q_2)*np.cos(q_1 + roll) + L_3*np.sin(pitch)*np.sin(q_3)*np.cos(yaw)*np.cos(q_1 + roll) + L_3*np.sin(pitch)*np.sin(q_1 + roll)*np.cos(q_2)*np.cos(q_3)*np.cos(yaw) + L_3*np.sin(q_3)*np.sin(yaw)*np.sin(q_1 + roll) - L_3*np.sin(yaw)*np.cos(q_2)*np.cos(q_3)*np.cos(q_1 + roll)
        Jc[0,7]=L_2*np.sin(pitch)*np.sin(q_2)*np.cos(yaw)*np.cos(q_1 + roll) + L_2*np.sin(q_2)*np.sin(yaw)*np.sin(q_1 + roll) - L_2*np.cos(pitch)*np.cos(q_2)*np.cos(yaw) + L_3*np.sin(pitch)*np.sin(q_2)*np.cos(q_3)*np.cos(yaw)*np.cos(q_1 + roll) + L_3*np.sin(q_2)*np.sin(yaw)*np.sin(q_1 + roll)*np.cos(q_3) - L_3*np.cos(pitch)*np.cos(q_2)*np.cos(q_3)*np.cos(yaw)
        Jc[0,8]=L_3*np.sin(pitch)*np.sin(q_3)*np.cos(q_2)*np.cos(yaw)*np.cos(q_1 + roll) + L_3*np.sin(pitch)*np.sin(q_1 + roll)*np.cos(q_3)*np.cos(yaw) + L_3*np.sin(q_2)*np.sin(q_3)*np.cos(pitch)*np.cos(yaw) + L_3*np.sin(q_3)*np.sin(yaw)*np.sin(q_1 + roll)*np.cos(q_2) - L_3*np.sin(yaw)*np.cos(q_3)*np.cos(q_1 + roll)
        Jc[1,1]=1
        Jc[1,3]=L_1*np.sin(pitch)*np.sin(yaw)*np.sin(q_1 + roll) + L_1*np.cos(yaw)*np.cos(q_1 + roll) + L_2*np.sin(pitch)*np.sin(yaw)*np.sin(q_1 + roll)*np.cos(q_2) + L_2*np.cos(q_2)*np.cos(yaw)*np.cos(q_1 + roll) + L_3*np.sin(pitch)*np.sin(q_3)*np.sin(yaw)*np.cos(q_1 + roll) + L_3*np.sin(pitch)*np.sin(yaw)*np.sin(q_1 + roll)*np.cos(q_2)*np.cos(q_3) - L_3*np.sin(q_3)*np.sin(q_1 + roll)*np.cos(yaw) + L_3*np.cos(q_2)*np.cos(q_3)*np.cos(yaw)*np.cos(q_1 + roll)
        Jc[1,4]=-L_1*np.sin(yaw)*np.cos(pitch)*np.cos(q_1 + roll) + L_2*np.sin(pitch)*np.sin(q_2)*np.sin(yaw) - L_2*np.sin(yaw)*np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll) + L_3*np.sin(pitch)*np.sin(q_2)*np.sin(yaw)*np.cos(q_3) + L_3*np.sin(q_3)*np.sin(yaw)*np.sin(q_1 + roll)*np.cos(pitch) - L_3*np.sin(yaw)*np.cos(pitch)*np.cos(q_2)*np.cos(q_3)*np.cos(q_1 + roll)
        Jc[1,5]=-L_1*np.sin(pitch)*np.cos(yaw)*np.cos(q_1 + roll) - L_1*np.sin(yaw)*np.sin(q_1 + roll) - L_2*np.sin(pitch)*np.cos(q_2)*np.cos(yaw)*np.cos(q_1 + roll) - L_2*np.sin(q_2)*np.cos(pitch)*np.cos(yaw) - L_2*np.sin(yaw)*np.sin(q_1 + roll)*np.cos(q_2) + L_3*np.sin(pitch)*np.sin(q_3)*np.sin(q_1 + roll)*np.cos(yaw) - L_3*np.sin(pitch)*np.cos(q_2)*np.cos(q_3)*np.cos(yaw)*np.cos(q_1 + roll) - L_3*np.sin(q_2)*np.cos(pitch)*np.cos(q_3)*np.cos(yaw) - L_3*np.sin(q_3)*np.sin(yaw)*np.cos(q_1 + roll) - L_3*np.sin(yaw)*np.sin(q_1 + roll)*np.cos(q_2)*np.cos(q_3)
        Jc[1,6]=L_1*np.sin(pitch)*np.sin(yaw)*np.sin(q_1 + roll) + L_1*np.cos(yaw)*np.cos(q_1 + roll) + L_2*np.sin(pitch)*np.sin(yaw)*np.sin(q_1 + roll)*np.cos(q_2) + L_2*np.cos(q_2)*np.cos(yaw)*np.cos(q_1 + roll) + L_3*np.sin(pitch)*np.sin(q_3)*np.sin(yaw)*np.cos(q_1 + roll) + L_3*np.sin(pitch)*np.sin(yaw)*np.sin(q_1 + roll)*np.cos(q_2)*np.cos(q_3) - L_3*np.sin(q_3)*np.sin(q_1 + roll)*np.cos(yaw) + L_3*np.cos(q_2)*np.cos(q_3)*np.cos(yaw)*np.cos(q_1 + roll)
        Jc[1,7]=L_2*np.sin(pitch)*np.sin(q_2)*np.sin(yaw)*np.cos(q_1 + roll) - L_2*np.sin(q_2)*np.sin(q_1 + roll)*np.cos(yaw) - L_2*np.sin(yaw)*np.cos(pitch)*np.cos(q_2) + L_3*np.sin(pitch)*np.sin(q_2)*np.sin(yaw)*np.cos(q_3)*np.cos(q_1 + roll) - L_3*np.sin(q_2)*np.sin(q_1 + roll)*np.cos(q_3)*np.cos(yaw) - L_3*np.sin(yaw)*np.cos(pitch)*np.cos(q_2)*np.cos(q_3)
        Jc[1,8]=L_3*np.sin(pitch)*np.sin(q_3)*np.sin(yaw)*np.cos(q_2)*np.cos(q_1 + roll) + L_3*np.sin(pitch)*np.sin(yaw)*np.sin(q_1 + roll)*np.cos(q_3) + L_3*np.sin(q_2)*np.sin(q_3)*np.sin(yaw)*np.cos(pitch) - L_3*np.sin(q_3)*np.sin(q_1 + roll)*np.cos(q_2)*np.cos(yaw) + L_3*np.cos(q_3)*np.cos(yaw)*np.cos(q_1 + roll)
        Jc[2,2]=1
        Jc[2,3]=L_1*np.sin(q_1 + roll)*np.cos(pitch) + L_2*np.sin(q_1 + roll)*np.cos(pitch)*np.cos(q_2) + L_3*np.sin(q_3)*np.cos(pitch)*np.cos(q_1 + roll) + L_3*np.sin(q_1 + roll)*np.cos(pitch)*np.cos(q_2)*np.cos(q_3)
        Jc[2,4]=L_1*np.sin(pitch)*np.cos(q_1 + roll) + L_2*np.sin(pitch)*np.cos(q_2)*np.cos(q_1 + roll) + L_2*np.sin(q_2)*np.cos(pitch) - L_3*np.sin(pitch)*np.sin(q_3)*np.sin(q_1 + roll) + L_3*np.sin(pitch)*np.cos(q_2)*np.cos(q_3)*np.cos(q_1 + roll) + L_3*np.sin(q_2)*np.cos(pitch)*np.cos(q_3)
        Jc[2,6]=L_1*np.sin(q_1 + roll)*np.cos(pitch) + L_2*np.sin(q_1 + roll)*np.cos(pitch)*np.cos(q_2) + L_3*np.sin(q_3)*np.cos(pitch)*np.cos(q_1 + roll) + L_3*np.sin(q_1 + roll)*np.cos(pitch)*np.cos(q_2)*np.cos(q_3)
        Jc[2,7]=L_2*np.sin(pitch)*np.cos(q_2) + L_2*np.sin(q_2)*np.cos(pitch)*np.cos(q_1 + roll) + L_3*np.sin(pitch)*np.cos(q_2)*np.cos(q_3) + L_3*np.sin(q_2)*np.cos(pitch)*np.cos(q_3)*np.cos(q_1 + roll)
        Jc[2,8]=-L_3*np.sin(pitch)*np.sin(q_2)*np.sin(q_3) + L_3*np.sin(q_3)*np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll) + L_3*np.sin(q_1 + roll)*np.cos(pitch)*np.cos(q_3)
        Jc[3,3]=-(-(-np.sin(pitch)*np.sin(yaw)*np.cos(q_1 + roll) + np.sin(q_1 + roll)*np.cos(yaw))*np.sin(q_2) - np.sin(yaw)*np.cos(pitch)*np.cos(q_2))*(-np.sin(pitch)*np.sin(q_1 + roll)*np.cos(yaw) + np.sin(yaw)*np.cos(q_1 + roll))*np.sin(q_2)/(((-np.sin(pitch)*np.sin(yaw)*np.cos(q_1 + roll) + np.sin(q_1 + roll)*np.cos(yaw))*np.sin(q_2) + np.sin(yaw)*np.cos(pitch)*np.cos(q_2))**2 + (-(np.sin(pitch)*np.cos(yaw)*np.cos(q_1 + roll) + np.sin(yaw)*np.sin(q_1 + roll))*np.sin(q_2) + np.cos(pitch)*np.cos(q_2)*np.cos(yaw))**2) + (-(np.sin(pitch)*np.cos(yaw)*np.cos(q_1 + roll) + np.sin(yaw)*np.sin(q_1 + roll))*np.sin(q_2) + np.cos(pitch)*np.cos(q_2)*np.cos(yaw))*(np.sin(pitch)*np.sin(yaw)*np.sin(q_1 + roll) + np.cos(yaw)*np.cos(q_1 + roll))*np.sin(q_2)/(((-np.sin(pitch)*np.sin(yaw)*np.cos(q_1 + roll) + np.sin(q_1 + roll)*np.cos(yaw))*np.sin(q_2) + np.sin(yaw)*np.cos(pitch)*np.cos(q_2))**2 + (-(np.sin(pitch)*np.cos(yaw)*np.cos(q_1 + roll) + np.sin(yaw)*np.sin(q_1 + roll))*np.sin(q_2) + np.cos(pitch)*np.cos(q_2)*np.cos(yaw))**2)
        Jc[3,4]=(-(-np.sin(pitch)*np.sin(yaw)*np.cos(q_1 + roll) + np.sin(q_1 + roll)*np.cos(yaw))*np.sin(q_2) - np.sin(yaw)*np.cos(pitch)*np.cos(q_2))*(-np.sin(pitch)*np.cos(q_2)*np.cos(yaw) - np.sin(q_2)*np.cos(pitch)*np.cos(yaw)*np.cos(q_1 + roll))/(((-np.sin(pitch)*np.sin(yaw)*np.cos(q_1 + roll) + np.sin(q_1 + roll)*np.cos(yaw))*np.sin(q_2) + np.sin(yaw)*np.cos(pitch)*np.cos(q_2))**2 + (-(np.sin(pitch)*np.cos(yaw)*np.cos(q_1 + roll) + np.sin(yaw)*np.sin(q_1 + roll))*np.sin(q_2) + np.cos(pitch)*np.cos(q_2)*np.cos(yaw))**2) + (-(np.sin(pitch)*np.cos(yaw)*np.cos(q_1 + roll) + np.sin(yaw)*np.sin(q_1 + roll))*np.sin(q_2) + np.cos(pitch)*np.cos(q_2)*np.cos(yaw))*(-np.sin(pitch)*np.sin(yaw)*np.cos(q_2) - np.sin(q_2)*np.sin(yaw)*np.cos(pitch)*np.cos(q_1 + roll))/(((-np.sin(pitch)*np.sin(yaw)*np.cos(q_1 + roll) + np.sin(q_1 + roll)*np.cos(yaw))*np.sin(q_2) + np.sin(yaw)*np.cos(pitch)*np.cos(q_2))**2 + (-(np.sin(pitch)*np.cos(yaw)*np.cos(q_1 + roll) + np.sin(yaw)*np.sin(q_1 + roll))*np.sin(q_2) + np.cos(pitch)*np.cos(q_2)*np.cos(yaw))**2)
        Jc[3,5]=(-(-np.sin(pitch)*np.sin(yaw)*np.cos(q_1 + roll) + np.sin(q_1 + roll)*np.cos(yaw))*np.sin(q_2) - np.sin(yaw)*np.cos(pitch)*np.cos(q_2))**2/(((-np.sin(pitch)*np.sin(yaw)*np.cos(q_1 + roll) + np.sin(q_1 + roll)*np.cos(yaw))*np.sin(q_2) + np.sin(yaw)*np.cos(pitch)*np.cos(q_2))**2 + (-(np.sin(pitch)*np.cos(yaw)*np.cos(q_1 + roll) + np.sin(yaw)*np.sin(q_1 + roll))*np.sin(q_2) + np.cos(pitch)*np.cos(q_2)*np.cos(yaw))**2) + ((-np.sin(pitch)*np.cos(yaw)*np.cos(q_1 + roll) - np.sin(yaw)*np.sin(q_1 + roll))*np.sin(q_2) + np.cos(pitch)*np.cos(q_2)*np.cos(yaw))*(-(np.sin(pitch)*np.cos(yaw)*np.cos(q_1 + roll) + np.sin(yaw)*np.sin(q_1 + roll))*np.sin(q_2) + np.cos(pitch)*np.cos(q_2)*np.cos(yaw))/(((-np.sin(pitch)*np.sin(yaw)*np.cos(q_1 + roll) + np.sin(q_1 + roll)*np.cos(yaw))*np.sin(q_2) + np.sin(yaw)*np.cos(pitch)*np.cos(q_2))**2 + (-(np.sin(pitch)*np.cos(yaw)*np.cos(q_1 + roll) + np.sin(yaw)*np.sin(q_1 + roll))*np.sin(q_2) + np.cos(pitch)*np.cos(q_2)*np.cos(yaw))**2)
        Jc[3,6]=-(-(-np.sin(pitch)*np.sin(yaw)*np.cos(q_1 + roll) + np.sin(q_1 + roll)*np.cos(yaw))*np.sin(q_2) - np.sin(yaw)*np.cos(pitch)*np.cos(q_2))*(-np.sin(pitch)*np.sin(q_1 + roll)*np.cos(yaw) + np.sin(yaw)*np.cos(q_1 + roll))*np.sin(q_2)/(((-np.sin(pitch)*np.sin(yaw)*np.cos(q_1 + roll) + np.sin(q_1 + roll)*np.cos(yaw))*np.sin(q_2) + np.sin(yaw)*np.cos(pitch)*np.cos(q_2))**2 + (-(np.sin(pitch)*np.cos(yaw)*np.cos(q_1 + roll) + np.sin(yaw)*np.sin(q_1 + roll))*np.sin(q_2) + np.cos(pitch)*np.cos(q_2)*np.cos(yaw))**2) + (-(np.sin(pitch)*np.cos(yaw)*np.cos(q_1 + roll) + np.sin(yaw)*np.sin(q_1 + roll))*np.sin(q_2) + np.cos(pitch)*np.cos(q_2)*np.cos(yaw))*(np.sin(pitch)*np.sin(yaw)*np.sin(q_1 + roll) + np.cos(yaw)*np.cos(q_1 + roll))*np.sin(q_2)/(((-np.sin(pitch)*np.sin(yaw)*np.cos(q_1 + roll) + np.sin(q_1 + roll)*np.cos(yaw))*np.sin(q_2) + np.sin(yaw)*np.cos(pitch)*np.cos(q_2))**2 + (-(np.sin(pitch)*np.cos(yaw)*np.cos(q_1 + roll) + np.sin(yaw)*np.sin(q_1 + roll))*np.sin(q_2) + np.cos(pitch)*np.cos(q_2)*np.cos(yaw))**2)
        Jc[3,7]=(-(-np.sin(pitch)*np.sin(yaw)*np.cos(q_1 + roll) + np.sin(q_1 + roll)*np.cos(yaw))*np.sin(q_2) - np.sin(yaw)*np.cos(pitch)*np.cos(q_2))*(-(np.sin(pitch)*np.cos(yaw)*np.cos(q_1 + roll) + np.sin(yaw)*np.sin(q_1 + roll))*np.cos(q_2) - np.sin(q_2)*np.cos(pitch)*np.cos(yaw))/(((-np.sin(pitch)*np.sin(yaw)*np.cos(q_1 + roll) + np.sin(q_1 + roll)*np.cos(yaw))*np.sin(q_2) + np.sin(yaw)*np.cos(pitch)*np.cos(q_2))**2 + (-(np.sin(pitch)*np.cos(yaw)*np.cos(q_1 + roll) + np.sin(yaw)*np.sin(q_1 + roll))*np.sin(q_2) + np.cos(pitch)*np.cos(q_2)*np.cos(yaw))**2) + ((-np.sin(pitch)*np.sin(yaw)*np.cos(q_1 + roll) + np.sin(q_1 + roll)*np.cos(yaw))*np.cos(q_2) - np.sin(q_2)*np.sin(yaw)*np.cos(pitch))*(-(np.sin(pitch)*np.cos(yaw)*np.cos(q_1 + roll) + np.sin(yaw)*np.sin(q_1 + roll))*np.sin(q_2) + np.cos(pitch)*np.cos(q_2)*np.cos(yaw))/(((-np.sin(pitch)*np.sin(yaw)*np.cos(q_1 + roll) + np.sin(q_1 + roll)*np.cos(yaw))*np.sin(q_2) + np.sin(yaw)*np.cos(pitch)*np.cos(q_2))**2 + (-(np.sin(pitch)*np.cos(yaw)*np.cos(q_1 + roll) + np.sin(yaw)*np.sin(q_1 + roll))*np.sin(q_2) + np.cos(pitch)*np.cos(q_2)*np.cos(yaw))**2)
        Jc[4,3]=-np.sin(q_2)*np.sin(q_1 + roll)*np.cos(pitch)/np.sqrt(1 - (np.sin(pitch)*np.cos(q_2) + np.sin(q_2)*np.cos(pitch)*np.cos(q_1 + roll))**2)
        Jc[4,4]=(-np.sin(pitch)*np.sin(q_2)*np.cos(q_1 + roll) + np.cos(pitch)*np.cos(q_2))/np.sqrt(1 - (np.sin(pitch)*np.cos(q_2) + np.sin(q_2)*np.cos(pitch)*np.cos(q_1 + roll))**2)
        Jc[4,6]=-np.sin(q_2)*np.sin(q_1 + roll)*np.cos(pitch)/np.sqrt(1 - (np.sin(pitch)*np.cos(q_2) + np.sin(q_2)*np.cos(pitch)*np.cos(q_1 + roll))**2)
        Jc[4,7]=(-np.sin(pitch)*np.sin(q_2) + np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))/np.sqrt(1 - (np.sin(pitch)*np.cos(q_2) + np.sin(q_2)*np.cos(pitch)*np.cos(q_1 + roll))**2)
        Jc[5,3]=((np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.sin(q_3) - np.sin(q_1 + roll)*np.cos(pitch)*np.cos(q_3))*(-np.sin(q_3)*np.cos(pitch)*np.cos(q_1 + roll) - np.sin(q_1 + roll)*np.cos(pitch)*np.cos(q_2)*np.cos(q_3))/((-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.sin(q_3) + np.sin(q_1 + roll)*np.cos(pitch)*np.cos(q_3))**2 + (-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.cos(q_3) - np.sin(q_3)*np.sin(q_1 + roll)*np.cos(pitch))**2) + (-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.cos(q_3) - np.sin(q_3)*np.sin(q_1 + roll)*np.cos(pitch))*(-np.sin(q_3)*np.sin(q_1 + roll)*np.cos(pitch)*np.cos(q_2) + np.cos(pitch)*np.cos(q_3)*np.cos(q_1 + roll))/((-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.sin(q_3) + np.sin(q_1 + roll)*np.cos(pitch)*np.cos(q_3))**2 + (-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.cos(q_3) - np.sin(q_3)*np.sin(q_1 + roll)*np.cos(pitch))**2)
        Jc[5,4]=((np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.sin(q_3) - np.sin(q_1 + roll)*np.cos(pitch)*np.cos(q_3))*(-(np.sin(pitch)*np.cos(q_2)*np.cos(q_1 + roll) + np.sin(q_2)*np.cos(pitch))*np.cos(q_3) + np.sin(pitch)*np.sin(q_3)*np.sin(q_1 + roll))/((-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.sin(q_3) + np.sin(q_1 + roll)*np.cos(pitch)*np.cos(q_3))**2 + (-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.cos(q_3) - np.sin(q_3)*np.sin(q_1 + roll)*np.cos(pitch))**2) + (-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.cos(q_3) - np.sin(q_3)*np.sin(q_1 + roll)*np.cos(pitch))*(-(np.sin(pitch)*np.cos(q_2)*np.cos(q_1 + roll) + np.sin(q_2)*np.cos(pitch))*np.sin(q_3) - np.sin(pitch)*np.sin(q_1 + roll)*np.cos(q_3))/((-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.sin(q_3) + np.sin(q_1 + roll)*np.cos(pitch)*np.cos(q_3))**2 + (-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.cos(q_3) - np.sin(q_3)*np.sin(q_1 + roll)*np.cos(pitch))**2)
        Jc[5,6]=((np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.sin(q_3) - np.sin(q_1 + roll)*np.cos(pitch)*np.cos(q_3))*(-np.sin(q_3)*np.cos(pitch)*np.cos(q_1 + roll) - np.sin(q_1 + roll)*np.cos(pitch)*np.cos(q_2)*np.cos(q_3))/((-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.sin(q_3) + np.sin(q_1 + roll)*np.cos(pitch)*np.cos(q_3))**2 + (-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.cos(q_3) - np.sin(q_3)*np.sin(q_1 + roll)*np.cos(pitch))**2) + (-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.cos(q_3) - np.sin(q_3)*np.sin(q_1 + roll)*np.cos(pitch))*(-np.sin(q_3)*np.sin(q_1 + roll)*np.cos(pitch)*np.cos(q_2) + np.cos(pitch)*np.cos(q_3)*np.cos(q_1 + roll))/((-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.sin(q_3) + np.sin(q_1 + roll)*np.cos(pitch)*np.cos(q_3))**2 + (-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.cos(q_3) - np.sin(q_3)*np.sin(q_1 + roll)*np.cos(pitch))**2)
        Jc[5,7]=-((np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.sin(q_3) - np.sin(q_1 + roll)*np.cos(pitch)*np.cos(q_3))*(np.sin(pitch)*np.cos(q_2) + np.sin(q_2)*np.cos(pitch)*np.cos(q_1 + roll))*np.cos(q_3)/((-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.sin(q_3) + np.sin(q_1 + roll)*np.cos(pitch)*np.cos(q_3))**2 + (-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.cos(q_3) - np.sin(q_3)*np.sin(q_1 + roll)*np.cos(pitch))**2) - (-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.cos(q_3) - np.sin(q_3)*np.sin(q_1 + roll)*np.cos(pitch))*(np.sin(pitch)*np.cos(q_2) + np.sin(q_2)*np.cos(pitch)*np.cos(q_1 + roll))*np.sin(q_3)/((-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.sin(q_3) + np.sin(q_1 + roll)*np.cos(pitch)*np.cos(q_3))**2 + (-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.cos(q_3) - np.sin(q_3)*np.sin(q_1 + roll)*np.cos(pitch))**2)
        Jc[5,8]=((np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.sin(q_3) - np.sin(q_1 + roll)*np.cos(pitch)*np.cos(q_3))**2/((-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.sin(q_3) + np.sin(q_1 + roll)*np.cos(pitch)*np.cos(q_3))**2 + (-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.cos(q_3) - np.sin(q_3)*np.sin(q_1 + roll)*np.cos(pitch))**2) + (-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.cos(q_3) - np.sin(q_3)*np.sin(q_1 + roll)*np.cos(pitch))**2/((-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.sin(q_3) + np.sin(q_1 + roll)*np.cos(pitch)*np.cos(q_3))**2 + (-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.cos(q_3) - np.sin(q_3)*np.sin(q_1 + roll)*np.cos(pitch))**2)
        return Jc

    def evaluate_JG(self, roll, pitch, yaw, q_1, q_2, q_3):
        JG = np.zeros((6,9))
        JG[0,0]=1
        JG[0,3]=L_1*np.sin(pitch)*np.sin(q_1 + roll)*np.cos(yaw) - L_1*np.sin(yaw)*np.cos(q_1 + roll) + L_2*np.sin(pitch)*np.sin(q_1 + roll)*np.cos(q_2)*np.cos(yaw) - L_2*np.sin(yaw)*np.cos(q_2)*np.cos(q_1 + roll) + L_3*np.sin(pitch)*np.sin(q_3)*np.cos(yaw)*np.cos(q_1 + roll) + L_3*np.sin(pitch)*np.sin(q_1 + roll)*np.cos(q_2)*np.cos(q_3)*np.cos(yaw) + L_3*np.sin(q_3)*np.sin(yaw)*np.sin(q_1 + roll) - L_3*np.sin(yaw)*np.cos(q_2)*np.cos(q_3)*np.cos(q_1 + roll)
        JG[0,4]=-L_1*np.cos(pitch)*np.cos(yaw)*np.cos(q_1 + roll) + L_2*np.sin(pitch)*np.sin(q_2)*np.cos(yaw) - L_2*np.cos(pitch)*np.cos(q_2)*np.cos(yaw)*np.cos(q_1 + roll) + L_3*np.sin(pitch)*np.sin(q_2)*np.cos(q_3)*np.cos(yaw) + L_3*np.sin(q_3)*np.sin(q_1 + roll)*np.cos(pitch)*np.cos(yaw) - L_3*np.cos(pitch)*np.cos(q_2)*np.cos(q_3)*np.cos(yaw)*np.cos(q_1 + roll)
        JG[0,5]=L_1*np.sin(pitch)*np.sin(yaw)*np.cos(q_1 + roll) - L_1*np.sin(q_1 + roll)*np.cos(yaw) + L_2*np.sin(pitch)*np.sin(yaw)*np.cos(q_2)*np.cos(q_1 + roll) + L_2*np.sin(q_2)*np.sin(yaw)*np.cos(pitch) - L_2*np.sin(q_1 + roll)*np.cos(q_2)*np.cos(yaw) - L_3*np.sin(pitch)*np.sin(q_3)*np.sin(yaw)*np.sin(q_1 + roll) + L_3*np.sin(pitch)*np.sin(yaw)*np.cos(q_2)*np.cos(q_3)*np.cos(q_1 + roll) + L_3*np.sin(q_2)*np.sin(yaw)*np.cos(pitch)*np.cos(q_3) - L_3*np.sin(q_3)*np.cos(yaw)*np.cos(q_1 + roll) - L_3*np.sin(q_1 + roll)*np.cos(q_2)*np.cos(q_3)*np.cos(yaw)
        JG[0,6]=L_1*np.sin(pitch)*np.sin(q_1 + roll)*np.cos(yaw) - L_1*np.sin(yaw)*np.cos(q_1 + roll) + L_2*np.sin(pitch)*np.sin(q_1 + roll)*np.cos(q_2)*np.cos(yaw) - L_2*np.sin(yaw)*np.cos(q_2)*np.cos(q_1 + roll) + L_3*np.sin(pitch)*np.sin(q_3)*np.cos(yaw)*np.cos(q_1 + roll) + L_3*np.sin(pitch)*np.sin(q_1 + roll)*np.cos(q_2)*np.cos(q_3)*np.cos(yaw) + L_3*np.sin(q_3)*np.sin(yaw)*np.sin(q_1 + roll) - L_3*np.sin(yaw)*np.cos(q_2)*np.cos(q_3)*np.cos(q_1 + roll)
        JG[0,7]=L_2*np.sin(pitch)*np.sin(q_2)*np.cos(yaw)*np.cos(q_1 + roll) + L_2*np.sin(q_2)*np.sin(yaw)*np.sin(q_1 + roll) - L_2*np.cos(pitch)*np.cos(q_2)*np.cos(yaw) + L_3*np.sin(pitch)*np.sin(q_2)*np.cos(q_3)*np.cos(yaw)*np.cos(q_1 + roll) + L_3*np.sin(q_2)*np.sin(yaw)*np.sin(q_1 + roll)*np.cos(q_3) - L_3*np.cos(pitch)*np.cos(q_2)*np.cos(q_3)*np.cos(yaw)
        JG[0,8]=L_3*np.sin(pitch)*np.sin(q_3)*np.cos(q_2)*np.cos(yaw)*np.cos(q_1 + roll) + L_3*np.sin(pitch)*np.sin(q_1 + roll)*np.cos(q_3)*np.cos(yaw) + L_3*np.sin(q_2)*np.sin(q_3)*np.cos(pitch)*np.cos(yaw) + L_3*np.sin(q_3)*np.sin(yaw)*np.sin(q_1 + roll)*np.cos(q_2) - L_3*np.sin(yaw)*np.cos(q_3)*np.cos(q_1 + roll)
        JG[1,1]=1
        JG[1,3]=L_1*np.sin(pitch)*np.sin(yaw)*np.sin(q_1 + roll) + L_1*np.cos(yaw)*np.cos(q_1 + roll) + L_2*np.sin(pitch)*np.sin(yaw)*np.sin(q_1 + roll)*np.cos(q_2) + L_2*np.cos(q_2)*np.cos(yaw)*np.cos(q_1 + roll) + L_3*np.sin(pitch)*np.sin(q_3)*np.sin(yaw)*np.cos(q_1 + roll) + L_3*np.sin(pitch)*np.sin(yaw)*np.sin(q_1 + roll)*np.cos(q_2)*np.cos(q_3) - L_3*np.sin(q_3)*np.sin(q_1 + roll)*np.cos(yaw) + L_3*np.cos(q_2)*np.cos(q_3)*np.cos(yaw)*np.cos(q_1 + roll)
        JG[1,4]=-L_1*np.sin(yaw)*np.cos(pitch)*np.cos(q_1 + roll) + L_2*np.sin(pitch)*np.sin(q_2)*np.sin(yaw) - L_2*np.sin(yaw)*np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll) + L_3*np.sin(pitch)*np.sin(q_2)*np.sin(yaw)*np.cos(q_3) + L_3*np.sin(q_3)*np.sin(yaw)*np.sin(q_1 + roll)*np.cos(pitch) - L_3*np.sin(yaw)*np.cos(pitch)*np.cos(q_2)*np.cos(q_3)*np.cos(q_1 + roll)
        JG[1,5]=-L_1*np.sin(pitch)*np.cos(yaw)*np.cos(q_1 + roll) - L_1*np.sin(yaw)*np.sin(q_1 + roll) - L_2*np.sin(pitch)*np.cos(q_2)*np.cos(yaw)*np.cos(q_1 + roll) - L_2*np.sin(q_2)*np.cos(pitch)*np.cos(yaw) - L_2*np.sin(yaw)*np.sin(q_1 + roll)*np.cos(q_2) + L_3*np.sin(pitch)*np.sin(q_3)*np.sin(q_1 + roll)*np.cos(yaw) - L_3*np.sin(pitch)*np.cos(q_2)*np.cos(q_3)*np.cos(yaw)*np.cos(q_1 + roll) - L_3*np.sin(q_2)*np.cos(pitch)*np.cos(q_3)*np.cos(yaw) - L_3*np.sin(q_3)*np.sin(yaw)*np.cos(q_1 + roll) - L_3*np.sin(yaw)*np.sin(q_1 + roll)*np.cos(q_2)*np.cos(q_3)
        JG[1,6]=L_1*np.sin(pitch)*np.sin(yaw)*np.sin(q_1 + roll) + L_1*np.cos(yaw)*np.cos(q_1 + roll) + L_2*np.sin(pitch)*np.sin(yaw)*np.sin(q_1 + roll)*np.cos(q_2) + L_2*np.cos(q_2)*np.cos(yaw)*np.cos(q_1 + roll) + L_3*np.sin(pitch)*np.sin(q_3)*np.sin(yaw)*np.cos(q_1 + roll) + L_3*np.sin(pitch)*np.sin(yaw)*np.sin(q_1 + roll)*np.cos(q_2)*np.cos(q_3) - L_3*np.sin(q_3)*np.sin(q_1 + roll)*np.cos(yaw) + L_3*np.cos(q_2)*np.cos(q_3)*np.cos(yaw)*np.cos(q_1 + roll)
        JG[1,7]=L_2*np.sin(pitch)*np.sin(q_2)*np.sin(yaw)*np.cos(q_1 + roll) - L_2*np.sin(q_2)*np.sin(q_1 + roll)*np.cos(yaw) - L_2*np.sin(yaw)*np.cos(pitch)*np.cos(q_2) + L_3*np.sin(pitch)*np.sin(q_2)*np.sin(yaw)*np.cos(q_3)*np.cos(q_1 + roll) - L_3*np.sin(q_2)*np.sin(q_1 + roll)*np.cos(q_3)*np.cos(yaw) - L_3*np.sin(yaw)*np.cos(pitch)*np.cos(q_2)*np.cos(q_3)
        JG[1,8]=L_3*np.sin(pitch)*np.sin(q_3)*np.sin(yaw)*np.cos(q_2)*np.cos(q_1 + roll) + L_3*np.sin(pitch)*np.sin(yaw)*np.sin(q_1 + roll)*np.cos(q_3) + L_3*np.sin(q_2)*np.sin(q_3)*np.sin(yaw)*np.cos(pitch) - L_3*np.sin(q_3)*np.sin(q_1 + roll)*np.cos(q_2)*np.cos(yaw) + L_3*np.cos(q_3)*np.cos(yaw)*np.cos(q_1 + roll)
        JG[2,2]=1
        JG[2,3]=L_1*np.sin(q_1 + roll)*np.cos(pitch) + L_2*np.sin(q_1 + roll)*np.cos(pitch)*np.cos(q_2) + L_3*np.sin(q_3)*np.cos(pitch)*np.cos(q_1 + roll) + L_3*np.sin(q_1 + roll)*np.cos(pitch)*np.cos(q_2)*np.cos(q_3)
        JG[2,4]=L_1*np.sin(pitch)*np.cos(q_1 + roll) + L_2*np.sin(pitch)*np.cos(q_2)*np.cos(q_1 + roll) + L_2*np.sin(q_2)*np.cos(pitch) - L_3*np.sin(pitch)*np.sin(q_3)*np.sin(q_1 + roll) + L_3*np.sin(pitch)*np.cos(q_2)*np.cos(q_3)*np.cos(q_1 + roll) + L_3*np.sin(q_2)*np.cos(pitch)*np.cos(q_3)
        JG[2,6]=L_1*np.sin(q_1 + roll)*np.cos(pitch) + L_2*np.sin(q_1 + roll)*np.cos(pitch)*np.cos(q_2) + L_3*np.sin(q_3)*np.cos(pitch)*np.cos(q_1 + roll) + L_3*np.sin(q_1 + roll)*np.cos(pitch)*np.cos(q_2)*np.cos(q_3)
        JG[2,7]=L_2*np.sin(pitch)*np.cos(q_2) + L_2*np.sin(q_2)*np.cos(pitch)*np.cos(q_1 + roll) + L_3*np.sin(pitch)*np.cos(q_2)*np.cos(q_3) + L_3*np.sin(q_2)*np.cos(pitch)*np.cos(q_3)*np.cos(q_1 + roll)
        JG[2,8]=-L_3*np.sin(pitch)*np.sin(q_2)*np.sin(q_3) + L_3*np.sin(q_3)*np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll) + L_3*np.sin(q_1 + roll)*np.cos(pitch)*np.cos(q_3)
        JG[3,3]=(-(-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.sin(q_3) + np.sin(q_1 + roll)*np.cos(pitch)*np.cos(q_3))*(-np.sin(q_3)*np.cos(pitch)*np.cos(q_1 + roll) - np.sin(q_1 + roll)*np.cos(pitch)*np.cos(q_2)*np.cos(q_3))/(((-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.sin(q_3) + np.sin(q_1 + roll)*np.cos(pitch)*np.cos(q_3))**2/np.cos(pitch)**2 + (-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.cos(q_3) - np.sin(q_3)*np.sin(q_1 + roll)*np.cos(pitch))**2/np.cos(pitch)**2)*np.cos(pitch)**2) + (-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.cos(q_3) - np.sin(q_3)*np.sin(q_1 + roll)*np.cos(pitch))*(-np.sin(q_3)*np.sin(q_1 + roll)*np.cos(pitch)*np.cos(q_2) + np.cos(pitch)*np.cos(q_3)*np.cos(q_1 + roll))/(((-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.sin(q_3) + np.sin(q_1 + roll)*np.cos(pitch)*np.cos(q_3))**2/np.cos(pitch)**2 + (-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.cos(q_3) - np.sin(q_3)*np.sin(q_1 + roll)*np.cos(pitch))**2/np.cos(pitch)**2)*np.cos(pitch)**2))*np.cos(pitch)*np.cos(yaw) + np.sin(q_2)*np.sin(yaw)*np.sin(q_1 + roll)*np.cos(pitch)/np.sqrt(1 - (np.sin(pitch)*np.cos(q_2) + np.sin(q_2)*np.cos(pitch)*np.cos(q_1 + roll))**2)
        JG[3,4]=(-(-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.sin(q_3) + np.sin(q_1 + roll)*np.cos(pitch)*np.cos(q_3))*((-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.cos(q_3) - np.sin(q_3)*np.sin(q_1 + roll)*np.cos(pitch))*np.sin(pitch)/np.cos(pitch)**2 + (-(np.sin(pitch)*np.cos(q_2)*np.cos(q_1 + roll) + np.sin(q_2)*np.cos(pitch))*np.cos(q_3) + np.sin(pitch)*np.sin(q_3)*np.sin(q_1 + roll))/np.cos(pitch))/(((-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.sin(q_3) + np.sin(q_1 + roll)*np.cos(pitch)*np.cos(q_3))**2/np.cos(pitch)**2 + (-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.cos(q_3) - np.sin(q_3)*np.sin(q_1 + roll)*np.cos(pitch))**2/np.cos(pitch)**2)*np.cos(pitch)) + (-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.cos(q_3) - np.sin(q_3)*np.sin(q_1 + roll)*np.cos(pitch))*((-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.sin(q_3) + np.sin(q_1 + roll)*np.cos(pitch)*np.cos(q_3))*np.sin(pitch)/np.cos(pitch)**2 + (-(np.sin(pitch)*np.cos(q_2)*np.cos(q_1 + roll) + np.sin(q_2)*np.cos(pitch))*np.sin(q_3) - np.sin(pitch)*np.sin(q_1 + roll)*np.cos(q_3))/np.cos(pitch))/(((-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.sin(q_3) + np.sin(q_1 + roll)*np.cos(pitch)*np.cos(q_3))**2/np.cos(pitch)**2 + (-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.cos(q_3) - np.sin(q_3)*np.sin(q_1 + roll)*np.cos(pitch))**2/np.cos(pitch)**2)*np.cos(pitch)))*np.cos(pitch)*np.cos(yaw) - (-np.sin(pitch)*np.sin(q_2)*np.cos(q_1 + roll) + np.cos(pitch)*np.cos(q_2))*np.sin(yaw)/np.sqrt(1 - (np.sin(pitch)*np.cos(q_2) + np.sin(q_2)*np.cos(pitch)*np.cos(q_1 + roll))**2)
        JG[3,6]=(-(-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.sin(q_3) + np.sin(q_1 + roll)*np.cos(pitch)*np.cos(q_3))*(-np.sin(q_3)*np.cos(pitch)*np.cos(q_1 + roll) - np.sin(q_1 + roll)*np.cos(pitch)*np.cos(q_2)*np.cos(q_3))/(((-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.sin(q_3) + np.sin(q_1 + roll)*np.cos(pitch)*np.cos(q_3))**2/np.cos(pitch)**2 + (-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.cos(q_3) - np.sin(q_3)*np.sin(q_1 + roll)*np.cos(pitch))**2/np.cos(pitch)**2)*np.cos(pitch)**2) + (-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.cos(q_3) - np.sin(q_3)*np.sin(q_1 + roll)*np.cos(pitch))*(-np.sin(q_3)*np.sin(q_1 + roll)*np.cos(pitch)*np.cos(q_2) + np.cos(pitch)*np.cos(q_3)*np.cos(q_1 + roll))/(((-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.sin(q_3) + np.sin(q_1 + roll)*np.cos(pitch)*np.cos(q_3))**2/np.cos(pitch)**2 + (-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.cos(q_3) - np.sin(q_3)*np.sin(q_1 + roll)*np.cos(pitch))**2/np.cos(pitch)**2)*np.cos(pitch)**2))*np.cos(pitch)*np.cos(yaw) + np.sin(q_2)*np.sin(yaw)*np.sin(q_1 + roll)*np.cos(pitch)/np.sqrt(1 - (np.sin(pitch)*np.cos(q_2) + np.sin(q_2)*np.cos(pitch)*np.cos(q_1 + roll))**2)
        JG[3,7]=((-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.sin(q_3) + np.sin(q_1 + roll)*np.cos(pitch)*np.cos(q_3))*(np.sin(pitch)*np.cos(q_2) + np.sin(q_2)*np.cos(pitch)*np.cos(q_1 + roll))*np.cos(q_3)/(((-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.sin(q_3) + np.sin(q_1 + roll)*np.cos(pitch)*np.cos(q_3))**2/np.cos(pitch)**2 + (-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.cos(q_3) - np.sin(q_3)*np.sin(q_1 + roll)*np.cos(pitch))**2/np.cos(pitch)**2)*np.cos(pitch)**2) - (-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.cos(q_3) - np.sin(q_3)*np.sin(q_1 + roll)*np.cos(pitch))*(np.sin(pitch)*np.cos(q_2) + np.sin(q_2)*np.cos(pitch)*np.cos(q_1 + roll))*np.sin(q_3)/(((-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.sin(q_3) + np.sin(q_1 + roll)*np.cos(pitch)*np.cos(q_3))**2/np.cos(pitch)**2 + (-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.cos(q_3) - np.sin(q_3)*np.sin(q_1 + roll)*np.cos(pitch))**2/np.cos(pitch)**2)*np.cos(pitch)**2))*np.cos(pitch)*np.cos(yaw) - (-np.sin(pitch)*np.sin(q_2) + np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.sin(yaw)/np.sqrt(1 - (np.sin(pitch)*np.cos(q_2) + np.sin(q_2)*np.cos(pitch)*np.cos(q_1 + roll))**2)
        JG[3,8]=(-(-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.sin(q_3) + np.sin(q_1 + roll)*np.cos(pitch)*np.cos(q_3))*((np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.sin(q_3) - np.sin(q_1 + roll)*np.cos(pitch)*np.cos(q_3))/(((-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.sin(q_3) + np.sin(q_1 + roll)*np.cos(pitch)*np.cos(q_3))**2/np.cos(pitch)**2 + (-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.cos(q_3) - np.sin(q_3)*np.sin(q_1 + roll)*np.cos(pitch))**2/np.cos(pitch)**2)*np.cos(pitch)**2) + (-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.cos(q_3) - np.sin(q_3)*np.sin(q_1 + roll)*np.cos(pitch))**2/(((-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.sin(q_3) + np.sin(q_1 + roll)*np.cos(pitch)*np.cos(q_3))**2/np.cos(pitch)**2 + (-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.cos(q_3) - np.sin(q_3)*np.sin(q_1 + roll)*np.cos(pitch))**2/np.cos(pitch)**2)*np.cos(pitch)**2))*np.cos(pitch)*np.cos(yaw)
        JG[4,3]=(-(-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.sin(q_3) + np.sin(q_1 + roll)*np.cos(pitch)*np.cos(q_3))*(-np.sin(q_3)*np.cos(pitch)*np.cos(q_1 + roll) - np.sin(q_1 + roll)*np.cos(pitch)*np.cos(q_2)*np.cos(q_3))/(((-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.sin(q_3) + np.sin(q_1 + roll)*np.cos(pitch)*np.cos(q_3))**2/np.cos(pitch)**2 + (-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.cos(q_3) - np.sin(q_3)*np.sin(q_1 + roll)*np.cos(pitch))**2/np.cos(pitch)**2)*np.cos(pitch)**2) + (-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.cos(q_3) - np.sin(q_3)*np.sin(q_1 + roll)*np.cos(pitch))*(-np.sin(q_3)*np.sin(q_1 + roll)*np.cos(pitch)*np.cos(q_2) + np.cos(pitch)*np.cos(q_3)*np.cos(q_1 + roll))/(((-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.sin(q_3) + np.sin(q_1 + roll)*np.cos(pitch)*np.cos(q_3))**2/np.cos(pitch)**2 + (-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.cos(q_3) - np.sin(q_3)*np.sin(q_1 + roll)*np.cos(pitch))**2/np.cos(pitch)**2)*np.cos(pitch)**2))*np.sin(yaw)*np.cos(pitch) - np.sin(q_2)*np.sin(q_1 + roll)*np.cos(pitch)*np.cos(yaw)/np.sqrt(1 - (np.sin(pitch)*np.cos(q_2) + np.sin(q_2)*np.cos(pitch)*np.cos(q_1 + roll))**2)
        JG[4,4]=(-(-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.sin(q_3) + np.sin(q_1 + roll)*np.cos(pitch)*np.cos(q_3))*((-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.cos(q_3) - np.sin(q_3)*np.sin(q_1 + roll)*np.cos(pitch))*np.sin(pitch)/np.cos(pitch)**2 + (-(np.sin(pitch)*np.cos(q_2)*np.cos(q_1 + roll) + np.sin(q_2)*np.cos(pitch))*np.cos(q_3) + np.sin(pitch)*np.sin(q_3)*np.sin(q_1 + roll))/np.cos(pitch))/(((-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.sin(q_3) + np.sin(q_1 + roll)*np.cos(pitch)*np.cos(q_3))**2/np.cos(pitch)**2 + (-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.cos(q_3) - np.sin(q_3)*np.sin(q_1 + roll)*np.cos(pitch))**2/np.cos(pitch)**2)*np.cos(pitch)) + (-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.cos(q_3) - np.sin(q_3)*np.sin(q_1 + roll)*np.cos(pitch))*((-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.sin(q_3) + np.sin(q_1 + roll)*np.cos(pitch)*np.cos(q_3))*np.sin(pitch)/np.cos(pitch)**2 + (-(np.sin(pitch)*np.cos(q_2)*np.cos(q_1 + roll) + np.sin(q_2)*np.cos(pitch))*np.sin(q_3) - np.sin(pitch)*np.sin(q_1 + roll)*np.cos(q_3))/np.cos(pitch))/(((-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.sin(q_3) + np.sin(q_1 + roll)*np.cos(pitch)*np.cos(q_3))**2/np.cos(pitch)**2 + (-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.cos(q_3) - np.sin(q_3)*np.sin(q_1 + roll)*np.cos(pitch))**2/np.cos(pitch)**2)*np.cos(pitch)))*np.sin(yaw)*np.cos(pitch) + (-np.sin(pitch)*np.sin(q_2)*np.cos(q_1 + roll) + np.cos(pitch)*np.cos(q_2))*np.cos(yaw)/np.sqrt(1 - (np.sin(pitch)*np.cos(q_2) + np.sin(q_2)*np.cos(pitch)*np.cos(q_1 + roll))**2)
        JG[4,6]=(-(-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.sin(q_3) + np.sin(q_1 + roll)*np.cos(pitch)*np.cos(q_3))*(-np.sin(q_3)*np.cos(pitch)*np.cos(q_1 + roll) - np.sin(q_1 + roll)*np.cos(pitch)*np.cos(q_2)*np.cos(q_3))/(((-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.sin(q_3) + np.sin(q_1 + roll)*np.cos(pitch)*np.cos(q_3))**2/np.cos(pitch)**2 + (-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.cos(q_3) - np.sin(q_3)*np.sin(q_1 + roll)*np.cos(pitch))**2/np.cos(pitch)**2)*np.cos(pitch)**2) + (-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.cos(q_3) - np.sin(q_3)*np.sin(q_1 + roll)*np.cos(pitch))*(-np.sin(q_3)*np.sin(q_1 + roll)*np.cos(pitch)*np.cos(q_2) + np.cos(pitch)*np.cos(q_3)*np.cos(q_1 + roll))/(((-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.sin(q_3) + np.sin(q_1 + roll)*np.cos(pitch)*np.cos(q_3))**2/np.cos(pitch)**2 + (-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.cos(q_3) - np.sin(q_3)*np.sin(q_1 + roll)*np.cos(pitch))**2/np.cos(pitch)**2)*np.cos(pitch)**2))*np.sin(yaw)*np.cos(pitch) - np.sin(q_2)*np.sin(q_1 + roll)*np.cos(pitch)*np.cos(yaw)/np.sqrt(1 - (np.sin(pitch)*np.cos(q_2) + np.sin(q_2)*np.cos(pitch)*np.cos(q_1 + roll))**2)
        JG[4,7]=((-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.sin(q_3) + np.sin(q_1 + roll)*np.cos(pitch)*np.cos(q_3))*(np.sin(pitch)*np.cos(q_2) + np.sin(q_2)*np.cos(pitch)*np.cos(q_1 + roll))*np.cos(q_3)/(((-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.sin(q_3) + np.sin(q_1 + roll)*np.cos(pitch)*np.cos(q_3))**2/np.cos(pitch)**2 + (-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.cos(q_3) - np.sin(q_3)*np.sin(q_1 + roll)*np.cos(pitch))**2/np.cos(pitch)**2)*np.cos(pitch)**2) - (-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.cos(q_3) - np.sin(q_3)*np.sin(q_1 + roll)*np.cos(pitch))*(np.sin(pitch)*np.cos(q_2) + np.sin(q_2)*np.cos(pitch)*np.cos(q_1 + roll))*np.sin(q_3)/(((-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.sin(q_3) + np.sin(q_1 + roll)*np.cos(pitch)*np.cos(q_3))**2/np.cos(pitch)**2 + (-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.cos(q_3) - np.sin(q_3)*np.sin(q_1 + roll)*np.cos(pitch))**2/np.cos(pitch)**2)*np.cos(pitch)**2))*np.sin(yaw)*np.cos(pitch) + (-np.sin(pitch)*np.sin(q_2) + np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.cos(yaw)/np.sqrt(1 - (np.sin(pitch)*np.cos(q_2) + np.sin(q_2)*np.cos(pitch)*np.cos(q_1 + roll))**2)
        JG[4,8]=(-(-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.sin(q_3) + np.sin(q_1 + roll)*np.cos(pitch)*np.cos(q_3))*((np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.sin(q_3) - np.sin(q_1 + roll)*np.cos(pitch)*np.cos(q_3))/(((-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.sin(q_3) + np.sin(q_1 + roll)*np.cos(pitch)*np.cos(q_3))**2/np.cos(pitch)**2 + (-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.cos(q_3) - np.sin(q_3)*np.sin(q_1 + roll)*np.cos(pitch))**2/np.cos(pitch)**2)*np.cos(pitch)**2) + (-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.cos(q_3) - np.sin(q_3)*np.sin(q_1 + roll)*np.cos(pitch))**2/(((-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.sin(q_3) + np.sin(q_1 + roll)*np.cos(pitch)*np.cos(q_3))**2/np.cos(pitch)**2 + (-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.cos(q_3) - np.sin(q_3)*np.sin(q_1 + roll)*np.cos(pitch))**2/np.cos(pitch)**2)*np.cos(pitch)**2))*np.sin(yaw)*np.cos(pitch)
        JG[5,3]=-(-(-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.sin(q_3) + np.sin(q_1 + roll)*np.cos(pitch)*np.cos(q_3))*(-np.sin(q_3)*np.cos(pitch)*np.cos(q_1 + roll) - np.sin(q_1 + roll)*np.cos(pitch)*np.cos(q_2)*np.cos(q_3))/(((-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.sin(q_3) + np.sin(q_1 + roll)*np.cos(pitch)*np.cos(q_3))**2/np.cos(pitch)**2 + (-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.cos(q_3) - np.sin(q_3)*np.sin(q_1 + roll)*np.cos(pitch))**2/np.cos(pitch)**2)*np.cos(pitch)**2) + (-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.cos(q_3) - np.sin(q_3)*np.sin(q_1 + roll)*np.cos(pitch))*(-np.sin(q_3)*np.sin(q_1 + roll)*np.cos(pitch)*np.cos(q_2) + np.cos(pitch)*np.cos(q_3)*np.cos(q_1 + roll))/(((-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.sin(q_3) + np.sin(q_1 + roll)*np.cos(pitch)*np.cos(q_3))**2/np.cos(pitch)**2 + (-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.cos(q_3) - np.sin(q_3)*np.sin(q_1 + roll)*np.cos(pitch))**2/np.cos(pitch)**2)*np.cos(pitch)**2))*np.sin(pitch) + ((-np.sin(pitch)*np.sin(yaw)*np.cos(q_1 + roll) + np.sin(q_1 + roll)*np.cos(yaw))*np.sin(q_2) + np.sin(yaw)*np.cos(pitch)*np.cos(q_2))*(-np.sin(pitch)*np.sin(q_1 + roll)*np.cos(yaw) + np.sin(yaw)*np.cos(q_1 + roll))*np.sin(q_2)/((((-np.sin(pitch)*np.sin(yaw)*np.cos(q_1 + roll) + np.sin(q_1 + roll)*np.cos(yaw))*np.sin(q_2) + np.sin(yaw)*np.cos(pitch)*np.cos(q_2))**2/np.cos(pitch)**2 + (-(np.sin(pitch)*np.cos(yaw)*np.cos(q_1 + roll) + np.sin(yaw)*np.sin(q_1 + roll))*np.sin(q_2) + np.cos(pitch)*np.cos(q_2)*np.cos(yaw))**2/np.cos(pitch)**2)*np.cos(pitch)**2) + (-(np.sin(pitch)*np.cos(yaw)*np.cos(q_1 + roll) + np.sin(yaw)*np.sin(q_1 + roll))*np.sin(q_2) + np.cos(pitch)*np.cos(q_2)*np.cos(yaw))*(np.sin(pitch)*np.sin(yaw)*np.sin(q_1 + roll) + np.cos(yaw)*np.cos(q_1 + roll))*np.sin(q_2)/((((-np.sin(pitch)*np.sin(yaw)*np.cos(q_1 + roll) + np.sin(q_1 + roll)*np.cos(yaw))*np.sin(q_2) + np.sin(yaw)*np.cos(pitch)*np.cos(q_2))**2/np.cos(pitch)**2 + (-(np.sin(pitch)*np.cos(yaw)*np.cos(q_1 + roll) + np.sin(yaw)*np.sin(q_1 + roll))*np.sin(q_2) + np.cos(pitch)*np.cos(q_2)*np.cos(yaw))**2/np.cos(pitch)**2)*np.cos(pitch)**2)
        JG[5,4]=-(-(-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.sin(q_3) + np.sin(q_1 + roll)*np.cos(pitch)*np.cos(q_3))*((-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.cos(q_3) - np.sin(q_3)*np.sin(q_1 + roll)*np.cos(pitch))*np.sin(pitch)/np.cos(pitch)**2 + (-(np.sin(pitch)*np.cos(q_2)*np.cos(q_1 + roll) + np.sin(q_2)*np.cos(pitch))*np.cos(q_3) + np.sin(pitch)*np.sin(q_3)*np.sin(q_1 + roll))/np.cos(pitch))/(((-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.sin(q_3) + np.sin(q_1 + roll)*np.cos(pitch)*np.cos(q_3))**2/np.cos(pitch)**2 + (-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.cos(q_3) - np.sin(q_3)*np.sin(q_1 + roll)*np.cos(pitch))**2/np.cos(pitch)**2)*np.cos(pitch)) + (-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.cos(q_3) - np.sin(q_3)*np.sin(q_1 + roll)*np.cos(pitch))*((-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.sin(q_3) + np.sin(q_1 + roll)*np.cos(pitch)*np.cos(q_3))*np.sin(pitch)/np.cos(pitch)**2 + (-(np.sin(pitch)*np.cos(q_2)*np.cos(q_1 + roll) + np.sin(q_2)*np.cos(pitch))*np.sin(q_3) - np.sin(pitch)*np.sin(q_1 + roll)*np.cos(q_3))/np.cos(pitch))/(((-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.sin(q_3) + np.sin(q_1 + roll)*np.cos(pitch)*np.cos(q_3))**2/np.cos(pitch)**2 + (-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.cos(q_3) - np.sin(q_3)*np.sin(q_1 + roll)*np.cos(pitch))**2/np.cos(pitch)**2)*np.cos(pitch)))*np.sin(pitch) - ((-np.sin(pitch)*np.sin(yaw)*np.cos(q_1 + roll) + np.sin(q_1 + roll)*np.cos(yaw))*np.sin(q_2) + np.sin(yaw)*np.cos(pitch)*np.cos(q_2))*((-(np.sin(pitch)*np.cos(yaw)*np.cos(q_1 + roll) + np.sin(yaw)*np.sin(q_1 + roll))*np.sin(q_2) + np.cos(pitch)*np.cos(q_2)*np.cos(yaw))*np.sin(pitch)/np.cos(pitch)**2 + (-np.sin(pitch)*np.cos(q_2)*np.cos(yaw) - np.sin(q_2)*np.cos(pitch)*np.cos(yaw)*np.cos(q_1 + roll))/np.cos(pitch))/((((-np.sin(pitch)*np.sin(yaw)*np.cos(q_1 + roll) + np.sin(q_1 + roll)*np.cos(yaw))*np.sin(q_2) + np.sin(yaw)*np.cos(pitch)*np.cos(q_2))**2/np.cos(pitch)**2 + (-(np.sin(pitch)*np.cos(yaw)*np.cos(q_1 + roll) + np.sin(yaw)*np.sin(q_1 + roll))*np.sin(q_2) + np.cos(pitch)*np.cos(q_2)*np.cos(yaw))**2/np.cos(pitch)**2)*np.cos(pitch)) + (-(np.sin(pitch)*np.cos(yaw)*np.cos(q_1 + roll) + np.sin(yaw)*np.sin(q_1 + roll))*np.sin(q_2) + np.cos(pitch)*np.cos(q_2)*np.cos(yaw))*(((-np.sin(pitch)*np.sin(yaw)*np.cos(q_1 + roll) + np.sin(q_1 + roll)*np.cos(yaw))*np.sin(q_2) + np.sin(yaw)*np.cos(pitch)*np.cos(q_2))*np.sin(pitch)/np.cos(pitch)**2 + (-np.sin(pitch)*np.sin(yaw)*np.cos(q_2) - np.sin(q_2)*np.sin(yaw)*np.cos(pitch)*np.cos(q_1 + roll))/np.cos(pitch))/((((-np.sin(pitch)*np.sin(yaw)*np.cos(q_1 + roll) + np.sin(q_1 + roll)*np.cos(yaw))*np.sin(q_2) + np.sin(yaw)*np.cos(pitch)*np.cos(q_2))**2/np.cos(pitch)**2 + (-(np.sin(pitch)*np.cos(yaw)*np.cos(q_1 + roll) + np.sin(yaw)*np.sin(q_1 + roll))*np.sin(q_2) + np.cos(pitch)*np.cos(q_2)*np.cos(yaw))**2/np.cos(pitch)**2)*np.cos(pitch))
        JG[5,5]=-(-(-np.sin(pitch)*np.sin(yaw)*np.cos(q_1 + roll) + np.sin(q_1 + roll)*np.cos(yaw))*np.sin(q_2) - np.sin(yaw)*np.cos(pitch)*np.cos(q_2))*((-np.sin(pitch)*np.sin(yaw)*np.cos(q_1 + roll) + np.sin(q_1 + roll)*np.cos(yaw))*np.sin(q_2) + np.sin(yaw)*np.cos(pitch)*np.cos(q_2))/((((-np.sin(pitch)*np.sin(yaw)*np.cos(q_1 + roll) + np.sin(q_1 + roll)*np.cos(yaw))*np.sin(q_2) + np.sin(yaw)*np.cos(pitch)*np.cos(q_2))**2/np.cos(pitch)**2 + (-(np.sin(pitch)*np.cos(yaw)*np.cos(q_1 + roll) + np.sin(yaw)*np.sin(q_1 + roll))*np.sin(q_2) + np.cos(pitch)*np.cos(q_2)*np.cos(yaw))**2/np.cos(pitch)**2)*np.cos(pitch)**2) + ((-np.sin(pitch)*np.cos(yaw)*np.cos(q_1 + roll) - np.sin(yaw)*np.sin(q_1 + roll))*np.sin(q_2) + np.cos(pitch)*np.cos(q_2)*np.cos(yaw))*(-(np.sin(pitch)*np.cos(yaw)*np.cos(q_1 + roll) + np.sin(yaw)*np.sin(q_1 + roll))*np.sin(q_2) + np.cos(pitch)*np.cos(q_2)*np.cos(yaw))/((((-np.sin(pitch)*np.sin(yaw)*np.cos(q_1 + roll) + np.sin(q_1 + roll)*np.cos(yaw))*np.sin(q_2) + np.sin(yaw)*np.cos(pitch)*np.cos(q_2))**2/np.cos(pitch)**2 + (-(np.sin(pitch)*np.cos(yaw)*np.cos(q_1 + roll) + np.sin(yaw)*np.sin(q_1 + roll))*np.sin(q_2) + np.cos(pitch)*np.cos(q_2)*np.cos(yaw))**2/np.cos(pitch)**2)*np.cos(pitch)**2)
        JG[5,6]=-(-(-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.sin(q_3) + np.sin(q_1 + roll)*np.cos(pitch)*np.cos(q_3))*(-np.sin(q_3)*np.cos(pitch)*np.cos(q_1 + roll) - np.sin(q_1 + roll)*np.cos(pitch)*np.cos(q_2)*np.cos(q_3))/(((-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.sin(q_3) + np.sin(q_1 + roll)*np.cos(pitch)*np.cos(q_3))**2/np.cos(pitch)**2 + (-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.cos(q_3) - np.sin(q_3)*np.sin(q_1 + roll)*np.cos(pitch))**2/np.cos(pitch)**2)*np.cos(pitch)**2) + (-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.cos(q_3) - np.sin(q_3)*np.sin(q_1 + roll)*np.cos(pitch))*(-np.sin(q_3)*np.sin(q_1 + roll)*np.cos(pitch)*np.cos(q_2) + np.cos(pitch)*np.cos(q_3)*np.cos(q_1 + roll))/(((-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.sin(q_3) + np.sin(q_1 + roll)*np.cos(pitch)*np.cos(q_3))**2/np.cos(pitch)**2 + (-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.cos(q_3) - np.sin(q_3)*np.sin(q_1 + roll)*np.cos(pitch))**2/np.cos(pitch)**2)*np.cos(pitch)**2))*np.sin(pitch) + ((-np.sin(pitch)*np.sin(yaw)*np.cos(q_1 + roll) + np.sin(q_1 + roll)*np.cos(yaw))*np.sin(q_2) + np.sin(yaw)*np.cos(pitch)*np.cos(q_2))*(-np.sin(pitch)*np.sin(q_1 + roll)*np.cos(yaw) + np.sin(yaw)*np.cos(q_1 + roll))*np.sin(q_2)/((((-np.sin(pitch)*np.sin(yaw)*np.cos(q_1 + roll) + np.sin(q_1 + roll)*np.cos(yaw))*np.sin(q_2) + np.sin(yaw)*np.cos(pitch)*np.cos(q_2))**2/np.cos(pitch)**2 + (-(np.sin(pitch)*np.cos(yaw)*np.cos(q_1 + roll) + np.sin(yaw)*np.sin(q_1 + roll))*np.sin(q_2) + np.cos(pitch)*np.cos(q_2)*np.cos(yaw))**2/np.cos(pitch)**2)*np.cos(pitch)**2) + (-(np.sin(pitch)*np.cos(yaw)*np.cos(q_1 + roll) + np.sin(yaw)*np.sin(q_1 + roll))*np.sin(q_2) + np.cos(pitch)*np.cos(q_2)*np.cos(yaw))*(np.sin(pitch)*np.sin(yaw)*np.sin(q_1 + roll) + np.cos(yaw)*np.cos(q_1 + roll))*np.sin(q_2)/((((-np.sin(pitch)*np.sin(yaw)*np.cos(q_1 + roll) + np.sin(q_1 + roll)*np.cos(yaw))*np.sin(q_2) + np.sin(yaw)*np.cos(pitch)*np.cos(q_2))**2/np.cos(pitch)**2 + (-(np.sin(pitch)*np.cos(yaw)*np.cos(q_1 + roll) + np.sin(yaw)*np.sin(q_1 + roll))*np.sin(q_2) + np.cos(pitch)*np.cos(q_2)*np.cos(yaw))**2/np.cos(pitch)**2)*np.cos(pitch)**2)
        JG[5,7]=-((-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.sin(q_3) + np.sin(q_1 + roll)*np.cos(pitch)*np.cos(q_3))*(np.sin(pitch)*np.cos(q_2) + np.sin(q_2)*np.cos(pitch)*np.cos(q_1 + roll))*np.cos(q_3)/(((-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.sin(q_3) + np.sin(q_1 + roll)*np.cos(pitch)*np.cos(q_3))**2/np.cos(pitch)**2 + (-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.cos(q_3) - np.sin(q_3)*np.sin(q_1 + roll)*np.cos(pitch))**2/np.cos(pitch)**2)*np.cos(pitch)**2) - (-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.cos(q_3) - np.sin(q_3)*np.sin(q_1 + roll)*np.cos(pitch))*(np.sin(pitch)*np.cos(q_2) + np.sin(q_2)*np.cos(pitch)*np.cos(q_1 + roll))*np.sin(q_3)/(((-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.sin(q_3) + np.sin(q_1 + roll)*np.cos(pitch)*np.cos(q_3))**2/np.cos(pitch)**2 + (-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.cos(q_3) - np.sin(q_3)*np.sin(q_1 + roll)*np.cos(pitch))**2/np.cos(pitch)**2)*np.cos(pitch)**2))*np.sin(pitch) - ((-np.sin(pitch)*np.sin(yaw)*np.cos(q_1 + roll) + np.sin(q_1 + roll)*np.cos(yaw))*np.sin(q_2) + np.sin(yaw)*np.cos(pitch)*np.cos(q_2))*(-(np.sin(pitch)*np.cos(yaw)*np.cos(q_1 + roll) + np.sin(yaw)*np.sin(q_1 + roll))*np.cos(q_2) - np.sin(q_2)*np.cos(pitch)*np.cos(yaw))/((((-np.sin(pitch)*np.sin(yaw)*np.cos(q_1 + roll) + np.sin(q_1 + roll)*np.cos(yaw))*np.sin(q_2) + np.sin(yaw)*np.cos(pitch)*np.cos(q_2))**2/np.cos(pitch)**2 + (-(np.sin(pitch)*np.cos(yaw)*np.cos(q_1 + roll) + np.sin(yaw)*np.sin(q_1 + roll))*np.sin(q_2) + np.cos(pitch)*np.cos(q_2)*np.cos(yaw))**2/np.cos(pitch)**2)*np.cos(pitch)**2) + ((-np.sin(pitch)*np.sin(yaw)*np.cos(q_1 + roll) + np.sin(q_1 + roll)*np.cos(yaw))*np.cos(q_2) - np.sin(q_2)*np.sin(yaw)*np.cos(pitch))*(-(np.sin(pitch)*np.cos(yaw)*np.cos(q_1 + roll) + np.sin(yaw)*np.sin(q_1 + roll))*np.sin(q_2) + np.cos(pitch)*np.cos(q_2)*np.cos(yaw))/((((-np.sin(pitch)*np.sin(yaw)*np.cos(q_1 + roll) + np.sin(q_1 + roll)*np.cos(yaw))*np.sin(q_2) + np.sin(yaw)*np.cos(pitch)*np.cos(q_2))**2/np.cos(pitch)**2 + (-(np.sin(pitch)*np.cos(yaw)*np.cos(q_1 + roll) + np.sin(yaw)*np.sin(q_1 + roll))*np.sin(q_2) + np.cos(pitch)*np.cos(q_2)*np.cos(yaw))**2/np.cos(pitch)**2)*np.cos(pitch)**2)
        JG[5,8]=-(-(-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.sin(q_3) + np.sin(q_1 + roll)*np.cos(pitch)*np.cos(q_3))*((np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.sin(q_3) - np.sin(q_1 + roll)*np.cos(pitch)*np.cos(q_3))/(((-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.sin(q_3) + np.sin(q_1 + roll)*np.cos(pitch)*np.cos(q_3))**2/np.cos(pitch)**2 + (-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.cos(q_3) - np.sin(q_3)*np.sin(q_1 + roll)*np.cos(pitch))**2/np.cos(pitch)**2)*np.cos(pitch)**2) + (-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.cos(q_3) - np.sin(q_3)*np.sin(q_1 + roll)*np.cos(pitch))**2/(((-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.sin(q_3) + np.sin(q_1 + roll)*np.cos(pitch)*np.cos(q_3))**2/np.cos(pitch)**2 + (-(np.sin(pitch)*np.sin(q_2) - np.cos(pitch)*np.cos(q_2)*np.cos(q_1 + roll))*np.cos(q_3) - np.sin(q_3)*np.sin(q_1 + roll)*np.cos(pitch))**2/np.cos(pitch)**2)*np.cos(pitch)**2))*np.sin(pitch)

        return JG


    def T_euler_rate_to_angular_velocity(self, pitch, yaw):
        T = np.array([[np.cos(yaw)*np.cos(pitch), -np.sin(yaw), 0], 
                      [np.sin(yaw)*np.cos(pitch), np.cos(yaw), 0], 
                      [-np.sin(pitch), 0, 1]])
        return T

    def T_euler_rate_to_angular_velocity_inv(self, pitch, yaw):
        T_inv = np.zeros((3,3))
        T_inv[0,0]=np.cos(pitch)/(np.sin(pitch)**2*np.cos(yaw) + np.cos(pitch)**2*np.cos(yaw))
        T_inv[0,1]=np.sin(pitch)/(np.sin(pitch)**2*np.cos(yaw) + np.cos(pitch)**2*np.cos(yaw))
        T_inv[0,2]=0
        T_inv[1,0]=-np.sin(pitch)/(np.sin(pitch)**2 + np.cos(pitch)**2)
        T_inv[1,1]=np.cos(pitch)/(np.sin(pitch)**2 + np.cos(pitch)**2)
        T_inv[1,2]=0
        T_inv[2,0]=np.sin(yaw)*np.cos(pitch)/(np.sin(pitch)**2*np.cos(yaw) + np.cos(pitch)**2*np.cos(yaw))
        T_inv[2,1]=np.sin(pitch)*np.sin(yaw)/(np.sin(pitch)**2*np.cos(yaw) + np.cos(pitch)**2*np.cos(yaw))
        T_inv[2,2]=1
        return T_inv

    ''' Returns the full 9DOF states
    '''
    def get_state(self):
        euler = R.from_quat([self.vehicle_odometry.q[1], self.vehicle_odometry.q[2], self.vehicle_odometry.q[3], self.vehicle_odometry.q[0]]).as_euler('ZYX') # Scipy uses intrinsic rotations when capitalized seq and scalar (w) last format for quaternions
        current_state = np.array([
            self.vehicle_odometry.position[0],
            self.vehicle_odometry.position[1],
            self.vehicle_odometry.position[2],
            euler[2], # roll (scipy as_euler 'zyx' gives yaw, pitch, roll)
            euler[1],
            euler[0],
            self.servo_state.position[0],
            self.servo_state.position[1],
            self.servo_state.position[2]
        ])
        return current_state

    def transformation_to_vector(self, HTM):
        vector = np.zeros(6)
        vector[0] = HTM[0,3]
        vector[1] = HTM[1,3]
        vector[2] = HTM[2,3]
        vector[3] = np.arctan2(HTM[2,1], HTM[2,2])  # Rx, roll
        vector[4] = -np.arcsin(HTM[2,0])            # Ry, pitch
        vector[5] = np.arctan2(HTM[1,0], HTM[0,0])  # Rz, yaw

        return vector

    ''' Get homogeneous transformation matrix from pose vector (XYZ, RPY) with the (intrinsic) Z-Y'-X" or (extrinsic) XYZ convention.
    '''
    def vector_to_transformation(self, vector):
        roll = vector[3]
        pitch = vector[4]
        yaw = vector[5]
        HTM = np.array(
            [[np.cos(yaw)*np.cos(pitch), -np.sin(yaw)*np.cos(roll) + np.sin(pitch)*np.sin(roll)*np.cos(yaw), np.sin(yaw)*np.sin(roll) + np.sin(pitch)*np.cos(yaw)*np.cos(roll), vector[0]],
             [np.sin(yaw)*np.cos(pitch), np.sin(yaw)*np.sin(pitch)*np.sin(roll) + np.cos(yaw)*np.cos(roll), np.sin(yaw)*np.sin(pitch)*np.cos(roll) - np.sin(roll)*np.cos(yaw), vector[1]],
             [-np.sin(pitch), np.sin(roll)*np.cos(pitch), np.cos(pitch)*np.cos(roll), vector[2]],
             [0., 0., 0., 1.]])
        return HTM

    def broadcast_tf2(self, T:np.array, parent_frame:str, child_frame:str):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = parent_frame
        t.child_frame_id = child_frame
        t.transform.translation.x = T[0,3]
        t.transform.translation.y = T[1,3]
        t.transform.translation.z = T[2,3]
        rot = R.from_matrix(T[:3,:3])
        quat = rot.as_quat()  # Returns in (x, y, z, w) format
        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]

        self.broadcaster_tf2.sendTransform(t)

    def publish_twist(self, vec:np.array, publisher):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.twist.linear.x = vec[0]
        msg.twist.linear.y = vec[1]
        msg.twist.linear.z = vec[2]
        msg.twist.angular.x = vec[3]
        msg.twist.angular.y = vec[4]
        msg.twist.angular.z = vec[5]
        publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = VelocityBasedATS()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()