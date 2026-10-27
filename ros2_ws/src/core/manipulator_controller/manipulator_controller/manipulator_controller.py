import math
import rclpy
from rclpy.node import Node

import numpy as np
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PointStamped
import itertools

class ManipulatorController(Node):
    def __init__(self):
        """
        Joint-velocity controller for the UAM's 3-DOF arm(s).

        Drives one or more serial arms (q1, q2, q3 per arm) by streaming joint
        velocity commands to the servo driver. The servos always run in velocity mode,
        so every control mode below ultimately resolves to a joint-velocity
        command on /servo/in/state; joint feedback comes back on
        /servo/out/state.

        Each arm exposes four command interfaces (indexed by arm):
          * /manipulator/in/arm_<i>/joint/position  (JointState) -- target joint
            angles, tracked by a PD law: qdot = kp*(q_cmd - q) - kd*qdot, zeroed
            once the joint error is within `tolerance`.
          * /manipulator/in/arm_<i>/joint/velocity  (JointState) -- joint rates
            passed straight through.
          * /manipulator/in/arm_<i>/task/position   (PointStamped) -- an
            end-effector position [x, y, z] in the arm base frame, converted to a
            joint target via the analytic inverse kinematics.
          * /manipulator/in/arm_<i>/task/velocity   (PointStamped) -- an
            end-effector velocity, mapped to joint rates via the pseudo-inverse
            of the arm Jacobian.
        The most recent command wins; a position command clears any standing
        velocity command and vice versa.

        Each cycle (at `frequency` Hz) the node also publishes the measured
        end-effector pose and velocity from forward kinematics on
        /manipulator/out/arm_<i>/task/{position,velocity}.

        For multi-arm morphologies (n_arms > 1) a collision guard scales down
        only the *closing* joint velocities as the arms' base joints come within
        `minimum_pivot_distance` (ramped over `pivot_ramp`), so opposed arms do
        not drive into each other while still allowing them to open at full
        speed. Link geometry is set by `link_lengths`.

        TODO: Implement singularity guards and out of workspace guards (for position commands)
        """
        # Param definitions
        super().__init__('manipulator_controller')
        self.declare_parameter('frequency', 10.) # Hz
        self.declare_parameter('minimum_pivot_distance', 1.5) # rad
        self.declare_parameter('pivot_ramp', 1.0) # dimensionless
        self.declare_parameter('verbose', False) # boolean
        self.declare_parameter('n_arms', 1) # integer
        self.declare_parameter('kp', 2.0) # joint position control p gain
        self.declare_parameter('kd', 0.3) # joint position control d gain
        self.declare_parameter('tolerance', 0.2) # 
        self.declare_parameter('link_lengths', [0.12025, 0.336, 0.325])

        # Param grabbing
        self.frequency = self.get_parameter('frequency').get_parameter_value().double_value
        self.min_pivot_distance = self.get_parameter('minimum_pivot_distance').get_parameter_value().double_value
        self.pivot_ramp = self.get_parameter('pivot_ramp').get_parameter_value().double_value
        self.verbose = self.get_parameter('verbose').get_parameter_value().bool_value
        self.n_arms = self.get_parameter('n_arms').get_parameter_value().integer_value
        self.kp = self.get_parameter('kp').get_parameter_value().double_value # Arm position control proportional gain
        self.kd = self.get_parameter('kd').get_parameter_value().double_value # Arm position control derivative gain
        self.tol = self.get_parameter('tolerance').get_parameter_value().double_value
        self.arm_params = {
            'link_lengths': self.get_parameter('link_lengths').get_parameter_value().double_array_value,
        }

        # Servo interface
        self.subscriber_servo = self.create_subscription(JointState, '/servo/out/state', self.callback_servo, 10)
        self.publisher_servo = self.create_publisher(JointState, '/servo/in/state', 10)

        # Controller interface
        self.task_pos_subscribers_ = []
        self.joint_pos_subscribers_ = []
        self.task_vel_subscribers_ = []
        self.joint_vel_subscribers_ = []

        self.task_pos_publishers_ = []
        self.task_vel_publishers_ = []
        self.arms = [Arm(i, self.arm_params) for i in range(self.n_arms)]
        for arm in self.arms:
            self.task_pos_subscribers_.append(self.create_subscription(PointStamped, '/manipulator/in/arm_'+str(arm.index)+'/task/position', arm.on_task_position, 10))
            self.joint_pos_subscribers_.append(self.create_subscription(JointState, '/manipulator/in/arm_'+str(arm.index)+'/joint/position', arm.on_joint_position, 10))
            self.task_vel_subscribers_.append(self.create_subscription(PointStamped, '/manipulator/in/arm_'+str(arm.index)+'/task/velocity', arm.on_task_velocity, 10))
            self.joint_vel_subscribers_.append(self.create_subscription(JointState, '/manipulator/in/arm_'+str(arm.index)+'/joint/velocity', arm.on_joint_velocity, 10))

            self.task_pos_publishers_.append(self.create_publisher(PointStamped, '/manipulator/out/arm_'+str(arm.index)+'/task/position', 10))
            self.task_vel_publishers_.append(self.create_publisher(PointStamped, '/manipulator/out/arm_'+str(arm.index)+'/task/velocity', 10))
        # Data
        self.servo_state = None

        # Timer
        self.get_logger().info("Starting manipulator controller...")
        self.period = 1./self.frequency
        self.timer = self.create_timer(self.period, self.execute)

    def execute(self):
        if self.servo_state is None:
            return

        cmd_servo = [0.0] * self.n_arms * 3
        for arm in self.arms:
            if arm.q_cmd is not None and arm.qdot_cmd is None: # Commanded intended q -> position mode
                # Run control loop to determine cmd_servo
                qdot_cmd = []
                error = 0.

                for i in range(len(arm.q_cmd)):
                    qd = (self.kp*(arm.q_cmd[i] - arm.q_present[i]) - self.kd*arm.qdot_present[i]) # Simple P controller to reach target position
                    qdot_cmd.append(qd)
                    error += abs(arm.q_cmd[i]-arm.q_present[i])**2
                error = np.sqrt(error)

                if error <= self.tol:
                    qdot_cmd = [0.0 for x in arm.q_cmd]  # Stop the servos if within epsilon

                if len(qdot_cmd) ==3:
                    cmd_servo[arm.joint_slice] = qdot_cmd
                else:
                    self.get_logger().warn(f"[Manipulator] qdot_cmd is not length 3: {len(qdot_cmd)}")

            elif arm.q_cmd is None and arm.qdot_cmd is not None: # Commanded intended qdot -> velocity mode
                if len(arm.qdot_cmd) == 3:
                    cmd_servo[arm.joint_slice] = arm.qdot_cmd
                else:
                    self.get_logger().warn(f"[Manipulator] qdot_cmd is not length 3: {len(qdot_cmd)}")

            else: # if no commands were published yet
                self.get_logger().warn(f"[Arm {arm.index}] q_cmd = {arm.q_cmd}, qdot_cmd = {arm.qdot_cmd}", throttle_duration_sec=1)

        # Publish servo velocities to servo driver node
        self.publish_servo_velocities(cmd_servo)

        # Publish most recent arm position and velocities
        for i, arm in enumerate(self.arms):
            arm.update()
            self.publish_arm_task_positions(i)
            self.publish_arm_task_velocities(i)

    def callback_servo(self, msg):
        """ Update the servo data
        """
        self.servo_state = msg
        for arm in self.arms:
            arm.q_present = msg.position[arm.joint_slice]
            arm.qdot_present = msg.velocity[arm.joint_slice]

    def publish_arm_task_positions(self, index):
            positions = self.arms[index].task_positions
            if len(positions) != 3:
                self.get_logger().warn("Position vector is not 3")
                return
            msg = PointStamped()
            msg.point.x, msg.point.y, msg.point.z = positions
            msg.header.stamp = self.get_clock().now().to_msg()
            self.task_pos_publishers_[index].publish(msg)   
     
    def publish_arm_task_velocities(self, index):
            velocities = self.arms[index].task_velocities
            if len(velocities) != 3:
                self.get_logger().warn("Position vector is not 3")
                return
            msg = PointStamped()
            msg.point.x, msg.point.y, msg.point.z = velocities
            msg.header.stamp = self.get_clock().now().to_msg()
            self.task_vel_publishers_[index].publish(msg)        

    def publish_servo_velocities(self, velocities):
        if self.n_arms > 1:
            velocities = self.apply_collision_guard(velocities)

        msg = JointState()
        msg.name = ['q'+str(i+1) for i in range(len(velocities))]
        msg.position = [0.0 for i in range(len(velocities))]
        msg.velocity = velocities
        msg.effort = [0.0 for i in range(len(velocities))]
        msg.header.stamp = self.get_clock().now().to_msg()
        self.publisher_servo.publish(msg)

    def apply_collision_guard(self, cmd_servo):
        for a, b in itertools.combinations(self.arms, 2):
            h, l = (a, b) if a.q_present[0] >= b.q_present[0] else (b, a)   # high / low side
            ih, il = h.index * 3, l.index * 3
            gap = h.q_present[0] - l.q_present[0]

            # high arm closes when moving negative; low arm closes when moving positive.
            # Scale ONLY the closing component -> opening stays at full speed.
            scale = float(np.clip((gap - self.min_pivot_distance) / self.pivot_ramp, 0.0, 1.0))
            if cmd_servo[ih] < 0.0:
                cmd_servo[ih] *= scale
            if cmd_servo[il] > 0.0:
                cmd_servo[il] *= scale

        return cmd_servo

class Arm:
    def __init__(self, index, arm_params):
        self.index = index
        self.link_lengths = arm_params['link_lengths']
        self.joint_slice = slice(index*3, index*3+3)
        self.q_present = None
        self.q_cmd = None
        self.qdot_present = None
        self.qdot_cmd = None

        self.task_positions = None
        self.task_velocities = None

    def on_task_position(self, msg):
        solution = self.position_inverse_kinematics(target_position=[msg.point.x, msg.point.y, msg.point.z])
        if solution is not None:
            self.q_cmd = solution
            self.qdot_cmd = None
        else:
            return

    def on_joint_position(self, msg):
        self.q_cmd = [msg.position[0], msg.position[1], msg.position[2]]
        self.qdot_cmd = None

    def on_task_velocity(self, msg):
        self.qdot_cmd = self.velocity_inverse_kinematics(target_velocity=[msg.point.x, msg.point.y, msg.point.z])
        self.q_cmd = None

    def on_joint_velocity(self, msg):
        self.qdot_cmd = [msg.velocity[0], msg.velocity[1], msg.velocity[2]]
        self.q_cmd = None

    def update(self):
        self.task_positions = self.position_forward_kinematics()
        self.task_velocities = self.velocity_forward_kinematics()

    def position_forward_kinematics(self):
        q_1 = self.q_present[0]
        q_2 = self.q_present[1]
        q_3 = self.q_present[2]
        x_BS = -self.link_lengths[1]*np.sin(q_2) - self.link_lengths[2]*np.sin(q_2)*np.cos(q_3)
        y_BS = self.link_lengths[0]*np.sin(q_1) + self.link_lengths[1]*np.sin(q_1)*np.cos(q_2) + self.link_lengths[2]*np.sin(q_1)*np.cos(q_2)*np.cos(q_3) + self.link_lengths[2]*np.sin(q_3)*np.cos(q_1)
        z_BS = -self.link_lengths[0]*np.cos(q_1) - self.link_lengths[1]*np.cos(q_1)*np.cos(q_2) + self.link_lengths[2]*np.sin(q_1)*np.sin(q_3) - self.link_lengths[2]*np.cos(q_1)*np.cos(q_2)*np.cos(q_3)
        return [x_BS, y_BS, z_BS]

    def velocity_forward_kinematics(self):
        return self.jacobian() @ np.array(self.qdot_present)        

    def position_inverse_kinematics(self, target_position:list, tol = 1e-9) -> list:
        if self.q_present is None:
            return None

        X = target_position[0]
        Y = target_position[1]
        Z = target_position[2]
        # Step 1: compute D
        S = X**2 + Y**2 + Z**2
        D = S - (self.link_lengths[0]**2 + self.link_lengths[1]**2 + self.link_lengths[2]**2)
        
        # Step 2: quadratic coefficients for u = cos(q3)
        a = 4 * self.link_lengths[2]**2 * (self.link_lengths[1]**2 - self.link_lengths[0]**2)
        b = -4 * self.link_lengths[1] * self.link_lengths[2] * (D + 2 * self.link_lengths[0]**2)
        c = D**2 - 4 * self.link_lengths[0]**2 * self.link_lengths[1]**2 + 4 * self.link_lengths[0]**2 * X**2

        solutions = []

        # Step 3: solve quadratic
        if abs(a) < tol:  # degenerate to linearf
            if abs(b) < tol:
                return None  # no equation to solve
            u_roots = [-c / b]
        else:
            disc = b**2 - 4*a*c
            if disc < -tol:
                return None  # no real solutions
            elif abs(disc) < tol:
                u_roots = [-b / (2*a)]
            else:
                sqrt_disc = math.sqrt(max(disc, 0.0))
                u_roots = [(-b + sqrt_disc) / (2*a),
                        (-b - sqrt_disc) / (2*a)]

        # Step 4: loop over cos(q3) solutions
        for u in u_roots:
            if abs(u) > 1 + 1e-9:
                continue  # invalid
            u = max(-1, min(1, u))  # clamp

            for s3_sign in [1, -1]:
                s3 = s3_sign * math.sqrt(max(0.0, 1 - u**2))
                C = self.link_lengths[1] + self.link_lengths[2] * u

                if abs(C) < tol:
                    continue  # singular, skip or handle separately

                # q2 from atan2 form
                num = -X
                den = (D - 2 * self.link_lengths[1] * self.link_lengths[2] * u) / (2 * self.link_lengths[0])
                q2 = math.atan2(num, den)

                # compute A, B
                A = self.link_lengths[0] + C * math.cos(q2)
                B = self.link_lengths[2] * s3

                # q1
                q1 = math.atan2(Y, -Z) - math.atan2(B, A)

                # q3
                q3 = math.atan2(s3, u)
                solutions.append((q1, q2, q3))

        # Step 5: Find lowest Euclidean distance solution
        lowest_diff = np.inf
        if not solutions:
            return None
        else:
            for solution in solutions:
                diff = np.linalg.norm(np.array(solution) - np.array(self.q_present[0:3]))
                if diff < lowest_diff:
                    lowest_diff = diff
                    best_solution = solution

        return best_solution

    def velocity_inverse_kinematics(self, target_velocity:list) -> list:
        if self.q_present is None:
            return

        jacobian = self.jacobian()
        
        # Evaluate jacobian
        jacobian_pinv = np.linalg.pinv(jacobian)
        joint_velocities = jacobian_pinv @ np.array(target_velocity)

        return joint_velocities.tolist() 

    def jacobian(self):
        jac = np.empty((3,3))
        q1 = self.q_present[0]
        q2 = self.q_present[1]
        q3 = self.q_present[2]

        # Evaluate jacobian
        jac[0,0]=0.0
        jac[0,1]=-self.link_lengths[1]*math.cos(q2) - self.link_lengths[2]*math.cos(q2)*math.cos(q3)
        jac[0,2]=self.link_lengths[2]*math.sin(q2)*math.sin(q3)
        jac[1,0]=self.link_lengths[0]*math.cos(q1) + self.link_lengths[1]*math.cos(q1)*math.cos(q2) - self.link_lengths[2]*math.sin(q1)*math.sin(q3) + self.link_lengths[2]*math.cos(q1)*math.cos(q2)*math.cos(q3)
        jac[1,1]=-self.link_lengths[1]*math.sin(q1)*math.sin(q2) - self.link_lengths[2]*math.sin(q1)*math.sin(q2)*math.cos(q3)
        jac[1,2]=-self.link_lengths[2]*math.sin(q1)*math.sin(q3)*math.cos(q2) + self.link_lengths[2]*math.cos(q1)*math.cos(q3)
        jac[2,0]=self.link_lengths[0]*math.sin(q1) + self.link_lengths[1]*math.sin(q1)*math.cos(q2) + self.link_lengths[2]*math.sin(q1)*math.cos(q2)*math.cos(q3) + self.link_lengths[2]*math.sin(q3)*math.cos(q1)
        jac[2,1]=self.link_lengths[1]*math.sin(q2)*math.cos(q1) + self.link_lengths[2]*math.sin(q2)*math.cos(q1)*math.cos(q3)
        jac[2,2]=self.link_lengths[2]*math.sin(q1)*math.cos(q3) + self.link_lengths[2]*math.sin(q3)*math.cos(q1)*math.cos(q2)
        return jac

def main():
    rclpy.init(args=None)
    manipulator_controller = ManipulatorController()
    rclpy.spin(manipulator_controller)

    # Destroy the node explicitly
    manipulator_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()