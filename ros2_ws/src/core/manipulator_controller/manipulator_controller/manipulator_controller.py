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
        Joint controller for the UAM's 3-DOF arm(s).

        Drives one or more serial arms (q1, q2, q3 per arm) by streaming joint
        commands to the servo driver on /servo/in/state; joint feedback comes
        back on /servo/out/state.

        The controller is a two-stage pipeline:

          * Stage 1 (normalize) -- each of the four command interfaces is reduced
            to a joint-space reference, either a joint position target (q_cmd) or
            a joint velocity target (qdot_cmd), regardless of the servo mode.
          * Stage 2 (emit) -- the reference is converted to the field the active
            `servo_mode` requires, cross-converting when the reference type and
            the servo mode do not line up:

              servo_mode = 'velocity' (servos run their internal velocity loop):
                q_cmd    -> PD law qdot = kp*(q_cmd - q) - kd*qdot, zeroed within
                            `tolerance`, published on JointState.velocity
                qdot_cmd -> passed straight through on JointState.velocity
              servo_mode = 'position' (servos run their internal position loop):
                q_cmd    -> published straight through on JointState.position
                qdot_cmd -> integrated into a moving setpoint q_set += qdot*dt
                            (seeded from the measured pose, clamped to joint
                            limits), published on JointState.position

        Only the active field is filled; the other is left empty so that a
        servo_mode mismatch with the driver is detectable downstream.

        Each arm exposes four command interfaces (indexed by arm):
          * /manipulator/in/arm_<i>/joint/position  (JointState) -- target joint
            angles.
          * /manipulator/in/arm_<i>/joint/velocity  (JointState) -- joint rates.
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

        For multi-arm morphologies (n_arms > 1) a collision guard keeps opposed
        arms' base joints at least `minimum_pivot_distance` apart. In velocity
        mode it scales down only the *closing* joint velocities (ramped over
        `pivot_ramp`); in position mode it clamps the commanded base-joint
        setpoints to a hard floor/ceiling at `minimum_pivot_distance` from the
        other arm's present base position. Link geometry is set by `link_lengths`.

        TODO: Implement singularity guards and out of workspace guards (for position commands)
        """
        # Param definitions
        super().__init__('manipulator_controller')
        self.declare_parameter('frequency', 10.) # Hz
        self.declare_parameter('servo_mode', 'velocity') # 'position' or 'velocity' -- must match the servo driver
        self.declare_parameter('minimum_pivot_distance', 1.5) # rad
        self.declare_parameter('pivot_ramp', 1.0) # dimensionless
        self.declare_parameter('verbose', False) # boolean
        self.declare_parameter('n_arms', 1) # integer
        self.declare_parameter('kp', 2.0) # joint position control p gain (velocity mode only)
        self.declare_parameter('kd', 0.3) # joint position control d gain (velocity mode only)
        self.declare_parameter('tolerance', 0.2) # velocity-mode position tolerance
        self.declare_parameter('link_lengths', [0.12025, 0.336, 0.325])

        # Param grabbing
        self.frequency = self.get_parameter('frequency').get_parameter_value().double_value
        self.servo_mode = self.get_parameter('servo_mode').get_parameter_value().string_value
        self.min_pivot_distance = self.get_parameter('minimum_pivot_distance').get_parameter_value().double_value
        self.pivot_ramp = self.get_parameter('pivot_ramp').get_parameter_value().double_value
        self.verbose = self.get_parameter('verbose').get_parameter_value().bool_value
        self.n_arms = self.get_parameter('n_arms').get_parameter_value().integer_value
        self.kp = self.get_parameter('kp').get_parameter_value().double_value # Arm position control proportional gain
        self.kd = self.get_parameter('kd').get_parameter_value().double_value # Arm position control derivative gain
        self.tol = self.get_parameter('tolerance').get_parameter_value().double_value

        if self.servo_mode not in ('position', 'velocity'):
            self.get_logger().error(
                f"Invalid servo_mode '{self.servo_mode}', expected 'position' or 'velocity'. "
                f"Falling back to 'velocity'.")
            self.servo_mode = 'velocity'

        # Per-joint limits for the position-mode integrator, as FLAT length-(n_arms*3)
        # vectors indexed exactly like the servos (arm 0 joints 0..2, arm 1 joints
        # 3..5, ...), matching the driver's servos.min_angles/max_angles. Both limits
        # zero for a joint means no limit on that joint. Default: no limits anywhere.
        n_joints = self.n_arms * 3
        no_limits = [0.0] * n_joints
        self.declare_parameter('min_angles', no_limits)
        self.declare_parameter('max_angles', no_limits)
        min_angles = list(self.get_parameter('min_angles').get_parameter_value().double_array_value)
        max_angles = list(self.get_parameter('max_angles').get_parameter_value().double_array_value)
        if len(min_angles) != n_joints:
            self.get_logger().error(
                f"min_angles length {len(min_angles)} != n_arms*3 ({n_joints}); ignoring (no lower limits)")
            min_angles = no_limits
        if len(max_angles) != n_joints:
            self.get_logger().error(
                f"max_angles length {len(max_angles)} != n_arms*3 ({n_joints}); ignoring (no upper limits)")
            max_angles = no_limits

        self.arm_params = {
            'link_lengths': self.get_parameter('link_lengths').get_parameter_value().double_array_value,
            'min_angles': min_angles,
            'max_angles': max_angles,
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
        self.last_time = None  # for measured-dt integration in position mode

        # Timer
        self.get_logger().info(f"Starting manipulator controller in '{self.servo_mode}' servo mode...")
        self.period = 1./self.frequency
        self.timer = self.create_timer(self.period, self.execute)

    def execute(self):
        if self.servo_state is None:
            return

        # Measured dt for the position-mode integrator. Guard against the first
        # cycle and against pathological gaps (use the nominal period instead).
        now = self.get_clock().now()
        if self.last_time is None:
            dt = self.period
        else:
            dt = (now - self.last_time).nanoseconds * 1e-9
            if dt <= 0.0 or dt > 1.0:
                dt = self.period
        self.last_time = now

        cmd_servo = [0.0] * self.n_arms * 3
        for arm in self.arms:
            if arm.q_cmd is None and arm.qdot_cmd is None:
                self.get_logger().warn(
                    f"[Arm {arm.index}] no command received yet", throttle_duration_sec=1)
            qvec = arm.compute_command(self.servo_mode, dt, self.kp, self.kd, self.tol)
            if len(qvec) == 3:
                cmd_servo[arm.joint_slice] = qvec
            else:
                self.get_logger().warn(f"[Manipulator] command for arm {arm.index} is not length 3: {len(qvec)}")

        # Publish command to servo driver node
        self.publish_servo_command(cmd_servo)

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

    def publish_servo_command(self, cmd):
        if self.n_arms > 1:
            cmd = self.apply_collision_guard(cmd)

        msg = JointState()
        msg.name = ['q'+str(i+1) for i in range(len(cmd))]
        msg.header.stamp = self.get_clock().now().to_msg()
        # Fill ONLY the field the active servo mode uses; leave the other empty so
        # a servo_mode mismatch with the driver is detectable downstream.
        if self.servo_mode == 'velocity':
            msg.velocity = cmd
        else:  # position
            msg.position = cmd
        self.publisher_servo.publish(msg)

    def apply_collision_guard(self, cmd_servo):
        for a, b in itertools.combinations(self.arms, 2):
            if a.q_present is None or b.q_present is None:
                continue
            h, l = (a, b) if a.q_present[0] >= b.q_present[0] else (b, a)   # high / low side
            ih, il = h.index * 3, l.index * 3

            if self.servo_mode == 'velocity':
                gap = h.q_present[0] - l.q_present[0]
                # high arm closes when moving negative; low arm closes when moving positive.
                # Scale ONLY the closing component -> opening stays at full speed.
                scale = float(np.clip((gap - self.min_pivot_distance) / self.pivot_ramp, 0.0, 1.0))
                if cmd_servo[ih] < 0.0:
                    cmd_servo[ih] *= scale
                if cmd_servo[il] > 0.0:
                    cmd_servo[il] *= scale
            else:  # position: clamp each base-joint setpoint to a hard floor/ceiling
                # at minimum_pivot_distance from the OTHER arm's present base position.
                floor_high = l.q_present[0] + self.min_pivot_distance
                ceil_low = h.q_present[0] - self.min_pivot_distance
                if cmd_servo[ih] < floor_high:
                    cmd_servo[ih] = floor_high
                    if h.q_set is not None:  # keep the integrator from winding up past the clamp
                        h.q_set[0] = floor_high
                if cmd_servo[il] > ceil_low:
                    cmd_servo[il] = ceil_low
                    if l.q_set is not None:
                        l.q_set[0] = ceil_low

        return cmd_servo

class Arm:
    def __init__(self, index, arm_params):
        self.index = index
        self.link_lengths = arm_params['link_lengths']
        self.joint_slice = slice(index*3, index*3+3)
        # This arm's slice of the flat, per-joint limit vectors.
        self.min_angles = arm_params['min_angles'][self.joint_slice]
        self.max_angles = arm_params['max_angles'][self.joint_slice]
        self.q_present = None
        self.q_cmd = None
        self.qdot_present = None
        self.qdot_cmd = None
        self.q_set = None  # integrated position setpoint for velocity commands in position mode

        self.task_positions = None
        self.task_velocities = None

    def on_task_position(self, msg):
        solution = self.position_inverse_kinematics(target_position=[msg.point.x, msg.point.y, msg.point.z])
        if solution is not None:
            self.q_cmd = solution
            self.qdot_cmd = None
            self.q_set = None  # entering position intent; force the integrator to re-seed next time
        else:
            return

    def on_joint_position(self, msg):
        self.q_cmd = [msg.position[0], msg.position[1], msg.position[2]]
        self.qdot_cmd = None
        self.q_set = None  # entering position intent; force the integrator to re-seed next time

    def on_task_velocity(self, msg):
        solution = self.velocity_inverse_kinematics(target_velocity=[msg.point.x, msg.point.y, msg.point.z])
        if solution is not None:
            self.qdot_cmd = solution
            self.q_cmd = None

    def on_joint_velocity(self, msg):
        self.qdot_cmd = [msg.velocity[0], msg.velocity[1], msg.velocity[2]]
        self.q_cmd = None

    def compute_command(self, servo_mode, dt, kp, kd, tol):
        """Stage 2: convert the current joint-space reference into the field the
        active servo mode needs. Always returns a length-3 list."""
        if servo_mode == 'velocity':
            if self.q_cmd is not None and self.qdot_cmd is None:  # position intent -> PD
                qdot_cmd = []
                error = 0.
                for i in range(len(self.q_cmd)):
                    qdot_cmd.append(kp*(self.q_cmd[i] - self.q_present[i]) - kd*self.qdot_present[i])
                    error += abs(self.q_cmd[i] - self.q_present[i])**2
                if np.sqrt(error) <= tol:
                    qdot_cmd = [0.0 for _ in self.q_cmd]  # stop within epsilon
                return qdot_cmd
            elif self.qdot_cmd is not None:  # velocity intent -> passthrough
                return list(self.qdot_cmd)
            else:  # no command yet -> stop
                return [0.0, 0.0, 0.0]
        else:  # position mode
            if self.qdot_cmd is not None:  # velocity intent -> integrate into a setpoint
                if self.q_set is None:
                    self.q_set = list(self.q_present)  # seed from the measured pose
                for i in range(len(self.q_set)):
                    self.q_set[i] += self.qdot_cmd[i] * dt
                self._clamp_to_limits(self.q_set)
                return list(self.q_set)
            elif self.q_cmd is not None:  # position intent -> passthrough
                return list(self.q_cmd)
            else:  # no command yet -> hold present pose
                return list(self.q_present)

    def _clamp_to_limits(self, q):
        """Clamp joint setpoints in place. Both limits zero for a joint = no limit."""
        for i in range(len(q)):
            if not (self.min_angles[i] == 0.0 and self.max_angles[i] == 0.0):
                q[i] = min(max(q[i], self.min_angles[i]), self.max_angles[i])

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
