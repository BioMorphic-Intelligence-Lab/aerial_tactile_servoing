import datetime
import numpy as np

from mission_director.base_classes.tactile_mission_director import TactileMissionDirector, run_mission

# TODO: Add enum class for state numbers

class PinchGraspMission(TactileMissionDirector):
    """Tactile pinch grasping dry test mission.


    """

    def __init__(self):
        super().__init__('mission_director')

        # --- Mission-graph constants (from the per-mission YAML) ---
        self.declare_parameter('im.takeoff_altitude', 1.5)  # [m]
        self.declare_parameter('im.pre_contact_arm_position', [1.57, 0.0, -1.57, -1.57, 0.0, 1.57])  # [rad] q1,q2,q3
        self.declare_parameter('im.pre_grasp_arm_position', [2.0943, 0.0, 1.0, -2.0943, 0.0, -1.0])
        self.declare_parameter('im.land_arm_position', [1.57, 0.0, -1.57, -1.57, 0.0, 1.57])  # [rad]


        # Make into parameters
        self.kp_shear = 0.04
        self.kp_force = 0.01
        self.squeeze_force_ref = 4.0
        self.approach_speed = 0.45

        self.takeoff_altitude = self.get_parameter('im.takeoff_altitude').get_parameter_value().double_value
        self.pre_contact_arm_position = list(self.get_parameter('im.pre_contact_arm_position').get_parameter_value().double_array_value)
        self.pre_grasp_arm_position = list(self.get_parameter('im.pre_grasp_arm_position').get_parameter_value().double_array_value)
        self.land_arm_position = list(self.get_parameter('im.land_arm_position').get_parameter_value().double_array_value)

        self.get_logger().info('Tactile Pinch Grasping mission initialized.')

    def execute(self):
        match self.FSM_state:
            case "entrypoint":
                self.state_entrypoint(next_state="pre_grasp_config")

            case "pre_grasp_config":
                self.state_move_arms_joint_position(q_des=self.pre_grasp_arm_position, next_state="wait", state_number=24)

            case "wait":
                if self.first_state_loop:
                    self.first_state_loop = False
                    if not self.sim:
                        self.set_new_ssim_ref(True) # Get new SSIM reference image
                self.state_do_nothing(next_state="approach_grasp")

            case "approach_grasp":
                # self.get_logger().info(f"{self.tactip_contact_}")
                next_state="hold_grasp"
                if abs(self.arm_task_positions_[0][1]) < 0.05 or abs(self.arm_task_positions_[1][1]) < 0.05:
                    self.get_logger().info("Transition by approach of center")
                    next_state="control_grasp"
                    transition = True
                elif self.tactip_contact_[0] and not self.tactip_contact_[1]:
                    self.get_logger().info("Transition by contact on 0")
                    next_state="approach_grasp_0"
                    transition = True
                elif not self.tactip_contact_[0] and self.tactip_contact_[1]:
                    self.get_logger().info("Transition by contact on 1")
                    next_state="approach_grasp_1"
                    transition = True
                else:
                    transition = False
                self.state_move_arms_task_velocity(pdot_des=[0.0, -self.approach_speed, 0.0, 0.0, self.approach_speed, 0.0], transition=transition, next_state=next_state, state_number=25)

            case "approach_grasp_0":
                if all(self.tactip_contact_):
                    transition = True
                else:
                    transition = False

                self.state_move_arms_task_velocity(pdot_des=[0.0, 0.0, 0.0, 0.0, self.approach_speed, 0.0], transition=transition, next_state="control_grasp", state_number=26)

            case "approach_grasp_1":
                if all(self.tactip_contact_):
                    transition = True
                else:
                    transition = False
                self.state_move_arms_task_velocity(pdot_des=[0.0, -self.approach_speed, 0.0, 0.0, 0.0, 0.0], transition=transition, next_state="control_grasp", state_number=27)

            case "control_grasp":
                # Test: calculate the 

                # Step 1: calculate 
                v_c_cmd = np.zeros(self.n_arms*3)
                p_BC = np.zeros(self.n_arms*3)
                R_CS_block = np.zeros((self.n_arms*3, self.n_arms*3))
                R_BS_block = np.zeros((self.n_arms*3, self.n_arms*3))
                for n in range(self.n_arms):
                    P_BC = self.evaluate_P_BC(
                        *self.servo_state.position[n*3 : n*3+3], 
                        np.deg2rad(self.tactip_pose_[n].angular.x), 
                        np.deg2rad(self.tactip_pose_[n].angular.y), 
                        self.tactip_pose_[n].linear.x/1000., 
                        self.tactip_pose_[n].linear.y/1000., 
                        self.tactip_pose_[n].linear.z/1000.)

                    p_BC[n*3 : n*3+3] = P_BC[0:3, 3]

                    v_c_cmd[n*3 : n*3+3] = np.array([
                        self.kp_shear * np.deg2rad(self.tactip_pose_[n].angular.y),
                        0.0, #-self.kp_shear * np.deg2rad(self.tactip_pose_[n].angular.x),
                        -self.kp_force * (self.squeeze_force_ref - self.tactip_wrench_[n].force.z)
                    ])
                    R_CS_block[n*3:n*3+3, n*3:n*3+3] = self.evaluate_R_CS(np.deg2rad(self.tactip_pose_[n].angular.x), np.deg2rad(self.tactip_pose_[n].angular.y))
                    R_BS_block[n*3:n*3+3, n*3:n*3+3] = self.evaluate_R_BS(*self.servo_state.position[n*3:n*3+3])
                # Print direction vector
                direction = p_BC[0:3] - p_BC[3:6]
                angle = self.calculate_angle(direction, np.array([0.0, 1.0, 0.0]))
                midpoint = (p_BC[0:3] + p_BC[3:6])/2.
                self.get_logger().info(f"Dir between fingertips: [{direction[0]:.2f}, {direction[1]:.2f}, {direction[2]:.2f}], angle: {angle:.2f} rad/{angle*180./np.pi:.2f} deg, midpoint [{midpoint[0]:.2f}, {midpoint[1]:.2f}, {midpoint[2]:.2f}]", throttle_duration_sec=1.0)
                # Invert rotation
                R_SC_block = np.linalg.inv(R_CS_block)
                R_BC_block = R_BS_block@R_SC_block
                v_b_cmd = R_BC_block@v_c_cmd # Rotate from contact frame into sensor frame

                transition = False
                self.get_logger().info(f"[29] Grasp force: {self.tactip_wrench_[0].force.z:.2f} N | {self.tactip_wrench_[1].force.z:.2f} N, {(datetime.datetime.now()-self.state_start_time).seconds:.1f}/20 sec", throttle_duration_sec=1.0)
                if (datetime.datetime.now()-self.state_start_time).seconds > 30.:
                    transition = True

                self.state_move_arms_task_velocity(pdot_des=v_b_cmd, transition=transition, next_state="release", state_number=29)

            case "hold_grasp":
                transition = False
                self.get_logger().info(f"[28] Grasp force: {self.tactip_wrench_[0].force.z:.2f} N | {self.tactip_wrench_[1].force.z:.2f} N, {(datetime.datetime.now()-self.state_start_time).seconds:.1f}/20 sec", throttle_duration_sec=1.0)
                
                if (datetime.datetime.now()-self.state_start_time).seconds > 40.:
                    transition = True
                elif self.input_state == 1:
                    transition = True
                self.state_move_arms_task_velocity(pdot_des=[0.0, -0.0, 0.0, 0.0, 0.0, 0.0], transition=transition, next_state="release", state_number=28)

            case "release":
                self.state_move_arms_joint_position(q_des=self.land_arm_position, next_state="done")

            case "done":
                self.state_done(next_state="entrypoint")

            # --- Do not remove these states ---
            case "emergency":
                self.state_emergency()

            case _:
                self.get_logger().error(f"Unknown state: {self.FSM_state}")
                self.transition_to_state(new_state="emergency")

    def calculate_angle(self, v1, v2, axis=None):
        """Angle between v1 and v2 in radians.

        Without `axis`: unsigned angle in [0, pi].
        With `axis` (normal of the plane of rotation): signed angle in (-pi, pi],
        positive for a right-handed rotation from v1 to v2 about `axis`.
        """
        v1 = np.asarray(v1, dtype=float)
        v2 = np.asarray(v2, dtype=float)
        cross = np.cross(v1, v2)
        dot = np.dot(v1, v2)

        if axis is None:
            return np.arctan2(np.linalg.norm(cross), dot)

        axis = np.asarray(axis, dtype=float)
        axis = axis / np.linalg.norm(axis)
        return np.arctan2(np.dot(cross, axis), dot)

def main():
    run_mission(PinchGraspMission)


if __name__ == '__main__':
    main()
