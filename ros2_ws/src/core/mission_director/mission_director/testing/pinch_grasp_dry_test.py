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
        self.kp_shear = 1.0
        self.kp_force = 1.0
        self.squeeze_force_ref = 3.0

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
                self.state_move_arms_joint_position(q_des=self.pre_grasp_arm_position, next_state="approach_grasp")

            case "approach_grasp":
                # self.get_logger().info(f"{self.tactip_contact_}")
                next_state="control_grasp"
                if abs(self.arm_task_positions_[0][1]) < 0.05 or abs(self.arm_task_positions_[1][1]) < 0.05:
                    self.get_logger().info("Transition by approach of center")
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
                self.state_move_arms_task_velocity(pdot_des=[0.0, -0.03, 0.0, 0.0, 0.03, 0.0], transition=transition, next_state=next_state, state_number=25)

            case "approach_grasp_0":
                if all(self.tactip_contact_):
                    transition = True
                else:
                    transition = False

                self.state_move_arms_task_velocity(pdot_des=[0.0, 0.0, 0.0, 0.0, 0.03, 0.0], transition=transition, next_state="control_grasp", state_number=26)

            case "approach_grasp_1":
                if all(self.tactip_contact_):
                    transition = True
                else:
                    transition = False
                self.state_move_arms_task_velocity(pdot_des=[0.0, -0.03, 0.0, 0.0, 0.0, 0.0], transition=transition, next_state="control_grasp", state_number=27)

            case "control_grasp":
                # Step 1: calculate 
                v_c_cmd = np.zeros(self.n_arms*3)
                R_CS_block = np.zeros((self.n_arms*3, self.n_arms*3))
                R_BS_block = np.zeros((self.n_arms*3, self.n_arms*3))
                for n in range(self.n_arms):
                    v_c_cmd[n*3 : n*3+3] = np.array([
                        self.kp_shear * self.tactip_pose_[n].angular.y,
                        -self.kp_shear * self.tactip_pose_[n].angular.x,
                        -self.kp_force * (self.squeeze_force_ref - self.tactip_wrench_[n].force.z)
                    ])
                    R_CS_block[n*3:n*3+3, n*3:n*3+3] = self.evaluate_R_CS(self.tactip_pose_[n].angular.x, self.tactip_pose_[n].angular.y)
                    R_BS_block[n*3:n*3+3, n*3:n*3+3] = self.evaluate_R_BS(*self.servo_state.position[n*3:n*3+3])
                # Invert rotation
                R_SC_block = np.linalg.inv(R_CS_block)
                R_BC_block = R_BS_block@R_SC_block
                v_b_cmd = R_BC_block@v_c_cmd # Rotate from contact frame into sensor frame

                transition = False
                if (datetime.datetime.now()-self.state_start_time).seconds > 10.:
                    transition = True

                self.state_move_arms_task_velocity(pdot_des=v_b_cmd, transition=transition, next_state="release", state_number=29)

            case "release":
                self.state_move_arms_joint_position(q_des=self.land_arm_position, next_state="done")

            case "done":
                self.get_logger().info("Mission completed successfully.", throttle_duration_sec=5.)

            # --- Do not remove these states ---
            case "emergency":
                self.state_emergency()

            case _:
                self.get_logger().error(f"Unknown state: {self.FSM_state}")
                self.transition_to_state(new_state="emergency")

def main():
    run_mission(PinchGraspMission)


if __name__ == '__main__':
    main()
