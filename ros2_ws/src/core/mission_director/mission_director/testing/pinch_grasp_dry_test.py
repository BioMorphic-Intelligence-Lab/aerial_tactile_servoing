import datetime

from mission_director.base_classes.tactile_mission_director import TactileMissionDirector, run_mission


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
                if abs(self.arm_task_positions_[0][1]) < 0.1 or abs(self.arm_task_positions_[1][1]) < 0.1:
                    self.get_logger().info("Transition by approach of center")
                    transition = True
                elif all(self.tactip_contact_):
                    self.get_logger().info("Transition by contact")
                    transition = True
                else:
                    transition = False
                self.state_move_arms_task_velocity(pdot_des=[0.0, -0.05, 0.0, 0.0, 0.05, 0.0], transition=transition, next_state="control_grasp")

            case "control_grasp":
                self.state_hover(duration_sec=20., next_state="release")

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
