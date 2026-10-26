import datetime

from mission_director.base_classes.tactile_mission_director import TactileMissionDirector, run_mission


class PinchGraspMission(TactileMissionDirector):
    """Tactile pinch grasping dry test mission.


    """

    def __init__(self):
        super().__init__('mission_director')

        # --- Mission-graph constants (from the per-mission YAML) ---
        self.declare_parameter('im.takeoff_altitude', 1.5)  # [m]
        self.declare_parameter('im.pre_contact_arm_position', [1.0472, 0.0, 0.5236, -1.0472, 0.0, -0.5236])  # [rad] q1,q2,q3
        self.declare_parameter('im.land_arm_position', [1.57, 0.0, -1.57, -1.57, 0.0, 1.57])  # [rad]

        self.pre_contact_arm_position = list(self.get_parameter('im.pre_contact_arm_position').get_parameter_value().double_array_value)
        self.land_arm_position = list(self.get_parameter('im.land_arm_position').get_parameter_value().double_array_value)

        self.get_logger().info('Tactile Pinch Grasping mission initialized.')

    def execute(self):
        match self.FSM_state:
            case "entrypoint":
                self.state_entrypoint(next_state="joint_position")

            case "joint_position":
                self.state_move_arms_joint_position(q_des=self.pre_contact_arm_position, next_state="wait_1")
            case "wait_1":
                self.state_do_nothing(next_state="task_position")

            case "task_position":
                self.state_move_arms_task_position(p_des=[0.0, 0.5, -0.2, 0.0, -0.5, -0.2], next_state='wait_2')
            case "wait_2":
                self.state_do_nothing(next_state="task_velocity")
            
            case "task_velocity":
                if (datetime.datetime.now()-self.state_start_time).seconds > 3.:
                    transition = True
                else:
                    transition = False
                self.state_move_arms_task_velocity(pdot_des=[0.0, 0.03, 0.0, 0.0, -0.03, 0.0], transition=transition, next_state='wait_3')
            case "wait_3":
                self.state_do_nothing(next_state="joint_velocity")                

            case "joint_velocity":
                if (datetime.datetime.now()-self.state_start_time).seconds > 5.:
                    transition = True
                else:
                    transition = False
                self.state_move_arms_joint_velocity(qdot_des=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1], transition=transition, next_state='wait_4')
            case "wait_4":
                self.state_do_nothing(next_state="done")

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
