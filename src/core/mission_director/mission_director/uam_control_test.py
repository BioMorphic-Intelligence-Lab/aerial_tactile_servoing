import rclpy
import numpy as np

from mission_director.uam_state_machine import UAMStateMachine

class MissionDirector(UAMStateMachine):
    def __init__(self):
        super().__init__('mission_director')
        self.get_logger().info("MissionDirector node uam_control_test initialized.")

        # Timer -- always last
        self.counter = 0
        self.timer = self.create_timer(self.timer_period, self.execute)

    def execute(self):
        match self.FSM_state:
            case "entrypoint":
                self.state_entrypoint(next_state="wait")

            case "wait":
                self.state_do_nothing(next_state="move_arms")

            case "move_arms":
                q_right = [1.0, 0.1, 1.3]
                self.state_move_arms(q=q_right, next_state="arms_takeoff_position")

            case "arms_takeoff_position":
                q_right = [1.57, 0.0, -1.57]
                self.state_move_arms(q=q_right, next_state="move_arms")
            
            case "sim_arm_vehicle":
                self.state_wait_for_arming(next_state="takeoff")

            case "takeoff":
                self.state_takeoff(target_altitude = 1.5, next_state="hover")

            case "hover":
                self.state_hover(duration_sec=5, next_state="hover_move_arms")

            case "hover_move_arms":
                q_right = [0.8, 0.0, 1.7] # put some position here
                q_right.append(q_right[0]-np.pi)
                self.state_move_arms(q=q_right, next_state="move_uam_to_position")

            case "move_uam_to_position":
                self.state_move_uam_to_position(target_position=[2.0, 0.0, -1.5], next_state="land") # TODO vehicle should move in the ground

            case "land":
                self.state_land(next_state="disarm")

            # --- Do not remove these states ---
            case "emergency":
                self.state_emergency()

            case _:
                self.get_logger().error(f"Unknown state: {self.FSM_state}")
                self.transition_to_state(new_state="emergency")
def main():
    rclpy.init(args=None)

    md = MissionDirector()

    rclpy.spin(md)
    md.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()