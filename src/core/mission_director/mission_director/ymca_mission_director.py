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
                self.state_entrypoint(next_state="move_arms")

            case "move_arms":
                q = [1.0, 0.1, 1.3,
                     -1.0, -0.1, -1.3]
                self.state_move_arms(q=q, next_state="arms_takeoff_position")

            case "arms_takeoff_position":
                q = [1.57, 0.0, -1.57,
                     -1.57, 0.0, 1.57]
                self.state_move_arms(q=q, next_state="sim_arm_vehicle")
            
            case "sim_arm_vehicle":
                self.state_wait_for_arming(next_state="takeoff")

            case "takeoff":
                self.state_takeoff(target_altitude = 1.5, next_state="hover")

            case "hover":
                self.state_hover(duration_sec=3, next_state="Y")

            case "Y":
                q = [1.0, 0.0, -0.2,
                    -1.0, 0.0, 0.2]
                self.state_move_arms(q=q, next_state="hold_Y")

            case "hold_Y":
                self.state_hover(duration_sec=3, next_state="M")
            
            case "M": # not good
                q = [0.9, 0.0, 1.8,
                    -0.9, 0.0, -1.8]
                self.state_move_arms(q=q, next_state="hold_M")

            case "hold_M":
                self.state_hover(duration_sec=1, next_state="C")
            
            case "C":# not good
                q = [0.7, 0.0, 1.1, # 0.7, 0.0, -0.8
                    -3.7, 0.0, -1.1]
                self.state_move_arms(q=q, next_state="hold_C")

            case "hold_C":
                self.state_hover(duration_sec=0.2, next_state="A")  
            
            case "A":
                q = [0.8, 0.0, -1.6,
                    -0.8, 0.0, 1.6]
                self.state_move_arms(q=q, next_state="hold_A")
            
            case "hold_A":
                self.state_hover(duration_sec=5, next_state="wave_left")

            case "wave_left":
                q = [1.0, 0.0, -1.5,
                    -1.0, 0.0, -1.5]
                self.state_move_arms(q=q, next_state="hold_wave_left")

            case "hold_wave_left":
                self.state_hover(duration_sec=5, next_state="wave_right")
            
            case "wave_right":
                q = [1.0, 0.0, 1.5,
                    -1.0, 0.0, 1.5]
                self.state_move_arms(q=q, next_state="hold_wave_right")
            
            case "hold_wave_right":
                self.state_hover(duration_sec=5, next_state="arms_landing_position")

            case "arms_landing_position":
                q = [1.57, 0.0, -1.57,
                     -1.57, 0.0, 1.57]
                self.state_move_arms(q=q, next_state="land")

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