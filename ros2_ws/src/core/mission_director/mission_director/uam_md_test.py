import datetime
import rclpy

from mission_director.uam_state_machine import UAMStateMachine

class MissionDirectorTest(UAMStateMachine):
    def __init__(self):
        super().__init__('mission_director_test_node')
        self.get_logger().info("MissionDirectorTest node initialized.")


        # Timer -- always last
        self.counter = 0
        self.timer = self.create_timer(self.timer_period, self.execute)


    def execute(self):
        match self.FSM_state:
            case "entrypoint":
                self.state_entrypoint(next_state="dummy")

            case "dummy":
                self.handle_state(state_number=1)
                if self.first_state_loop:
                    self.get_logger().info("In state: dummy")
                    self.first_state_loop = False

                if self.counter%10 ==0:
                    self.get_logger().info("Dummy state executing...")
                
                if (datetime.datetime.now()-self.state_start_time).seconds > 5.:
                    self.transition_to_state("entrypoint")
                

def main():
    rclpy.init(args=None)

    md = MissionDirectorTest()

    rclpy.spin(md)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    md.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()