import rclpy
import numpy as np
import datetime
from std_msgs.msg import Int8
from geometry_msgs.msg import TwistStamped
from px4_msgs.msg import TrajectorySetpoint
from sensor_msgs.msg import JointState

from mission_director.uam_state_machine import UAMStateMachine

class MissionDirector(UAMStateMachine):
    def __init__(self):
        super().__init__('mission_director')
        self.get_logger().info("MissionDirector node ats_mission initialized.")

        # Tactile servoing parameters
        self.ts_no_contact_counter = 0
        self.ts_no_contact_max_seconds = 10  # Max cycles without contact before aborting tactile servoing
        self.ts_no_contact_max_cycles = int(self.ts_no_contact_max_seconds * self.frequency)

        # Tactip interfaces
        self.sub_tactip = self.create_subscription(TwistStamped, '/tactip/pose', self.tactip_callback, 10)
        self.sub_tactip_contact = self.create_subscription(Int8, '/tactip/contact', self.tactip_contact_callback, 10)
        self.tactip_data = TwistStamped()
        self.contact = False

        # Controller interfaces
        self.sub_controller = self.create_subscription(TrajectorySetpoint, '/controller/out/trajectory_setpoint', self.controller_callback, 10)
        self.sub_controller_servo = self.create_subscription(JointState, '/controller/out/servo_state', self.controller_servo_callback, 10)

        # Tactile servoing specific variables
        self.vehicle_trajectory_setpoint = TrajectorySetpoint()
        self.servo_reference = JointState()

        # Timer -- always last
        self.counter = 0
        self.timer = self.create_timer(self.timer_period, self.execute)


    def execute(self):
        match self.FSM_state:
            case "entrypoint":
                self.state_entrypoint(next_state="arms_takeoff_position")

            case "arms_takeoff_position":
                q_right = [np.pi/2, 0.0, -np.pi/2] # put some position here
                self.state_move_arms(q=q_right, mode='velocity', next_state="wait_for_arm_offboard")

            case "wait_for_arm_offboard":
                self.state_wait_for_arming(next_state="takeoff")

            case "takeoff":
                self.state_takeoff(target_altitude=1.7, next_state="hover")

            case "hover":
                self.state_hover(duration_sec=3, next_state="pre_contact_uam_position")

            case "pre_contact_uam_position":
                self.state_move_uam_to_position([0.0, 0.7, -1.7, 0.0], next_state="pre_contact_arm_position")

            case "pre_contact_arm_position":
                q_right = [np.pi/3, 0.0, np.pi/6] # put some position here
                self.state_move_arms(q=q_right, mode='velocity', next_state="approach")

            case "approach": # Better way is to command a negative z velocity on the end-effector and run it through the inverse kinematics
                self.handle_state(state_number=21)
                approach_speed = 0.05  # m/s

                # First state loop
                if self.first_state_loop:
                    self.get_logger().info(f'[21] Approaching contact surface at {approach_speed} m/s')
                    self.first_state_loop = False
                
                # Update position setpoint
                self.get_logger().info(f'Approaching... : {self.vehicle_odometry.velocity[1]:.3f}/{approach_speed} m/s, {self.vehicle_odometry.position[1]:.2f}/{self.position_clip} m', throttle_duration_sec=1)
                self.publish_trajectory_velocity_setpoint(
                    0.0,
                    approach_speed,
                    0.0,
                    0.0
                )

                # State transition
                if not self.offboard and self.fcu_on:
                    self.transition_to_state('emergency')
                elif self.contact or self.input_state==1:
                    self.transition_to_state(new_state="tactile_servoing")
            
            case "tactile_servoing":
                self.handle_state(state_number=22)
                if not self.contact:
                    self.ts_no_contact_counter += 1
                    self.get_logger().info(f'No contact detected. Counter: {self.ts_no_contact_counter}/{self.ts_no_contact_max_cycles}', throttle_duration_sec=1)
                else:
                    self.ts_no_contact_counter = 0

                # First state loop
                if self.first_state_loop:
                    self.get_logger().info(f'[22] Tactile servoing initiated.')
                    self.first_state_loop = False
                
                # Pass through controller setpoints
                self.publish_trajectory_velocity_setpoint(
                    self.vehicle_trajectory_setpoint.velocity[0],
                    self.vehicle_trajectory_setpoint.velocity[1],
                    self.vehicle_trajectory_setpoint.velocity[2],
                    self.vehicle_trajectory_setpoint.yawspeed
                )

                self.publish_servo_velocity_references(self.servo_reference.velocity)
                self.get_logger().info(f'Contact depth: {self.tactip_data.twist.linear.z} m', throttle_duration_sec=1)
                if self.ts_no_contact_counter > self.ts_no_contact_max_cycles: # If no contact for 10 cycles, go back to approach
                    self.get_logger().info('Lost contact, returning to approach state.')
                    self.ts_no_contact_counter = 0
                    self.transition_to_state('pre_contact_uam_position')
                elif (datetime.datetime.now() - self.state_start_time).seconds > 300. or self.input_state==1:
                    self.transition_to_state('land_position')
                elif not self.offboard and self.fcu_on:
                    self.transition_to_state('emergency')
            
            case "land_position":
                self.state_move_uam_to_position([0.0, 0.0, -1.0, 0.0], next_state="land_arms_position")

            case "land_arms_position":
                q_right = [1.57, 0.0, -1.57] # put some position here
                self.state_move_arms(q=q_right, mode='velocity', next_state="land")

            case "land":
                self.state_land(next_state="done")

            case "done":
                self.get_logger().info("Mission completed successfully.", throttle_duration_sec=5.)

            # --- Do not remove these states ---
            case "emergency":
                self.state_emergency(mode='velocity')

            case _:
                self.get_logger().error(f"Unknown state: {self.FSM_state}")
                self.transition_to_state(new_state="emergency")

    # ------------------------------------------------------------------------------------
    # Tactile servoing specific callbacks
    # ------------------------------------------------------------------------------------
    def tactip_callback(self, msg):
        self.tactip_data = msg
    
    def tactip_contact_callback(self, msg):
        self.contact = bool(msg.data)

    def controller_callback(self, msg):
        self.vehicle_trajectory_setpoint = msg

    def controller_servo_callback(self, msg):
        self.servo_reference = msg

def main():
    rclpy.init(args=None)

    md = MissionDirector()

    rclpy.spin(md)
    md.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()