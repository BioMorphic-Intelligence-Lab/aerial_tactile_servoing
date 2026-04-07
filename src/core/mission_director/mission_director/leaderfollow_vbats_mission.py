from platform import node

from skimage import future

import rclpy
import numpy as np
import datetime
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Int8
from geometry_msgs.msg import TwistStamped, Vector3Stamped
from px4_msgs.msg import TrajectorySetpoint
from sensor_msgs.msg import JointState

from std_srvs.srv import SetBool
from mission_director.uam_state_machine import UAMStateMachine

class MissionDirector(UAMStateMachine):
    def __init__(self):
        super().__init__('mission_director')
        self.get_logger().info("MissionDirector node ats_mission initialized.")

        # Tactile servoing parameters
        self.ts_no_contact_counter = 0
        self.ts_no_contact_max_seconds = 10  # Max cycles without contact before aborting tactile servoing
        self.ts_no_contact_max_cycles = int(self.ts_no_contact_max_seconds * self.frequency)

        # Get tactile servoing time from parameters
        self.declare_parameter('im.tactile_servoing_time', 65.0) # seconds
        self.tactile_servoing_time = self.get_parameter('im.tactile_servoing_time').get_parameter_value().double_value

        # Tactip interfaces
        self.sub_tactip = self.create_subscription(TwistStamped, '/tactip/pose', self.tactip_callback, 10)
        self.sub_tactip_contact = self.create_subscription(Int8, '/tactip/contact', self.tactip_contact_callback, 10)
        self.tactip_data = None
        self.contact = False
        self.tactip_update_time = None
        self.last_tactip_update_time = None

        # Controller interfaces
        self.sub_controller = self.create_subscription(TrajectorySetpoint, '/controller/out/trajectory_setpoint', self.controller_callback, 10)
        self.sub_controller_servo = self.create_subscription(JointState, '/controller/out/servo_state', self.controller_servo_callback, 10)

        # Tactile servoing specific variables
        self.vehicle_trajectory_setpoint = TrajectorySetpoint()
        self.servo_reference = JointState()

        # Service clients
        # Set new SSIM reference on Tactip client
        self.cli_set_ssim_ref = self.create_client(SetBool, 'set_ssim_ref')
        self.got_ref = False
        while not self.cli_set_ssim_ref.wait_for_service(timeout_sec=1.0) and not self.sim:
            self.get_logger().info('Waiting for set_ssim_ref service to become available...')

        # Calibrate thrust coefficient client
        self.cli_calibrate_thrust = self.create_client(SetBool, 'calibrate_thrust_coefficient')
        self.calibrate_thrust_coeff = False
        while not self.cli_calibrate_thrust.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for calibrate_thrust_coefficient service to become available...')

        # Timer -- always last
        self.counter = 0
        self.timer = self.create_timer(self.timer_period, self.execute)


    def execute(self):
        match self.FSM_state:
            case "entrypoint":
                self.state_entrypoint(next_state="arms_takeoff_position")

            case "arms_takeoff_position":
                q_right = [np.pi/3, 0.0, np.pi/6] # put some position here
                self.state_move_arms(q_des=q_right, next_state="wait_for_arm_offboard")

            case "wait_for_arm_offboard":
                if not self.got_ref and not self.state_start_time:
                    self.set_new_ssim_ref(True) # Set the current tactip image as the reference for SSIM-based contact detection
                self.state_wait_for_arming(next_state="takeoff")

            case "takeoff":
                self.state_takeoff(target_altitude=1.5, next_state="hover")

            case "hover":
                self.state_hover(duration_sec=3., next_state="calibrate_thrust_coefficient")
            
            case "calibrate_thrust_coefficient":
                self.state_hover(duration_sec=1., next_state="pre_contact_arm_position") # Hover for a while to collect data for thrust coefficient calibration, then transition to pre-contact arm position
                if not self.calibrate_thrust_coeff:
                    self.calibrate_thrust_coefficient(True)
                    self.calibrate_thrust_coeff = True

            case "pre_contact_arm_position":
                q_right = [3*np.pi/4, 0.0, np.pi/4] # put some position here
                self.state_move_arms(q_des=q_right, next_state="approach")

            case "approach": # Better way is to command a negative z velocity on the end-effector and run it through the inverse kinematics
                self.state_approach_wall_position(approach_speed=[0.0, 0.0, 0.05], transition=False, next_state="tactile_servoing") # TODO set transition to self.contact when we have a reliable contact detection, currently just transition after a fixed time to ensure we get into the tactile servoing state
            
            case "tactile_servoing":
                self.handle_state(state_number=30)
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
                    self.vehicle_trajectory_setpoint.yawspeed,
                    acceleration = self.vehicle_trajectory_setpoint.acceleration, # Pass through feedforward acceleration from the controller if available
                )

                self.publish_servo_velocity_references(self.servo_reference.velocity)
                self.get_logger().info(f'[22] Contact depth: {self.tactip_data.twist.linear.z:.2f} mm, FF acceleration: x {self.vehicle_trajectory_setpoint.acceleration[0]:.2f} y {self.vehicle_trajectory_setpoint.acceleration[1]:.2f} z {self.vehicle_trajectory_setpoint.acceleration[2]:.2f}', throttle_duration_sec=1)
                if self.ts_no_contact_counter > self.ts_no_contact_max_cycles: # If no contact for x cycles, go back to approach
                    self.get_logger().info('Lost contact, returning to approach state.')
                    self.ts_no_contact_counter = 0
                    self.transition_to_state('pre_contact_uam_position')
                elif (datetime.datetime.now() - self.state_start_time).seconds > self.tactile_servoing_time or self.input_state==1:
                    self.transition_to_state('disengage')
                elif not self.offboard and self.fcu_on:
                    self.transition_to_state('emergency')
            
            case "disengage":
                self.state_disengage_wall_velocity(disengage_speed=0.15, duration_sec=5.0, next_state="land_arms_position")

            case "land_arms_position":
                q_right = [1.57, 0.0, -1.57] # put some position here
                self.state_move_arms(q_des=q_right, next_state="land")

            case "land":
                self.state_land(next_state="done")

            case "done":
                self.get_logger().info("Mission completed successfully.", throttle_duration_sec=5.)

            # --- Do not remove these states ---
            case "emergency":
                self.state_emergency()

            case _:
                self.get_logger().error(f"Unknown state: {self.FSM_state}")
                self.transition_to_state(new_state="emergency")

    # ------------------------------------------------------------------------------------
    # Tactile servoing specific callbacks
    # ------------------------------------------------------------------------------------
    def tactip_callback(self, msg):
        # if self.last_tactip_update_time is not None: # IF we have a previous timestamp, compute the update time
        #     self.tactip_update_time = msg.header.stamp - self.last_tactip_update_time
        self.tactip_data = msg
        # self.last_tactip_update_time = msg.header.stamp # Update the last update time
    
    def tactip_contact_callback(self, msg):
        self.contact = bool(msg.data)

    def controller_callback(self, msg):
        self.vehicle_trajectory_setpoint = msg

    def controller_servo_callback(self, msg):
        self.servo_reference = msg

    def external_torque_callback(self, msg):
        self.external_torque = msg

    def set_new_ssim_ref(self, value: bool):
        req = SetBool.Request()
        req.data = value

        self.future = self.cli_set_ssim_ref.call_async(req)
        self.got_ref = True
    
    def calibrate_thrust_coefficient(self, value: bool):
        req = SetBool.Request()
        req.data = value
        self.get_logger().info('Calibrating thrust coefficient...')
        self.future = self.cli_calibrate_thrust.call_async(req)
        self.calibrate_thrust_coeff = True

def main():
    rclpy.init(args=None)

    md = MissionDirector()

    rclpy.spin(md)
    md.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()