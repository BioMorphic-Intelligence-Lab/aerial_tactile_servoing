import datetime

import numpy as np

from mission_director.tactile_mission_director import TactileMissionDirector, run_mission


class VbatsMission(TactileMissionDirector):
    """Baseline velocity-based tactile servoing mission.

    Proof-of-concept for the consolidated TactileMissionDirector pattern: this
    file is *only* the mission-specific finite state machine. All shared wiring
    lives in the base class, and every tuned constant comes from the per-mission
    YAML (config/missions/vbats.yaml in ats_launch) rather than being hard-coded.
    """

    def __init__(self):
        super().__init__('mission_director')

        # --- Mission-graph constants (from the per-mission YAML) ---
        self.declare_parameter('im.takeoff_altitude', 1.0)  # [m]
        self.takeoff_altitude = self.get_parameter('im.takeoff_altitude').get_parameter_value().double_value

        self.declare_parameter('im.pre_contact_arm_position', [1.0472, 0.0, 0.5236])  # [rad] q1,q2,q3
        self.pre_contact_arm_position = list(self.get_parameter('im.pre_contact_arm_position').get_parameter_value().double_array_value)

        self.declare_parameter('im.land_arm_position', [1.57, 0.0, -1.57])  # [rad]
        self.land_arm_position = list(self.get_parameter('im.land_arm_position').get_parameter_value().double_array_value)

        self.declare_parameter('im.approach_speed', [0.0, 0.05, 0.0])  # [m/s] end-effector approach velocity
        self.approach_speed = list(self.get_parameter('im.approach_speed').get_parameter_value().double_array_value)

        self.declare_parameter('im.approach_heading_deg', 15.0)  # [deg]
        self.approach_heading_deg = self.get_parameter('im.approach_heading_deg').get_parameter_value().double_value

        self.declare_parameter('im.disengage_speed', 0.15)  # [m/s]
        self.disengage_speed = self.get_parameter('im.disengage_speed').get_parameter_value().double_value

        self.declare_parameter('im.ee_x_abort_threshold', -1.0)  # [m] abort servoing past this ee x
        self.ee_x_abort_threshold = self.get_parameter('im.ee_x_abort_threshold').get_parameter_value().double_value

        self.get_logger().info('VbatsMission initialized.')

    def execute(self):
        match self.FSM_state:
            case "entrypoint":
                self.state_entrypoint(next_state="arms_takeoff_position")

            case "arms_takeoff_position":
                self.state_move_arms(q_des=self.pre_contact_arm_position, next_state="wait_for_arm_offboard")

            case "wait_for_arm_offboard":
                if not self.got_ref and not self.sim:
                    self.set_new_ssim_ref(True)  # set the current tactip image as the SSIM reference
                self.state_wait_for_arming(next_state="takeoff")

            case "takeoff":
                self.state_takeoff(target_altitude=self.takeoff_altitude, next_state="hover")

            case "hover":
                self.state_hover(duration_sec=1, next_state="pre_contact_arm_position")

            case "pre_contact_arm_position":
                self.state_move_arms(q_des=self.pre_contact_arm_position, next_state="approach")

            case "approach":
                self.state_approach_wall_position(
                    approach_speed=self.approach_speed,
                    approach_heading=np.deg2rad(self.approach_heading_deg),
                    transition=self.contact,
                    next_state="tactile_servoing")

            case "tactile_servoing":
                self.handle_state(state_number=30)
                self.publish_tactile_servoing_outputs()

                depth = self.tactip_data.twist.linear.z if self.tactip_data else float('nan')
                elapsed = (datetime.datetime.now() - self.state_start_time).seconds
                self.get_logger().info(
                    f'Contact depth: {depth} mm, time: {elapsed:.1f}/{self.tactile_servoing_time}',
                    throttle_duration_sec=1)

                transform = self.get_transform()
                position_ee = transform.transform.translation if transform else None
                if position_ee is not None:
                    self.get_logger().info(
                        f'End-effector position: x={position_ee.x:.3f}, y={position_ee.y:.3f}, z={position_ee.z:.3f}',
                        throttle_duration_sec=1)

                if self.lost_contact():
                    self.get_logger().info('Lost contact, returning to approach.')
                    self.ts_no_contact_counter = 0
                    self.transition_to_state('pre_contact_arm_position')
                elif self.servoing_time_elapsed() or self.input_state == 1:
                    self.transition_to_state('disengage')
                elif position_ee is not None and position_ee.x < self.ee_x_abort_threshold:
                    self.transition_to_state('disengage')
                elif not self.offboard and self.fcu_on:
                    self.transition_to_state('emergency')

            case "disengage":
                self.state_disengage_wall_velocity(disengage_speed=self.disengage_speed, duration_sec=5.0, next_state="land_arms_position")

            case "land_arms_position":
                self.state_move_arms(q_des=self.land_arm_position, next_state="land")

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


def main():
    run_mission(VbatsMission)


if __name__ == '__main__':
    main()
