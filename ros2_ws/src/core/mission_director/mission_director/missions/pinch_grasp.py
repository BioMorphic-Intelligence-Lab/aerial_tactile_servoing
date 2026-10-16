from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import PoseStamped

from mission_director.base_classes.tactile_mission_director import TactileMissionDirector, run_mission


class PinchGraspMission(TactileMissionDirector):
    """Tactile pinch grasping mission.

    Same consolidated pattern as vbats: only the mission-specific FSM lives here.
    It adds a mocap subscription to the door pose and disengages once the door
    has swung past a configurable open angle. All tuning is in
    config/missions/pinch_grasp.yaml.
    """

    def __init__(self):
        super().__init__('mission_director')

        # --- Mission-graph constants (from the per-mission YAML) ---
        self.declare_parameter('im.takeoff_altitude', 1.5)  # [m]
        self.takeoff_altitude = self.get_parameter('im.takeoff_altitude').get_parameter_value().double_value

        self.declare_parameter('im.pre_contact_arm_position', [1.0472, 0.0, -0.5236])  # [rad] q1,q2,q3
        self.pre_contact_arm_position = list(self.get_parameter('im.pre_contact_arm_position').get_parameter_value().double_array_value)

        self.declare_parameter('im.land_arm_position', [1.57, 0.0, -1.57])  # [rad]
        self.land_arm_position = list(self.get_parameter('im.land_arm_position').get_parameter_value().double_array_value)


        self.get_logger().info('Tactile Pinch Grasping mission initialized.')

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
                self.state_hover(duration_sec=10., next_state="pre_contact_arm_position")

            case "pre_contact_arm_position":
                self.state_move_arms(q_des=self.pre_contact_arm_position, next_state="land_arms_position")

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
    run_mission(PinchGraspMission)


if __name__ == '__main__':
    main()
