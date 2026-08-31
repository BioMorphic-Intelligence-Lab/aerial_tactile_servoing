import rclpy

from mission_director.base_classes.uam_state_machine import UAMStateMachine


class WaypointsMission(UAMStateMachine):
    """Bare flight test: take off, fly a list of waypoints, land.

    Deliberately does NOT use the tactip/controller/servo stack, so it subclasses
    UAMStateMachine directly rather than TactileMissionDirector (whose __init__
    blocks on the set_ssim_ref service). The matching launch file
    (ats_launch/launch/testing/waypoints.launch.py) therefore starts *only* this
    node plus an optional rosbag -- no drivers, no manipulator, no perception.

    Waypoints come from the per-mission YAML as a flat [x, y, z, yaw, ...] array
    (groups of four). x and y are offsets from the take-off (home) position in the
    PX4 local frame [m], z is altitude above home [m] (positive up), yaw is heading
    [rad]. The list is flown in order, then the vehicle lands where it finishes.
    """

    def __init__(self):
        super().__init__('mission_director')

        # --- Mission-graph constants (from the per-mission YAML) ---
        self.declare_parameter('im.takeoff_altitude', 1.0)  # [m]
        self.takeoff_altitude = self.get_parameter('im.takeoff_altitude').get_parameter_value().double_value

        # Flat [x, y, z, yaw] quadruples; x/y relative to home, z altitude (up), yaw [rad].
        self.declare_parameter('im.waypoints', [
            1.0, 0.0, 1.0, 0.0,
            1.0, 1.0, 1.0, 0.0,
            0.0, 1.0, 1.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
        ])
        flat = list(self.get_parameter('im.waypoints').get_parameter_value().double_array_value)
        if len(flat) % 4 != 0:
            self.get_logger().error(
                f'im.waypoints length {len(flat)} is not a multiple of 4 (x, y, z, yaw). Ignoring extras.')
        self.waypoints = [flat[i:i + 4] for i in range(0, len(flat) - len(flat) % 4, 4)]

        self.get_logger().info(f'WaypointsMission initialized with {len(self.waypoints)} waypoint(s).')

        # UAMStateMachine does not create the FSM timer itself (only the tactile
        # base class does), so this mission owns it.
        self.counter = 0
        self.timer = self.create_timer(self.timer_period, self.execute)

    def execute(self):
        # Waypoint states are generated dynamically: waypoint_0, waypoint_1, ...
        if self.FSM_state.startswith("waypoint_"):
            idx = int(self.FSM_state.split("_", 1)[1])
            wp = self.waypoints[idx]
            target = [
                self.home_position[0] + wp[0],
                self.home_position[1] + wp[1],
                -abs(wp[2]),  # PX4 local frame: z is negative up
                wp[3],
            ]
            next_state = f"waypoint_{idx + 1}" if idx + 1 < len(self.waypoints) else "land"
            self.state_move_uam_to_position(target, next_state=next_state)
            return

        match self.FSM_state:
            case "entrypoint":
                # Position-only entrypoint: this mission has no manipulator, so
                # (unlike the shared state_entrypoint) it must not wait for servo state.
                self.handle_state(state_number=0)
                self.home_position[0] = self.vehicle_local_position.x
                self.home_position[1] = self.vehicle_local_position.y
                self.home_position[2] = self.vehicle_local_position.z
                self.home_position[3] = self.vehicle_local_position.heading

                if self.home_position[0] == 0.0 and self.home_position[1] == 0.0:
                    self.get_logger().info(
                        f'[ENTRYPOINT] Waiting for position fix! '
                        f'x {self.vehicle_local_position.x:.4f}, y {self.vehicle_local_position.y:.4f}',
                        throttle_duration_sec=1)

                if self.home_position[0] != 0.0 and self.home_position[1] != 0.0:
                    self.get_logger().info(
                        f'Got position fix! x: {self.home_position[0]:.3f} m, '
                        f'y: {self.home_position[1]:.3f} m, heading: {self.home_position[3]:.3f} rad')
                    self.transition_to_state("wait_for_arming")
                elif self.input_state == 1:
                    self.get_logger().info('Bypassing position fix wait')
                    self.transition_to_state("wait_for_arming")

            case "wait_for_arming":
                self.state_wait_for_arming(next_state="takeoff")

            case "takeoff":
                first = "waypoint_0" if self.waypoints else "land"
                self.state_takeoff(target_altitude=self.takeoff_altitude, next_state=first)

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
    rclpy.init(args=None)
    node = WaypointsMission()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
