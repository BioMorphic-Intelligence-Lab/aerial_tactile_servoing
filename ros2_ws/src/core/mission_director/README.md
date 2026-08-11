# mission_director

Flight state machine that orchestrates an aerial tactile servoing mission and holds all
authority over the aircraft. Each mission variant is a node that subclasses `UAMStateMachine`
(`uam_state_machine.py`) and implements an `execute()` method matching on `self.FSM_state`.

During the `tactile_servoing` state the mission director **relays** the controller's setpoints
(`/controller/out/*`) to the flight controller and servos; in every other state it issues its own
takeoff/hover/land setpoints. See [docs/06-running-a-mission.md](../../../../docs/06-running-a-mission.md)
for the full state walkthrough.

## Nodes

| Executable | Source | Mission |
|------------|--------|---------|
| `vbats_mission` | `vbats_mission.py` | Baseline velocity-based tactile servoing |
| `door_vbats_mission` | `door_vbats_mission.py` | Door opening (uses `/mocap/door/pose`) |
| `slide_vbats_mission` | `slide_vbats_mission.py` | Sliding along a surface |
| `wallfollow_vbats_mission` | `wallfollow_vbats_mission.py` | Wall following (uses `/mocap/leader/pose`) |
| `leaderfollow_vbats_mission` | `leaderfollow_vbats_mission.py` | Leader following |
| `ats_mission` | `ats_mission.py` | Earlier ATS mission |
| `uam_control_test` | `uam_control_test.py` | Control bring-up test |

`old_implementation/` holds superseded mission directors kept for reference.

## Run

Launched as part of a mission (parameters are set in the launch file), e.g.:

```bash
ros2 launch ats_bringup vbats_mission.launch.py
```

## Parameters

Base (`UAMStateMachine`), namespaced `sm.`:

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `sm.frequency` | 10.0 | FSM loop rate (Hz) |
| `sm.position_clip` | 0.0 | Position/velocity bound (m); 0 disables |
| `sm.fcu_on` | true | Expect a real flight controller (forced false when `sm.sim`) |
| `sm.sim` | false | Simulation mode |
| `sm.manipulator_mode` | position | `position` or `velocity` joint commands |

Mission (`vbats_mission`), namespaced `im.`:

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `im.tactile_servoing_time` | 65.0 | Max seconds in the tactile-servoing state |

<!-- TODO: document parameters for the other mission variants. -->

## Topics

Base class (`UAMStateMachine`):

| Direction | Topic | Type |
|-----------|-------|------|
| pub | `/md/state` | `std_msgs/Int32` |
| pub | `/fmu/in/trajectory_setpoint` | `px4_msgs/TrajectorySetpoint` |
| pub | `/fmu/in/offboard_control_mode` | `px4_msgs/OffboardControlMode` |
| pub | `/fmu/in/vehicle_command` | `px4_msgs/VehicleCommand` |
| pub | `/servo/in/state` | `sensor_msgs/JointState` |
| sub | `/md/input` | `std_msgs/Int32` |
| sub | `/fmu/out/vehicle_status_v1` | `px4_msgs/VehicleStatus` |
| sub | `/fmu/out/vehicle_odometry` | `px4_msgs/VehicleOdometry` |
| sub | `/fmu/out/vehicle_local_position` | `px4_msgs/VehicleLocalPosition` |
| sub | `/servo/out/state` | `sensor_msgs/JointState` |

`vbats_mission` adds:

| Direction | Topic | Type |
|-----------|-------|------|
| sub | `/tactip/pose` | `geometry_msgs/TwistStamped` |
| sub | `/tactip/contact` | `std_msgs/Int8` |
| sub | `/controller/out/trajectory_setpoint` | `px4_msgs/TrajectorySetpoint` |
| sub | `/controller/out/servo_state` | `sensor_msgs/JointState` |
| client | `set_ssim_ref` | `std_srvs/SetBool` |

## Notes

- `/md/input == 1` manually triggers the next transition from `tactile_servoing`.
- The base class exposes reusable state helpers (`state_takeoff`, `state_hover`,
  `state_move_arms`, `state_approach_wall_position`, `state_disengage_wall_velocity`,
  `state_land`, `state_emergency`).
