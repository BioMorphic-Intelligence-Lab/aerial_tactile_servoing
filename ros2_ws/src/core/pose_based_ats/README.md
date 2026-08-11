# pose_based_ats

The tactile-servoing controllers. Given the tactile pose error and the reference from the
planner, these nodes compute a drone body twist and arm joint velocities using a weighted
pseudo-inverse of the manipulator Jacobian, with a null-space objective pulling the arm toward a
nominal configuration. They publish setpoints to `/controller/out/*`; the mission director
relays them to the aircraft. See
[docs/03-system-architecture.md](../../../../docs/03-system-architecture.md) for the control law.

## Nodes

| Executable | Source | Notes |
|------------|--------|-------|
| `velocity_based_ats` | `velocity_based_ats.py` | Current primary controller (vbats) |
| `pose_based_ats` | `pose_based_ats.py` | Earlier pose-based controller |

## Run

```bash
ros2 launch ats_bringup vbats_mission.launch.py   # gains are set in the launch file
```

## Parameters (`velocity_based_ats`)

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `frequency` | 10.0 | Control loop rate (Hz) |
| `Kp_depth` / `Ki_depth` / `Kd_depth` | 3.0 / 0.1 / 0.1 | PID gains on contact depth |
| `Kp_shear` / `Ki_shear` / `Kd_shear` | 5.0 / 0.1 / 2.0 | PID gains on shear |
| `Kp_angular` / `Ki_angular` / `Kd_angular` | 3.0 / 0.1 / 0.1 | PID gains on surface orientation |
| `Kp_secondary` | 1.0 | Null-space (secondary objective) gain |
| `IK_weights_flight` | `[25,25,10,10,5,20,5]` | Pseudo-inverse weights in flight |
| `IK_weights_contact` | `[25,25,10,10,1,20,1]` | Pseudo-inverse weights in contact |
| `fade_time_ik_weights` | 5.0 | Seconds to fade flight → contact weights after contact |
| `nominal_state` | see source | Nominal joint config for the null-space objective |
| `alpha` | 1.0 | Low-pass filter coefficient on the error |
| `windup_clip` | 10.0 | Integrator anti-windup clip |
| `test_execution_time` | false | Log per-cycle execution time |

> Note: link lengths `L_1/L_2/L_3` are module-level constants at the top of
> `velocity_based_ats.py`, not parameters.

## Topics (`velocity_based_ats`)

| Direction | Topic | Type |
|-----------|-------|------|
| sub | `/tactip/pose` | `geometry_msgs/TwistStamped` |
| sub | `/tactip/contact` | `std_msgs/Int8` |
| sub | `/servo/out/state` | `sensor_msgs/JointState` |
| sub | `/fmu/in/vehicle_visual_odometry` | `px4_msgs/VehicleOdometry` |
| sub | `/md/state` | `std_msgs/Int32` |
| sub | `/references/sensor_in_contact` | `geometry_msgs/TwistStamped` |
| pub | `/controller/out/servo_state` | `sensor_msgs/JointState` |
| pub | `/controller/out/trajectory_setpoint` | `px4_msgs/TrajectorySetpoint` |
| pub | `/controller/log/*` | debug (`us`, `e_sr`, `e_sr_filtered`, `uss`, `proportional`, `integrator`, `derivative`) |

`pose_based_ats` uses the same inputs but publishes `/controller/out/servo_positions` and
`/controller/optimizer/*` debug topics.

## Notes

- The controller only takes effect on the aircraft while the mission director is in the
  `tactile_servoing` state.
- `/tactip/pose` can only close the loop on depth (z), roll, and pitch; commanding a nonzero
  reference on an unobservable axis produces open-loop drift (see the planner docstring).
