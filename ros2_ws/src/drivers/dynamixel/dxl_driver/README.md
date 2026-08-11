# dxl_driver

ROS 2 driver for the arm's Dynamixel servos, built on the ROBOTIS DynamixelSDK (the
`dynamixel_sdk` submodule). It reads joint references off `/servo/in/state` and writes joint
feedback to `/servo/out/state`, supporting position and velocity operating modes with per-servo
direction, gear ratio, limits, and offsets.

## Run

```bash
ros2 run dxl_driver dxl_driver_node --ros-args --params-file <config>.yaml
# or via a launch file, e.g.:
ros2 launch ats_bringup dynamixel_launch.launch.py
```

## Configuration

Servos are configured from a YAML parameter file. The active flight config is
[`ats_bringup/config/dxl_ros2_vbats.yaml`](../../../ats_bringup/config/dxl_ros2_vbats.yaml); an
annotated template is
[`config/example_dxl_config.yaml`](config/example_dxl_config.yaml).

| Parameter | Meaning |
|-----------|---------|
| `driver.frequency` | Bus update rate (Hz) |
| `servos.ids` | Dynamixel bus IDs, in joint order (q1, q2, q3) |
| `servos.operating_modes` | Per servo: 1 = velocity, 4 = extended position |
| `servos.homing_modes` | Homing behaviour (0 = none) |
| `servos.directions` | Rotation sign (`-1`/`1`) |
| `servos.max_speeds` | rad/s limit |
| `servos.gear_ratios` | Servo horn : output shaft |
| `servos.min_angles` / `max_angles` | Joint limits (rad); both 0 = no limit |
| `servos.start_offsets` | Offset (rad) added after homing |

## Topics

| Direction | Topic | Type |
|-----------|-------|------|
| sub | `/servo/in/state` | `sensor_msgs/JointState` |
| pub | `/servo/out/state` | `sensor_msgs/JointState` |

## Notes

- The joint index in `/servo/*` messages follows the *order* of `servos.ids`, not the bus IDs.
- See [docs/04-hardware-setup.md](../../../../../docs/04-hardware-setup.md) for wiring and ID
  assignment.
