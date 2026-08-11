# Hardware setup

> **Status: scaffold.** Fill in the physical-build details and photos.

## Platform overview

<!-- TODO: describe the airframe, arm mounting, and where the companion computer sits. Add a
     labelled photo in docs/images/. -->

## Companion computer

<!-- TODO: which SBC, OS image, how the repo is deployed on it, autostart. Reference dev_env.txt
     (the devcontainer environment it was validated in). -->

## Flight controller (PX4)

- Parameter set: [`px4_params/fc_params_vbats.params`](../px4_params/fc_params_vbats.params) —
  load this onto the flight controller.
- The companion computer talks to PX4 over **uXRCE-DDS**; `px4_ros2_driver` maintains the
  offboard publish rate and safety behaviour.

<!-- TODO: document the uXRCE-DDS wiring (serial/UDP, baud), how to start the agent, and the
     key non-default PX4 parameters and why they were changed. -->

## Manipulator (Dynamixel servos)

Three servos form the arm. The current flight configuration is
[`ats_bringup/config/dxl_ros2_vbats.yaml`](../ros2_ws/src/ats_bringup/config/dxl_ros2_vbats.yaml):

| Parameter | Value (vbats) | Meaning |
|-----------|---------------|---------|
| `ids` | `[31, 32, 33]` | Dynamixel bus IDs for q1, q2, q3 |
| `operating_modes` | `[1, 1, 1]` | 1 = velocity control, 4 = extended position control |
| `directions` | `[-1, -1, 1]` | Rotation sign per joint |
| `max_speeds` | `[1.0, 1.0, 1.5]` | rad/s limit per joint |
| `gear_ratios` | `[4.0, 1.0, 1.765]` | Servo horn : output shaft |
| `min_angles` / `max_angles` | see file | Joint limits (rad) |
| `start_offsets` | `[1.78, 0.0, -2.07]` | Offset added to initial position |

Other config presets live alongside it (`dxl_ros2_onearm.yaml`, `dxl_ros2_twoarms.yaml`, etc.).

<!-- TODO: document servo model, power, bus wiring (U2D2/serial), how to set IDs, and the homing
     procedure. Explain the mapping from bus IDs to q1/q2/q3 and the physical zero pose. -->

## TacTip sensor

A soft optical tactile sensor read as a USB camera (`/dev/videoN`, selected by the `source`
parameter of the tactip driver).

<!-- TODO: camera model, mounting, lighting, /dev/video enumeration, and how to identify the
     right source index. Cross-link to 07-tactile-sensing.md for the model. -->

## RC transmitter

RC channels reach the stack through PX4 (`/fmu/out/rc_channels`) and are interpreted by the
planner. The full channel map is documented in the planner source and reproduced in
[Running a mission](06-running-a-mission.md).

<!-- TODO: photo of the transmitter with switches labelled to match the channel map. -->
