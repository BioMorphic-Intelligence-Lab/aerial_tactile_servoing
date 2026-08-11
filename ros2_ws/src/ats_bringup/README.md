# ats_bringup

The launch and configuration package for the project. It has no nodes of its own — it wires the
other packages together into runnable missions and holds the parameter files. This is the package
you `ros2 launch` from.

## Launch files (`launch/`)

Mission launch files bring up the full stack (servo driver, tactile driver, controller, planner,
mission director) plus rosbag recording. Node parameters (controller gains, tactile thresholds,
etc.) are set **inline in the launch file**, not in the config YAMLs.

| Launch file | Purpose |
|-------------|---------|
| `vbats_mission.launch.py` | Baseline velocity-based tactile servoing |
| `door_vbats_mission.launch.py` | Door opening |
| `slide_vbats_mission.launch.py` | Sliding along a surface |
| `wallfollow_vbats_mission.launch.py` | Wall following |
| `leaderfollow_vbats_mission.launch.py` | Leader following |
| `ats_mission.launch.py` | Earlier ATS mission |
| `uam_control_test.launch.py` | Control bring-up test |
| `sim_gz_one_new.launch.py` | Gazebo simulation |
| `tactip.launch.py` | TacTip driver only (sensor bring-up) |
| `dynamixel_launch.launch.py` | Servo driver only |

Superseded launch files are kept under `launch/old/`. See
[docs/06-running-a-mission.md](../../../docs/06-running-a-mission.md).

## Config files (`config/`)

Dynamixel parameter files selected per platform configuration:

| File | Configuration |
|------|---------------|
| `dxl_ros2_vbats.yaml` | Current vbats flight arm (IDs 31/32/33, velocity mode) |
| `dxl_ros2_onearm.yaml` | Single-arm setup |
| `dxl_ros2_twoarms.yaml` | Two-arm setup |
| `dxl_ros2_single.yaml` | Single servo |
| `dxl_ros2_ats.yaml` | ATS arm |
| `feetech_ros2.yaml` | Feetech servos |

Parameter meanings are documented in the
[`dxl_driver`](../drivers/dynamixel/dxl_driver/README.md) README.

## Notes

- Each mission launch file has top-of-file switches: `logging` (record a rosbag), `log_path`
  (bag destination), and `config_name` (which dxl config to load).
- The recorded bag uses `ros2 bag record -a`; see
  [docs/08-data-and-logging.md](../../../docs/08-data-and-logging.md).
