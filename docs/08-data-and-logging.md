# Data & logging

> **Status: scaffold**, but the rosbag sync workflow below is complete.

## The `data/` folder

Flight and sensor data is collected under `data/` in the repository root:

| Subfolder | Contents |
|-----------|----------|
| `data/rosbags` | ROS 2 bags recorded during missions |
| `data/px4_logs` | PX4 flight logs (`.ulg`) |
| `data/serial_logs` | Serial capture logs |
| `data/tactip_images` | Debug images saved by the tactile driver |

`data/` is intended for local artifacts and is largely git-ignored.

## Recording

Mission launch files record a full bag automatically when their `logging` flag is set (top of
the launch file), via an `ExecuteProcess` running `ros2 bag record -a`. The bag name is
timestamped and written to the launch file's `log_path`.

<!-- TODO: note that log_path is currently an absolute container path
     (/ros2_ws/aerial_tactile_servoing/rosbags/) — document the intended path on the target
     computer and how to change it. -->

## Syncing bags off the drone

`sync_ros2bags.sh` (repository root) pulls new rosbag directories from the drone's companion
computer over SSH with `rsync`, skipping any already present locally.

1. Copy the example config and fill it in:

   ```bash
   cp env.example .env
   ```

   `.env` (git-ignored) holds:

   | Variable | Meaning |
   |----------|---------|
   | `REMOTE_USER` | SSH user on the drone |
   | `REMOTE_HOST` | Drone hostname or IP |
   | `REMOTE_DIR` | Rosbag directory on the drone |
   | `LOCAL_DIR` | Destination directory on your machine |
   | `SSH_KEY` | SSH private key to authenticate with |

2. Run the sync:

   ```bash
   ./sync_ros2bags.sh
   ```

   New directories are downloaded; existing ones are skipped. A transfer log is written to
   `$LOCAL_DIR/log_ros2bag_sync.txt`.

## Analysis

<!-- TODO: document how bags are analysed — which topics matter, the /controller/log/* and
     /contact/log/* debug topics, and any plotting scripts/notebooks used. -->
