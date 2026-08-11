# Aerial Tactile Servoing

Companion-computer software for the **Aerial Tactile Servoing (ATS)** project: a PX4 quadrotor
carrying a 3-DOF arm that ends in a [TacTip](https://www.softformlab.com/) optical tactile
sensor. The system flies to a surface, establishes contact, and then servos the drone and arm
together to track a tactile reference (contact depth, shear, orientation) while sliding along
the surface.

This repository is a ROS 2 (Humble) workspace: the `ros2_ws/src` tree contains every node that
runs on the onboard companion computer, plus a Gazebo simulation of the platform.

> **New here? Start with the [documentation](docs/README.md).** The `docs/` folder is the
> intended reading path — this README is only the front door.

## Documentation

| Guide | What it covers |
|-------|----------------|
| [Overview](docs/01-overview.md) | System architecture, node/topic graph, mission concept |
| [Getting started](docs/02-getting-started.md) | Install, build, and run the simulation for the first time |
| [System architecture](docs/03-system-architecture.md) | Control loop, inverse kinematics, contact detection, frames |
| [Hardware setup](docs/04-hardware-setup.md) | Drone, servos, TacTip camera, RC mapping, PX4 params |
| [Simulation](docs/05-simulation.md) | Running the Gazebo UAM simulation |
| [Running a mission](docs/06-running-a-mission.md) | Launch files, mission presets, the flight state machine, safety |
| [Tactile sensing](docs/07-tactile-sensing.md) | The TacTip model, contact detection, swapping/retraining |
| [Data & logging](docs/08-data-and-logging.md) | Rosbags, the sync script, the `data/` folder |
| [Troubleshooting](docs/09-troubleshooting.md) | Common failures and fixes |
| [Glossary](docs/10-glossary.md) | Project-specific terms |

Each ROS 2 package under `ros2_ws/src` also has its own `README.md` documenting its nodes,
parameters, and topics.

## Quickstart

```bash
# Clone with submodules (px4_msgs, DynamixelSDK)
git clone --recursive https://github.com/mbrummelhuis/aerial_tactile_servoing.git
cd aerial_tactile_servoing/ros2_ws

# Build and source
colcon build
source install/setup.bash

# Run a simulated mission
ros2 launch ats_bringup sim_gz_one_new.launch.py
```

See [Getting started](docs/02-getting-started.md) for the full setup (virtual environment,
dependencies, Docker) and [Running a mission](docs/06-running-a-mission.md) for the hardware
workflow.

## License

See [LICENSE](LICENSE).
