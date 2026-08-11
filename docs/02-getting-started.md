# Getting started

Start reading here to get up to speed.

## Prerequisites

- **ROS 2 Humble** on Ubuntu 22.04 (Jazzy may also work).
- **Python 3.10**.
- For simulation: **Gazebo** and **PX4-Autopilot** with the uXRCE-DDS agent (see
  [Simulation](05-simulation.md)).
- For hardware: a PX4 flight controller, Dynamixel servos, and a TacTip sensor (see
  [Hardware setup](04-hardware-setup.md)).

## 1. Clone with submodules

The workspace pulls two dependencies as git submodules — [`px4_msgs`](https://github.com/PX4/px4_msgs)
and the [DynamixelSDK](https://github.com/ROBOTIS-GIT/DynamixelSDK) — so clone recursively:

```bash
git clone --recursive https://github.com/mbrummelhuis/aerial_tactile_servoing.git
cd aerial_tactile_servoing
# If you forgot --recursive:
git submodule update --init --recursive
```

The colcon workspace root is `ros2_ws/`. The repository root additionally holds `data/`,
`px4_params/`, `docker/`, and this documentation.

## 2. Setup environment for data syncing
Optional: Rename .env.example into .env and fill out the fields in the file. This allows you to call the `sync_ros2bags.sh`, which syncs the local rosbag data from the drone onto the ground station. 

## 2. Python dependencies

The tactile pipeline needs PyTorch, OpenCV, scikit-image, and friends on top of the ROS 2
Python packages. `requirements.txt` (in the repository root) exports the environment, although admittedly it contains a lot of clutter.

The cleanest approach that does not fight the system ROS 2 install is a virtual environment
with access to the system site packages:

```bash
python3 -m venv --system-site-packages venv-ats
source venv-ats/bin/activate
pip install -r requirements.txt
```

> `requirements.txt` includes the CUDA `nvidia-*` wheels and `torch`. On a machine without an
> NVIDIA GPU the model still runs on CPU; installation just pulls the CUDA wheels regardless.

## 3. Build

```bash
cd ros2_ws
colcon build --symlink-install
source install/setup.bash
```

`--symlink-install` lets you edit the Python nodes without rebuilding. The C++ packages
(`dxl_driver`, `px4_msgs`) still need a rebuild when their sources change.

## 4. Run the simulation
This is the recommended first run — no hardware required. See [Simulation](05-simulation.md)
for how to start a simulation run and what this brings up (Gazebo, the sim remapper, the keyboard RC emulator) and how to fly it.

## Running with Docker

The `docker/` folder provides a devcontainer-style setup so you can run the full stack without
installing ROS 2 on the host, mainly meant for deployment on different companion computers on flying robots. There are several Compose files for different workflows:

| Compose file | Use |
|--------------|-----|
| `docker-compose-build.yaml` | Build the image locally and colcon-build the workspace inside it |
| `docker-compose-no-build.yaml` | Use a prebuilt image, mount the workspace, build inside |
| `docker-compose-pull.yaml` | Pull the prebuilt image from Docker Hub and run |
| `docker-compose-no-build-jazzy.yaml` | Same, on ROS 2 Jazzy |

```bash
cd docker
docker compose -f docker-compose-pull.yaml up
```

The currently used workflow spins up the `docker-compose-no-build-jazzy.yaml` services, the others are not guaranteed to work

<!-- TODO: sort out docker setup. -->

## Next steps

- Understand the control loop: [System architecture](03-system-architecture.md)
- Fly a simulated mission: [Running a mission](06-running-a-mission.md)
- Move to hardware: [Hardware setup](04-hardware-setup.md)
