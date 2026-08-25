# Simulation

The simulation lets you run the full control stack against a Gazebo model of the UAM, with no
hardware. It is the recommended environment for developing and testing missions.

## What runs in simulation

- **Gazebo** with the UAM model — URDF/xacro in
  [`px4_uam_sim/urdf`](../ros2_ws/src/simulation/px4_uam_sim/urdf/) (`martijn.urdf`,
  `martijn_one_arm.urdf`).
- **PX4 SITL** providing the flight dynamics and the `/fmu/*` interface.
- [`sim_remapper`](../ros2_ws/src/simulation/sim_remapper/README.md) — standalone ROS2 node that converts the ATS stack's
  `/servo/in/state` joint commands into Gazebo per-joint velocity commands
  (`/shoulder_1_vel_cmd`, `/elbow_1_vel_cmd`, `/forearm_1_vel_cmd`). Its purpose is to make the
  **servo interface in simulation identical to the real system**: the control code publishes the
  same topics.
- [`keyboard_rc_emulator`](../ros2_ws/src/simulation/keyboard_rc_emulator/README.md) — stands in
  for the RC transmitter so you can arm, switch modes, and trigger transitions from the keyboard if needed.

## Prerequisites

Install these once, outside the workspace:

- **PX4-Autopilot** cloned and built for SITL at `$HOME/PX4-Autopilot` (both the launch file and the
  manual PX4 command below assume this path). See [PX4 getting started docs](https://docs.px4.io/main/en/dev_setup/getting_started).
- **`MicroXRCEAgent`** on the `PATH` (the uXRCE-DDS agent; the launch file starts it on UDP port
  8888 after a short delay). See [PX4 ROS2 docs](https://docs.px4.io/main/en/ros2/user_guide).
- **Gazebo** with the `ros_gz_sim` / `ros_gz_bridge` packages (the ROS ↔ GZ bridge for clock (install through `sudo apt install ros-jazzy-ros-gz`),
  odometry, and per-joint commands). See [Gazebo install page](https://gazebosim.org/docs/harmonic/ros_installation/). Simulation was tested with Gazebo Harmonic. The packages ()
- **QGroundControl** (e.g. `~/QGroundControl.AppImage`). PX4 SITL needs a ground station connected
  to behave correctly, so keep QGC running for the whole session — you don't need to interact with
  it. Download from [QGroundControl installation](https://docs.qgroundcontrol.com/Stable_V5.0/en/qgc-user-guide/getting_started/download_and_install.html).

## Dependencies
There are some Python dependencies associated with the Tactip driver. These can be installed (virtual environment recommended) with the `requirements.txt` in project root.

## Building the simulation

The UAM's `.urdf` models are **generated from `.xacro` sources**. You generate them once after
cloning, and regenerate whenever you edit a `.xacro`. 
```bash
cd ~/aerial_tactile_servoing/ros2_ws

# 1. Build first — installs the meshes and makes the package discoverable so
#    that $(find px4_uam_sim) resolves when xacro expands the mesh paths.
colcon build
source install/setup.bash

# 2. Generate the URDF(s) from xacro — one-arm and/or two-arm as needed.
cd src/simulation/px4_uam_sim
xacro urdf/martijn_one_arm.xacro > urdf/martijn_one_arm.urdf
xacro urdf/martijn.xacro         > urdf/martijn.urdf
cd ~/aerial_tactile_servoing/ros2_ws

# 3. Build again so the generated .urdf is copied into the install share,
#    which is where the launch file loads the model from.
colcon build
source install/setup.bash
```

Gazebo also has to find the world file (`test_world.sdf`), which the launch passes by bare name. Add
the package's `worlds/` directory to the Gazebo resource path once (e.g. in `~/.bashrc` but should also work after just calling the below once):

```bash
export GZ_SIM_RESOURCE_PATH=$HOME/aerial_tactile_servoing/ros2_ws/src/simulation/px4_uam_sim/worlds:${GZ_SIM_RESOURCE_PATH}
```

## Running the simulation

A full sim needs different things to be run simultaneously. The most straightforward way is to run each in its own terminal (source the workspace in each first:
`source ~/aerial_tactile_servoing/ros2_ws/install/setup.bash`):

1. **QGroundControl** — start it first and leave it running. You don't need to look at it, but PX4
   won't behave correctly without a ground station connected.
   ```bash
   ~/QGroundControl.AppImage
   ```
Then, per new simulation run:

2. **ROS 2 / Gazebo stack** — Launches the ROS2 and Gazebo stack
   ```bash
   ros2 launch ats_bringup sim_gz_one_new.launch.py
   ```
3. **PX4 SITL** — start the ROS launch so it connects to the already-running Gazebo instance:
   ```bash
   cd ~/PX4-Autopilot
   PX4_SYS_AUTOSTART=4001 PX4_SIMULATOR=GZ PX4_GZ_MODEL_NAME=my_custom_model \
       ./build/px4_sitl_default/bin/px4
   ```
4. **Start the clock** — wait until PX4 has finished spinning up, then press the **▶ play button**
   in the Gazebo GUI. Simulation time only advances once you do this, so nothing moves until you
   press play.

About step 2: `sim_gz_one_new.launch.py` is the current sim entry point. It *includes*
`px4_uam_sim/launch/gz_martijn_one_arm.launch.py` (which starts Gazebo, spawns the model, starts the
`MicroXRCEAgent` and the `ros_gz` bridge) and then adds the control stack on top:

- the **sim remapper**, **mission director** (with `sm.sim: True`, `sm.fcu_on: False`),
  **TacTip driver** (with `fake_data: True`), **controller**, and **planner** — the same nodes as a
  real mission, wired to the simulated `/fmu/*` interface.
- The mission director variant launched here is `wallfollow_vbats_mission`; edit the launch file to
  run a different one.

Do not launch `gz_martijn_one_arm.launch.py` directly for a mission — it only brings up Gazebo and
the bridge, not the ROS2 control stack. (`gz_martijn.launch.py` is the same Gazebo-only bringup for the
two-arm model.)

## Flying the simulation

In simulation the mission director **arms and switches to offboard itself** (`sm.sim: True`): in
`wait_for_arming` it periodically sends the SITL arm/offboard commands, so you do not need an RC to
get airborne. The FSM then walks takeoff → hover → approach → tactile servoing on its own. To step
past any waiting state manually, publish to `/md/input`:

```bash
ros2 topic pub --once /md/input std_msgs/msg/Int32 "{data: 1}"
```

The `keyboard_rc_emulator` stands in for the RC sticks (hold-to-deflect, published on
`/fmu/in/rc_channels`). Start it in its own terminal and keep that window focused for the keys to
register:

| Key | Stick |
|-----|-------|
| `w` / `s` | pitch forward / back |
| `a` / `d` | roll left / right |
| `q` / `e` | yaw left / right |
| `r` / `f` | throttle up / down |
| `space` | reset sticks |
| `esc` | quit |

The emulator only drives the four analog sticks (channels 0–3); it does not provide the arm /
offboard / reference-manipulation switches, which is why the director self-arms in sim.

## Sim vs. hardware differences

The mission director takes a `sm.sim` parameter; when true, `fcu_on` is forced off and the node
uses the SITL arm/offboard command helpers instead of expecting a real FCU handshake. The concrete
behavioural differences to keep in mind when reading sim results:

| Aspect | Simulation | Hardware |
|--------|-----------|----------|
| Arming / offboard | Director self-commands (`sm.sim: True`) | Pilot commands via RC |
| `fcu_on` | Forced `False` (no real FCU watchdog transitions) | `True` |
| Tactile data | `fake_data: True` — a **fixed** −3 mm depth with SSIM 0.3 (always "in contact"); the pose model and camera are bypassed | Real camera → SSIM → pose model |
| Contact detection | Effectively hard-wired on, so approach transitions as soon as the FSM checks | Depends on real SSIM crossing the threshold |
| Servos | Gazebo joint dynamics via `sim_remapper` per-joint velocity commands | Dynamixel driver on the real bus |
| Odometry | PX4 SITL / Gazebo | PX4 EKF (typically mocap-aided) |

Because `fake_data` publishes a constant pose, the sim exercises the **FSM, plumbing, and servo/
body kinematics**, not the tactile perception or the closed-loop contact behaviour. Validate the
perception and gains on hardware (or with recorded data) before trusting them.
