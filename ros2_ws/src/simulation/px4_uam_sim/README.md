# px4_uam_sim

Gazebo simulation of the UAM: the drone + arm model (URDF/xacro), worlds, and launch files that
bring up Gazebo alongside PX4 SITL. See
[docs/05-simulation.md](../../../../docs/05-simulation.md) for the full simulation workflow.

## Prerequisites

You need **PX4 SITL** and a **Micro XRCE-DDS Agent**. Follow the *Install PX4* and *Setup Micro
XRCE-DDS Agent & Client* instructions at
<https://docs.px4.io/main/en/ros2/user_guide.html>, and install **QGroundControl**
(<https://docs.qgroundcontrol.com/master/en/qgc-user-guide/getting_started/download_and_install.html>).

Indicative apt dependencies (substitute your ROS distro):

```
ros-<distro>-gz-sim-vendor ros-<distro>-ros-gz-bridge ros-<distro>-effort-controllers \
ros-<distro>-mavros-msgs ros-<distro>-libmavconn
```

## One-time setup

After `colcon build`, make the meshes and world resources discoverable by Gazebo:

```bash
cp -r <ros2_ws>/src/simulation/px4_uam_sim/urdf/meshes \
      <ros2_ws>/install/px4_uam_sim/share/px4_uam_sim
echo 'export GZ_SIM_RESOURCE_PATH=$HOME/<ros2_ws>/src/simulation/px4_uam_sim/worlds:${GZ_SIM_RESOURCE_PATH}' >> ~/.bashrc
```

The X500 base model comes from [PX4-gazebo-models](https://github.com/PX4/PX4-gazebo-models).

## Models

| File | Description |
|------|-------------|
| `urdf/martijn.urdf` / `.xacro` | Two-arm UAM model |
| `urdf/martijn_one_arm.urdf` / `.xacro` | One-arm UAM model |

## Launch

```bash
# Recommended entry point (brings up the full sim stack):
ros2 launch ats_bringup sim_gz_one_new.launch.py

# Package-local launch files:
ros2 launch px4_uam_sim gz_martijn_one_arm.launch.py
ros2 launch px4_uam_sim gz_martijn.launch.py
```

<!-- TODO: document what each launch file spawns and the difference between the ats_bringup sim
     launch and the package-local ones. Note the provided helper script and the folder paths that
     must be customized before running. -->

## Related sim-only nodes

- [`sim_remapper`](../sim_remapper/README.md) — bridges `/servo/*` to Gazebo joint commands.
- [`keyboard_rc_emulator`](../keyboard_rc_emulator/README.md) — RC input from the keyboard.

> This package also contains a `sim_remapper.py` node (`px4_uam_sim/sim_remapper.py`) that
> remaps the one-arm joints. The standalone `sim_remapper` package covers both arms.
