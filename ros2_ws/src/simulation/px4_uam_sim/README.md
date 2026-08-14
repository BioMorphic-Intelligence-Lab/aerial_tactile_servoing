# px4_uam_sim

Gazebo simulation of the UAM: the drone + arm model (URDF/xacro), the Gazebo world, and the launch
files that bring up Gazebo and spawn the model alongside PX4 SITL.

> **Setting the sim up for the first time?** The full workflow — prerequisites, building, generating
> the URDF from xacro, and the manual startup sequence (QGroundControl + ROS launch + PX4 SITL) — is
> in **[docs/05-simulation.md](../../../../docs/05-simulation.md)**. This README is the package-level
> reference.

## Contents

| Path | Description |
|------|-------------|
| `urdf/martijn.xacro` | Two-arm UAM model (source) |
| `urdf/martijn_one_arm.xacro` | One-arm UAM model (source) |
| `urdf/meshes/` | STL/DAE meshes the models reference |
| `worlds/test_world.sdf` | Gazebo world the launch files load |
| `launch/gz_martijn_one_arm.launch.py` | Gazebo + one-arm model bringup |
| `launch/gz_martijn.launch.py` | Gazebo + two-arm model bringup |
| `src/mocap_forwarder.cpp` | Mocap → PX4 odometry forwarder (C++ node) |
| `src/uam_handler.cpp` | Standalone PID offboard commander (C++ node) |

**The `.urdf` files are generated from the `.xacro` sources and are intentionally not tracked in
git** — they bake in machine-specific absolute mesh paths. Generate them after building (see the
build steps in docs/05), and always edit the `.xacro`, never the `.urdf`. The X500 base model comes
from [PX4-gazebo-models](https://github.com/PX4/PX4-gazebo-models).

## Building

```bash
cd <ros2_ws>
colcon build --packages-select px4_uam_sim
```

`colcon build` installs the launch files, the URDF/xacro sources, the meshes, and the world into the
package's install share (`share/px4_uam_sim/`). The generated `.urdf` embeds absolute paths to the
installed meshes; Gazebo finds the world file (`test_world.sdf`) via `GZ_SIM_RESOURCE_PATH`, which
you point at the package's `worlds/` directory (see docs/05). The old manual mesh-copy step is no
longer needed.

## Launch

The package's own launch files bring up **only Gazebo, the model, the MicroXRCE-DDS agent, and the
`ros_gz` bridge** — not the control stack. For a full mission use the `ats_bringup` entry point,
which *includes* the one-arm bringup and adds the controllers on top:

```bash
# Full sim stack (recommended entry point):
ros2 launch ats_bringup sim_gz_one_new.launch.py

# Package-local bringup only (Gazebo + model + bridge, no control stack):
ros2 launch px4_uam_sim gz_martijn_one_arm.launch.py   # one arm
ros2 launch px4_uam_sim gz_martijn.launch.py           # two arms
```

Do not use the package-local launch files for a mission — without the control stack the model just
sits in Gazebo. See docs/05 for the full run procedure (PX4 SITL, QGroundControl, and pressing play
in Gazebo to start the clock).

## C++ nodes

- **`mocap_forwarder`** — subscribes to external/motion-capture odometry (`nav_msgs/Odometry`) and
  republishes it as PX4 `VehicleOdometry` on `/fmu/in/vehicle_visual_odometry` (PX4's external
  vision/odometry input). Launched as part of the sim bringup.
- **`uam_handler`** — a self-contained PID offboard commander for the UAV + arms. **Not referenced by
  any current launch file** — it predates the refactored control stack (`pose_based_ats`,
  `mission_director`, …) and is kept for reference only.

## Related sim-only nodes

- [`sim_remapper`](../sim_remapper/README.md) — makes the sim's servo interface **identical to
  hardware** by bridging `/servo/*` to Gazebo joint velocity commands, so controllers run unchanged
  in sim and on the real system. Launched by `sim_gz_one_new.launch.py`.
- [`keyboard_rc_emulator`](../keyboard_rc_emulator/README.md) — RC transmitter input from the
  keyboard.
