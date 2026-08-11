# Simulation

> **Status: scaffold.** Fill in the PX4 SITL + Gazebo bring-up specifics.

The simulation lets you run the full control stack against a Gazebo model of the UAM, with no
hardware. It is the recommended environment for developing and testing missions.

## What runs in simulation

- **Gazebo** with the UAM model — URDF/xacro in
  [`px4_uam_sim/urdf`](../ros2_ws/src/simulation/px4_uam_sim/urdf/) (`martijn.urdf`,
  `martijn_one_arm.urdf`).
- **PX4 SITL** providing the flight dynamics and the `/fmu/*` interface.
- [`sim_remapper`](../ros2_ws/src/simulation/sim_remapper/README.md) — converts the stack's
  `/servo/in/state` joint commands into Gazebo per-joint velocity commands
  (`/shoulder_1_vel_cmd`, `/elbow_1_vel_cmd`, `/forearm_1_vel_cmd`).
- [`keyboard_rc_emulator`](../ros2_ws/src/simulation/keyboard_rc_emulator/README.md) — stands in
  for the RC transmitter so you can arm, switch modes, and trigger transitions from the keyboard.

## Launch

```bash
ros2 launch ats_bringup sim_gz_one_new.launch.py
```

<!-- TODO: confirm this is the current canonical sim launch file (vs. px4_uam_sim/launch/
     gz_martijn_one_arm.launch.py) and document what each node in it does. -->

## Prerequisites

<!-- TODO: PX4-Autopilot version/branch, how to build SITL, the uXRCE-DDS agent, the Gazebo
     version, and any GZ_SIM_RESOURCE_PATH / model path setup. -->

## Flying the simulation

<!-- TODO: keyboard_rc_emulator key bindings; the arm/offboard sequence; how to drive the FSM
     through takeoff → approach → tactile servoing in sim. -->

## Sim vs. hardware differences

The mission director takes a `sm.sim` parameter; when true, `fcu_on` is forced off and the node
uses the SITL arm/offboard command helpers instead of expecting a real FCU handshake.

<!-- TODO: enumerate every sim/hardware behavioural difference (odometry source, contact
     detection with fake_data, servo dynamics) so results transfer predictably. -->
