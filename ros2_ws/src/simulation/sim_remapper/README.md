# sim_remapper

Simulation-only bridge between the control stack's joint interface and Gazebo. It subscribes to
the standard `/servo/in/state` joint commands and republishes them as per-joint velocity commands
that the Gazebo model's controllers consume, so the same stack that flies the real arm drives the
simulated one unchanged.

## Run

Brought up by the simulation launch (see
[docs/05-simulation.md](../../../../docs/05-simulation.md)).

## Topics

| Direction | Topic | Type |
|-----------|-------|------|
| sub | `/servo/in/state` | `sensor_msgs/JointState` |
| sub | `/servo/out/state` | `sensor_msgs/JointState` |
| pub | `/shoulder_1_vel_cmd`, `/elbow_1_vel_cmd`, `/forearm_1_vel_cmd` | arm 1 joint velocity |
| pub | `/shoulder_2_vel_cmd`, `/elbow_2_vel_cmd`, `/forearm_2_vel_cmd` | arm 2 joint velocity |

> The single-arm variant in `px4_uam_sim/sim_remapper.py` publishes only the arm-1 commands.

## Parameters

<!-- TODO: document parameters (if any) and how joint order maps to the Gazebo controllers. -->
