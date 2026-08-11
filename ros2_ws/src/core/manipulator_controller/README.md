# manipulator_controller

Body-frame kinematic controller for the 3-DOF arm. It converts commanded end-effector Cartesian
position or velocity (in the UAM body frame) into joint states for the servo driver. Unlike the
tactile controller, it is **event-driven** — there is no timer; it acts on each incoming
end-effector command.

> This node provides direct arm command/control outside the tactile-servoing loop (e.g. for
> testing or non-ATS motions). During a mission the arm is driven by `pose_based_ats` via the
> mission director.

## Run

<!-- TODO: confirm the executable name (setup.py entry_points) and which launch files use it. -->

## Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `minimum_pivot_distance` | 1.5 | Minimum pivot distance used by the kinematics |

> Link lengths `L1/L2/L3` are module-level constants at the top of `manipulator_controller.py`.
> They currently differ from the values in `velocity_based_ats.py` — see
> [docs/03-system-architecture.md](../../../../docs/03-system-architecture.md).

## Topics

| Direction | Topic | Type |
|-----------|-------|------|
| sub | `/servo/out/state` | `sensor_msgs/JointState` |
| sub | `/manipulator/in/position` | `geometry_msgs/TwistStamped` |
| sub | `/manipulator/in/velocity` | `geometry_msgs/TwistStamped` |
| pub | `/servo/in/state` | `sensor_msgs/JointState` |
