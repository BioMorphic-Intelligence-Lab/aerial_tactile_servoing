# manipulator_controller

Body-frame kinematic controller for the 3-DOF arm. It converts commanded joint- or task-space
position or velocity into joint references for the servo driver, published at `frequency` Hz.

It is a **two-stage pipeline**:

- **Stage 1 (normalize):** each of the four command interfaces is reduced to a joint-space
  reference — a joint position target or a joint velocity target — independent of the servo mode.
- **Stage 2 (emit):** the reference is converted to the field the active `servo_mode` requires,
  cross-converting where needed:

  | Joint-space ref | `servo_mode: velocity` | `servo_mode: position` |
  |---|---|---|
  | position target | PD law → `.velocity` | pass through → `.position` |
  | velocity target | pass through → `.velocity` | integrate `q += qdot·dt` (seeded from the measured pose, clamped to `min/max_angles`) → `.position` |

Only the active field of `/servo/in/state` is filled; the other is left empty so a `servo_mode`
mismatch with the driver is detectable. **`servo_mode` must match the driver's
`servos.operating_modes`** (velocity = `1`, position = `3`/`4`); set both from the same launch
argument. `kp`/`kd`/`tolerance` apply in velocity mode only.

> This node provides direct arm command/control outside the tactile-servoing loop (e.g. for
> testing or non-ATS motions). During a mission the arm is driven by `pose_based_ats` via the
> mission director.

## Run

<!-- TODO: confirm the executable name (setup.py entry_points) and which launch files use it. -->

## Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `servo_mode` | `velocity` | Which `/servo/in/state` field carries the reference: `velocity` or `position`. Must match the driver's `operating_modes`. |
| `frequency` | 10.0 | Control/publish rate (Hz). |
| `n_arms` | 1 | Number of serial arms. |
| `kp` / `kd` | 2.0 / 0.3 | Joint position PD gains (velocity mode only). |
| `tolerance` | 0.2 | Joint-error stop band for position commands (velocity mode only). |
| `min_angles` / `max_angles` | all `0.0` | Per-joint limits for the position-mode integrator, as **flat `n_arms*3` vectors** indexed like the servos (arm 0 → joints 0..2, arm 1 → 3..5, …), matching the driver's `servos.min_angles`/`max_angles`. Both zero for a joint = no limit. Default: no limits. |
| `minimum_pivot_distance` | 1.5 | Multi-arm collision guard: min base-joint separation (rad). |
| `pivot_ramp` | 1.0 | Velocity-mode guard ramp width (dimensionless). |
| `link_lengths` | `[0.12025, 0.336, 0.325]` | Link lengths for the kinematics. |

> Link lengths are the `link_lengths` parameter. They may differ from the values in
> `velocity_based_ats.py` — see
> [docs/03-system-architecture.md](../../../../docs/03-system-architecture.md).

## Topics

| Direction | Topic | Type |
|-----------|-------|------|
| sub | `/servo/out/state` | `sensor_msgs/JointState` |
| sub | `/manipulator/in/position` | `geometry_msgs/TwistStamped` |
| sub | `/manipulator/in/velocity` | `geometry_msgs/TwistStamped` |
| pub | `/servo/in/state` | `sensor_msgs/JointState` |
