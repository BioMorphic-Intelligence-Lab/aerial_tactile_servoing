# wrench_observer

Estimates the external contact wrench (force/torque) acting on the platform from onboard IMU
data, motor commands, and joint states — without a dedicated force/torque sensor.

## Nodes

| Executable | Source | Estimates |
|------------|--------|-----------|
| `wrench_observer` | `wrench_observer.py` | Full wrench: force, torque, line of action, contact point |
| `torque_observer` | `torque_observer.py` | Torque only |

<!-- TODO: confirm executable names from setup.py entry_points and document the observer model
     (inputs → estimate). State whether outputs feed the control loop or are logging only. -->

## Parameters

<!-- TODO: document parameters (smoothing/filter constants, mass/inertia, etc.). -->

## Topics

### `wrench_observer`
| Direction | Topic | Type (approx.) |
|-----------|-------|----------------|
| sub | `/fmu/out/sensor_combined` | `px4_msgs/SensorCombined` |
| sub | `/fmu/out/actuator_motors` | `px4_msgs/ActuatorMotors` |
| sub | `/servo/out/state` | `sensor_msgs/JointState` |
| pub | `/contact/out/estimated_force` | force estimate |
| pub | `/contact/out/estimated_torque` | torque estimate |
| pub | `/contact/out/force_magnitude` / `/contact/out/torque_magnitude` | magnitudes |
| pub | `/contact/out/loa_direction` / `/contact/out/loa_point` | line of action |
| pub | `/contact/out/contact_point` / `/contact/out/contact_point_coords` | estimated contact point |
| pub | `/contact/log/*` | smoothed/unsmoothed intermediates |

### `torque_observer`
| Direction | Topic |
|-----------|-------|
| sub | `/fmu/out/sensor_combined`, `/fmu/out/actuator_motors`, `/servo/out/state` |
| pub | `/wrench_observer/out/estimated_torque`, `/wrench_observer/out/torque_magnitude`, `/wrench_observer/log/*` |
