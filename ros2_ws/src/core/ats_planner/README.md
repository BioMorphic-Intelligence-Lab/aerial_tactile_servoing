# ats_planner

Generates the tactile pose reference the controller tracks, published in the **contact frame**
(Z perpendicular to the interaction surface). The reference shape is chosen by the
`mission_preset` parameter, and RC dials can overlay shear references on top.

Because `/tactip/pose` only observes depth (z), roll, and pitch, commanding a nonzero reference
on any other axis induces open-loop motion in that direction (the feedback stays zero).

## Run

Launched as part of a mission:

```bash
ros2 launch ats_bringup vbats_mission.launch.py
```

## Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `frequency` | 10.0 | Publish rate (Hz) |
| `default_depth` | 3.0 | Default contact depth (mm) |
| `mission_preset` | `blockref_x` | Reference pattern (see below) |
| `verbose` | true | Verbose logging |

### Mission presets

| Preset | Behaviour |
|--------|-----------|
| `default` | Hold the default contact depth |
| `blockref_x` | Block/step reference in x |
| `blockref_y` | Block/step reference in y |
| `blockref_z` | Block/step reference in depth (z) |
| `slide_x` | Slide reference along x |

<!-- TODO: document the exact magnitude and timing of each preset. -->

## Topics

| Direction | Topic | Type |
|-----------|-------|------|
| sub | `/md/state` | `std_msgs/Int32` |
| sub | `/fmu/out/rc_channels` | `px4_msgs/RcChannels` |
| pub | `/references/sensor_in_contact` | `geometry_msgs/TwistStamped` |

## RC channel map

The planner reads the RC transmitter via `/fmu/out/rc_channels`. Dials on channels 12 and 13 map
to shear x/y references; the full switch/dial map is documented in the class docstring in
`ats_planner/ats_planner.py` and reproduced in
[docs/06-running-a-mission.md](../../../../docs/06-running-a-mission.md).
