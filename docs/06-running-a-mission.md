# Running a mission

> **Status: scaffold.** The state machine and preset tables are filled from the code; the
> operating procedure and safety checklist still need to be written.

A "mission" is a launch file that brings up the stack (servo driver, tactile driver, controller,
planner, mission director) plus rosbag recording, and a **mission director** node that walks a
finite state machine through the flight.

## Launch files

All launch files live in [`ats_bringup/launch`](../ros2_ws/src/ats_bringup/launch/). The current
ones:

| Launch file | Mission director node | Purpose |
|-------------|----------------------|---------|
| `vbats_mission.launch.py` | `vbats_mission` | Baseline velocity-based tactile servoing on a wall |
| `door_vbats_mission.launch.py` | `door_vbats_mission` | Door-opening variant (uses `/mocap/door/pose`) |
| `slide_vbats_mission.launch.py` | `slide_vbats_mission` | Sliding along a surface |
| `wallfollow_vbats_mission.launch.py` | `wallfollow_vbats_mission` | Wall following (uses `/mocap/leader/pose`) |
| `leaderfollow_vbats_mission.launch.py` | `leaderfollow_vbats_mission` | Leader following |
| `ats_mission.launch.py` | `ats_mission` | Earlier ATS mission |
| `sim_gz_one_new.launch.py` | — | Gazebo simulation (see [Simulation](05-simulation.md)) |
| `tactip.launch.py` | — | TacTip driver only (sensor bring-up / testing) |
| `dynamixel_launch.launch.py` | — | Servo driver only |
| `uam_control_test.launch.py` | `uam_control_test` | Control bring-up test |

```bash
ros2 launch ats_bringup vbats_mission.launch.py
```

<!-- TODO: document how gains and parameters are set in the launch files (they are inlined, not
     in the config YAMLs — e.g. the controller PID gains in vbats_mission.launch.py). Explain the
     `logging` / `log_path` / `config_name` top-of-file switches. -->

## The flight state machine

Every mission director subclasses `UAMStateMachine` and implements an `execute()` method that
`match`es on `self.FSM_state`. Transitions are logged as `[TRANSITION] <from> -> <to>`. The
`vbats_mission` sequence:

| State | What happens | Transitions to |
|-------|--------------|----------------|
| `entrypoint` | Initial state | `arms_takeoff_position` |
| `arms_takeoff_position` | Move arm to takeoff pose | `wait_for_arm_offboard` |
| `wait_for_arm_offboard` | Set the SSIM contact reference; wait for arming + offboard | `takeoff` |
| `takeoff` | Climb to target altitude | `hover` |
| `hover` | Hold position briefly | `pre_contact_arm_position` |
| `pre_contact_arm_position` | Move arm to pre-contact pose | `approach` |
| `approach` | Move toward the surface until contact is detected | `tactile_servoing` |
| `tactile_servoing` | Relay controller setpoints to FCU and servos; run the servoing loop | `disengage` / `pre_contact_uam_position` / `emergency` |
| `disengage` | Back off from the surface | `land_arms_position` |
| `land_arms_position` | Move arm to landing pose | `land` |
| `land` | Land | `done` |
| `done` | Mission complete | — |
| `emergency` | Safe stop | — |

`tactile_servoing` exits on any of: the configured `tactile_servoing_time` elapsing, a manual
trigger (`/md/input == 1`), the end-effector passing a position bound, **loss of contact** for
longer than `ts_no_contact_max_seconds` (→ returns toward approach), or loss of offboard mode
(→ `emergency`).

<!-- TODO: document /md/input — the numeric transition-trigger topic — and how the operator uses
     it. Document the numeric /md/state codes (e.g. tactile_servoing publishes 30) and where they
     are defined. -->
<!-- TODO: describe how the other mission variants differ from vbats_mission. -->

## Planner mission presets

The [`ats_planner`](../ros2_ws/src/core/ats_planner/README.md) shapes the tactile reference
according to its `mission_preset` parameter:

| Preset | Behaviour |
|--------|-----------|
| `default` | Hold the default contact depth |
| `blockref_x` | Step/block reference in x |
| `blockref_y` | Step/block reference in y |
| `blockref_z` | Step/block reference in depth (z) |
| `slide_x` | Slide reference along x |

<!-- TODO: define exactly what each preset commands (magnitudes, timing) and how RC dials
     (channels 12/13 = shear x/y) overlay on top. -->

## RC channel map

The planner interprets RC channels (from `/fmu/out/rc_channels`) as follows:

| Ch | Function |
|----|----------|
| 0 | Yaw |
| 1 | Throttle |
| 2 | Pitch |
| 3 | Roll |
| 5 | Arm (1 armed / -1 disarmed) |
| 6 | Flight mode (1 position / 0 stabilized / -1 altitude) |
| 7 | Offboard (1 offboard / -1 manual) |
| 9 | Kill (1 killed / -1 alive) |
| 12 | Shear x reference (S1 dial) |
| 13 | Shear y reference (S2 dial) |

Unlisted channels are unused. See the docstring in
[`ats_planner.py`](../ros2_ws/src/core/ats_planner/ats_planner/ats_planner.py) for the full map.

## Pre-flight and safety

<!-- TODO: write the operating procedure and safety checklist:
     - power-up order, sourcing the workspace, sanity-checking topics
     - arming/offboard sequence via RC
     - the kill switch and what it does
     - position clipping (sm.position_clip) and speed limits (max_speed)
     - what to watch on approach and how to abort
     This is the most safety-critical page — do not fly from the scaffold. -->
