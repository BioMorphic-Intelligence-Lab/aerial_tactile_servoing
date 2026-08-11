# keyboard_rc_emulator

Simulation-only node that stands in for the RC transmitter: it reads the keyboard (via `pynput`,
hold-to-deflect) and publishes PX4 `RcChannels` messages, so you can arm, switch modes, and fly
the simulated platform without physical RC hardware.

## Run

```bash
ros2 run keyboard_rc_emulator keyboard_rc_emulator
```

Runs alongside the simulation (see [docs/05-simulation.md](../../../../docs/05-simulation.md)).
Publishes at 50 Hz; sticks are normalized to [-1, 1].

## Key bindings

| Key(s) | Action |
|--------|--------|
| `w` / `s` | Pitch forward / back |
| `a` / `d` | Roll left / right |
| `q` / `e` | Yaw left / right |
| `r` / `f` | Throttle up / down |
| `space` | Reset sticks |
| `esc` | Quit |

## Topics

| Direction | Topic | Type |
|-----------|-------|------|
| pub | `/fmu/in/rc_channels` | `px4_msgs/RcChannels` |

## Notes

- Requires `pynput` and keyboard focus / a display.
- Only the analogue sticks are emulated here; mode/arm switch channels are set in code.
  <!-- TODO: document how arm/offboard/mode switch channels are emulated, since the planner and
       mission director expect them (see RC channel map in docs/06-running-a-mission.md). -->
