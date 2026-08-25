# ats_launch

Launch and configuration package for the aerial tactile servoing project.
**Supersedes `ats_bringup`** (which is kept as-is for now so historical gain
settings stay recoverable).

The idea: declare the node graph **once**, and make each mission a thin wrapper
that selects a per-mission parameter file. No more copy-pasting a 120-line
launch file per experiment.

## Layout

```
launch/
  ats_stack.launch.py     # the full node graph, declared ONCE
  missions/
    vbats.launch.py       # thin wrapper: picks params + md executable, includes the stack
config/
  servo/
    vbats.yaml            # dxl_driver parameter file (per platform)
  missions/
    vbats.yaml            # all per-node params for the mission (controller gains, etc.)
```

## Running

```bash
ros2 launch ats_launch vbats.launch.py                # no rosbag
ros2 launch ats_launch vbats.launch.py logging:=true  # record a rosbag
```

## Adding / retuning a mission

1. Copy `config/missions/vbats.yaml` to `config/missions/<name>.yaml` and edit
   the values (gains, `mission_preset`, thresholds, ...).
2. Copy `launch/missions/vbats.launch.py` to `launch/missions/<name>.launch.py`
   and change `md_executable`, the `mission_params` path and `rosbag_prefix`.

You should not need to touch `ats_stack.launch.py`. If a mission needs an *extra*
node (e.g. a wrench observer) or the sim graph, that is a stack-level change --
see the notes in `ats_stack.launch.py`.

## Status

Proof-of-concept: only `vbats` is migrated. The remaining missions still live in
`ats_bringup/launch/` and will move over once this pattern is validated.
