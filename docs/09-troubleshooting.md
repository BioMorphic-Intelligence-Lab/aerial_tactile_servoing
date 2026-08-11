# Troubleshooting

> **Status: scaffold.** Add entries as failure modes are encountered. Format: symptom → cause →
> fix.

## Build

<!-- TODO: colcon build failures — px4_msgs not built first, missing submodules, Python/torch
     version mismatches with the venv. -->

## Nodes not talking

- **No `/tactip/*` topics** — check the camera `source` index matches the actual `/dev/videoN`,
  and that the driver found the model (`BASE_MODEL_PATH` is logged at startup).
- **No `/fmu/out/*` topics** — the uXRCE-DDS agent is not running or not connected to PX4.
- **Mission director stuck in `wait_for_arm_offboard`** — the `set_ssim_ref` service never
  returned, or arming/offboard was never commanded from the RC.

<!-- TODO: expand with real incidents from flight testing. -->

## Contact detection

<!-- TODO: contact never/always asserted — SSIM threshold, stale reference image, lighting
     changes. How to inspect /tactip/ssim live. -->

## Servos

<!-- TODO: wrong joint moves / wrong direction — bus ID vs q-index mapping, `directions` sign,
     start_offsets, joint limits clipping. Servo not responding — power, bus, operating_mode. -->

## Simulation

<!-- TODO: Gazebo model not spawning, sim_remapper topic names, PX4 SITL not arming. -->
