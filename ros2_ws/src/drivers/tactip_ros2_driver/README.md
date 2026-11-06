# tactip_ros2_driver

ROS 2 interface for the TacTip optical tactile sensor. Captures the sensor's internal camera,
processes each frame, runs a PyTorch pose-estimation model, and decides contact by comparing the
live image to a stored reference with structural similarity (SSIM). See
[docs/07-tactile-sensing.md](../../../../docs/07-tactile-sensing.md).

## Run

```bash
ros2 launch ats_bringup tactip.launch.py    # sensor only, for bring-up/testing
```

The sensor logic lives in `tactip.py` (capture + processing) and `tactip_ros2_driver.py` (the
node); model definitions and the label encoder are under `dependencies/`.

## Pipeline

One capture thread per camera reads frames, runs the image processing and the SSIM contact check,
and keeps only the newest result; a single inference worker wakes on every new frame and runs the
model (when in contact) and publishes. Output therefore follows the camera rate (~17 Hz per tip on
the current cameras), each frame is predicted at most once, and messages are stamped with the
frame's capture time. Inference is kept serial in one process: running the tips in separate
PyTorch processes oversubscribes the CPU. SSIM uses `fast_ssim.py`, an OpenCV implementation that
matches `skimage.metrics.structural_similarity` (~1e-9) at about a quarter of the cost.

## NPU backend

With `backend:=rknn` each sensor runs the quantized `model_int8.rknn` from its model directory on
NPU core 0 (cores 1 and 2 hang on the current Orange Pi kernel). The models take turns from the
single inference worker, which is what the NPU driver needs. For B1 + B2 a full round takes about
7.5 ms on the NPU (3.8 ms per model), against about 85 ms with PyTorch on the CPU, so both sensors
publish at the camera rate (~17 Hz) and the CPU stays free for the rest of the stack. The NPU
models are within about 2% of fp32 accuracy on the validation sets.

`tactip_dual_bringup.launch.py` uses the NPU by default; `backend:=torch` switches back to PyTorch.

The `.rknn` keeps the convolutions in int8 but runs the fully connected head in fp16. RKNN uses one
quantization scale per tensor, fitted to the calibration images, so an all-int8 head clips every
output at the largest value seen in the training data (e.g. depth stuck at 3.24 mm), whereas
contacts in flight go beyond it. The fp16 head extrapolates like the PyTorch model.

A retrained model needs a new `.rknn`. The conversion workflow (setup, commands, how to read the
accuracy check, checks on the Pi) is in [tools/npu_bench/README.md](tools/npu_bench/README.md).

## Parameters

Per-sensor parameters are lists with one entry per TacTip.

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `sources` | [0] | Camera indices (`/dev/videoN`) |
| `model_names` | — | Model directory per sensor, under `share/tactip_ros2_driver/models/` |
| `dimensions` | [8] | Model output size per sensor (>6 also publishes a wrench) |
| `ssim_contact_thresholds` | [0.9] | Contact asserted below this SSIM |
| `frequency` | 0.0 | Max publish rate per sensor (Hz); 0 = every camera frame |
| `zero_when_no_contact` | true | Publish zero pose (and skip the model) when not in contact |
| `backend` | torch | `torch` (PyTorch, CPU) or `rknn` (`model_int8.rknn` on the RK3588 NPU, falls back to `torch` if it cannot load) |
| `test_model_time` | false | Log grab/publish rates, per-stage times and frame age every 5 s |
| `verbose` | true | Log every measurement |
| `fake_data` | false | Bypass the cameras and publish synthetic data |

## Topics

`N` is the sensor's index in `sources`.

| Direction | Topic | Type |
|-----------|-------|------|
| pub | `/tactip_N/pose` | `geometry_msgs/TwistStamped` |
| pub | `/tactip_N/wrench` | `geometry_msgs/WrenchStamped` (if `dimensions` > 6) |
| pub | `/tactip_N/contact` | `std_msgs/Int8` |
| pub | `/tactip_N/ssim` | `std_msgs/Float64` |
| tf | `present_sensor_N` → `present_contact_N` | |

## Services

| Service | Type | Purpose |
|---------|------|---------|
| `set_ssim_ref` | `std_srvs/SetBool` | Use each sensor's next frame as its SSIM contact reference |

## Notes

- The model path is logged at startup (`BASE_MODEL_PATH` from
  `dependencies/label_encoder.py`) — check it if `/tactip_N/pose` is missing.
- `/tactip_N/pose`: linear = shear x/y and depth z (mm); angular = surface orientation.
- The mission director calls `set_ssim_ref` before approach so the reference matches current
  lighting.
