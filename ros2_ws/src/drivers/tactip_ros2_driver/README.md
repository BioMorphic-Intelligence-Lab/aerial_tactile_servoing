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

## Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `source` | 0 | Camera index (`/dev/videoN`) |
| `frequency` | 10.0 | Processing/publish rate (Hz) |
| `dimension` | 5 | Number of pose components the model outputs |
| `verbose` | true | Verbose logging |
| `test_model_time` | false | Log model inference time |
| `save_debug_image` | false | Periodically save processed frames |
| `save_interval` | 1.0 | Seconds between debug-image saves |
| `save_directory` | — | Where debug images go (set in launch file) |
| `ssim_contact_threshold` | 0.7 | Contact asserted below this SSIM |
| `zero_when_no_contact` | true | Publish zero pose when not in contact |
| `fake_data` | false | Bypass the camera and publish synthetic data |

## Topics

| Direction | Topic | Type |
|-----------|-------|------|
| pub | `/tactip/pose` | `geometry_msgs/TwistStamped` |
| pub | `/tactip/contact` | `std_msgs/Int8` |
| pub | `/tactip/ssim` | `std_msgs/Float64` |

## Services

| Service | Type | Purpose |
|---------|------|---------|
| `set_ssim_ref` | `std_srvs/SetBool` | Capture the current frame as the SSIM contact reference |

## Notes

- The model path is logged at startup (`BASE_MODEL_PATH` from
  `dependencies/label_encoder.py`) — check it if `/tactip/pose` is missing.
- `/tactip/pose`: linear = shear x/y and depth z (mm); angular = surface orientation.
- The mission director calls `set_ssim_ref` before approach so the reference matches current
  lighting.
