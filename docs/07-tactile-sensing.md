# Tactile sensing

> **Status: scaffold.** Fill in the model details and the retraining workflow.

The TacTip is a soft optical tactile sensor: an internal camera watches the deformation of a
dome of markers. The [`tactip_ros2_driver`](../ros2_ws/src/drivers/tactip_ros2_driver/README.md)
turns each camera frame into a contact pose estimate and a contact flag.

## Pipeline

```
camera frame → image processing → pose model (PyTorch) → /tactip/pose
                       └────────→ SSIM vs reference     → /tactip/contact
```

Outputs:
- `/tactip/pose` (`TwistStamped`) — linear = shear x/y and depth z (mm); angular = surface
  orientation.
- `/tactip/contact` (`Int8`) — 1 in contact, 0 otherwise.
- `/tactip/ssim` (`Float64`) — the raw structural-similarity score used for the contact decision.

## The pose model

The pose estimator is a neural network (the `dependencies/` folder includes CNN and ViT model
definitions and a label encoder). The `dimension` parameter selects how many pose components the
model outputs.

<!-- TODO: document the model architecture actually used in flight, where the weights live
     (BASE_MODEL_PATH in dependencies/label_encoder.py), the training data format, and the
     label encoding. Reference the hyperparameter-optimized "A1" model from the git history. -->
<!-- TODO: explain the sensor frame directions (see commit "Verified tactip frame directions")
     and how pose components map to the contact frame. Cross-link to 03-system-architecture.md. -->

## Contact detection (SSIM)

At startup (or when the mission director calls the `set_ssim_ref` service) the driver stores a
**reference image** of the undeformed sensor. Each new frame is compared to it with structural
similarity; when similarity drops below `ssim_contact_threshold`, contact is asserted.

<!-- TODO: document threshold tuning, the zero_when_no_contact behaviour, and why the reference
     is reset before each approach rather than baked in. -->

## Driver parameters

Set in the launch files (e.g. `vbats_mission.launch.py`):

| Parameter | Meaning |
|-----------|---------|
| `source` | `/dev/videoN` camera index |
| `frequency` | Processing rate (Hz) |
| `dimension` | Number of pose components the model outputs |
| `ssim_contact_threshold` | Contact SSIM threshold |
| `zero_when_no_contact` | Publish zero pose when not in contact |
| `save_debug_image` / `save_interval` / `save_directory` | Periodic debug image dumps |
| `fake_data` | Bypass the camera and publish synthetic data (for sim/testing) |

## Swapping or retraining the model

<!-- TODO: step-by-step: collect labelled data, train, export weights, point BASE_MODEL_PATH at
     the new model, validate. Note dependencies on vit-pytorch / torch versions in
     requirements.txt. -->

## Standalone sensor testing

Bring up only the sensor to check it before a flight:

```bash
ros2 launch ats_bringup tactip.launch.py
```

<!-- TODO: describe what to look for (pose sign conventions, contact toggling, debug images in
     data/tactip_images). -->
