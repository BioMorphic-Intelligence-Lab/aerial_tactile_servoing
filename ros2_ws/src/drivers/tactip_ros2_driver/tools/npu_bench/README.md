# npu_bench: convert TacTip models for the RK3588 NPU

Tools to turn a trained TacTip model (`resource/models/<model>/best_model.pth`) into the
`model_int8.rknn` the driver runs with `backend:=rknn`, check its accuracy, and benchmark it on the
Orange Pi. Why the NPU and how the model is quantized: see the driver [README](../../README.md).

| Script | Runs on | Does |
|---|---|---|
| `export_onnx.py` | laptop | `.pth` -> `model.onnx`, plus the calibration image list |
| `convert_rknn.py` | laptop | `model.onnx` -> `model_int8.rknn`, accuracy check in the RKNN simulator |
| `bench_npu.py` | Orange Pi | inference timing on the NPU (or CPU with `--onnx`) |

## One-time setup (laptop, x86)

```bash
python3 -m venv ~/rknn-env
source ~/rknn-env/bin/activate
pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch
pip install --no-cache-dir rknn-toolkit2==2.3.2 onnx==1.16.1 onnxruntime opencv-python-headless
```

`onnx` must stay at 1.16.1: rknn-toolkit2 2.3.2 uses `onnx.mapping`, which newer onnx removed.
The toolkit version must match the runtime on the Pi (`rknn-toolkit-lite2` and `librknnrt.so`
2.3.2, installed by the Docker image).

## Converting a model (after every retrain)

```bash
source ~/rknn-env/bin/activate
cd ~/aerial_tactile_servoing/ros2_ws/src/drivers/tactip_ros2_driver/tools/npu_bench
DATA=~/tactip-main/tactile_data_shear/data

# 1. export + calibration set: that sensor's TRAINING images (all of them by default)
python3 export_onnx.py simple_cnn_B1 --calib-dir $DATA/ur_aerial-B1/surface_9d/train_data/processed_images

# 2. convert + accuracy check on that sensor's held-out VALIDATION set
python3 convert_rknn.py out/simple_cnn_B1 --val-dir $DATA/ur_aerial-B1/surface_9d/val_data --n-check 0

# 3. install into the driver (then commit, pull on the Pi, colcon build)
cp out/simple_cnn_B1/model_int8.rknn ../../resource/models/simple_cnn_B1/
```

For B2 use `simple_cnn_B2` and `ur_aerial-B2`. The first argument of `export_onnx.py` and the
`out/<model>` folder name must match the folder in `resource/models/` (label limits and image
size are read from there).

**Check the table step 2 prints.** Columns: quantized vs fp32, fp32 vs ground truth, quantized vs
ground truth, and the relative change. Expect the change to be within about 2% on every label. A
large change means the conversion lost accuracy; do not install that model.

### convert_rknn.py options

| Option | Builds | Output |
|---|---|---|
| (default) | int8 convolutions + fp16 fully connected head: **use this** | `model_int8.rknn` |
| `--head int8` | all int8; clips every output at the training-data maximum | `model_int8.rknn` |
| `--fp16` | all fp16, no quantization, comparison only | `model_fp16.rknn` (not loaded by the driver) |

Other options: `--n-check N` (images in the accuracy check, 0 = all), `--val-dir` (without it the
check runs on the calibration images and has no ground truth).

`export_onnx.py`: `--n-calib N` limits the calibration set to a random N images (default 0 = all;
a subset can miss the extremes). Without `--calib-dir` it uses synthetic images: fine for timing,
meaningless for accuracy.

## Checking on the Orange Pi (inside the Docker container)

```bash
cd /aerial_tactile_servoing/ros2_ws/src/drivers/tactip_ros2_driver

# runtime reachable?
python3 -c "from rknnlite.api import RKNNLite; print('rknnlite ok')"

# timing, both models taking turns on NPU core 0 (as the driver runs them)
python3 tools/npu_bench/bench_npu.py \
    resource/models/simple_cnn_B1/model_int8.rknn \
    resource/models/simple_cnn_B2/model_int8.rknn --cores 0 0 --sequential
```

Then run the driver with `backend:=rknn test_model_time:=true` and check the log shows
`backend: rknn` for each sensor and both publish at the camera rate (~17 Hz).

## Known limitations

- Only NPU core 0 works on the current Orange Pi kernel (rknpu driver 0.9.3). Cores 1 and 2 hang
  with `failed to submit!`, even with a single model. Always use `--cores 0` / `0 0`.
- The startup warnings `query RKNN_QUERY_INPUT_DYNAMIC_RANGE error` and `Query dynamic range
  failed` are expected for fixed-input-size models and can be ignored.
- `Unkown op target: 0` during conversion is harmless.
