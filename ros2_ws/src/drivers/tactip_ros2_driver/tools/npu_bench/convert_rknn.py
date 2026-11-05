#!/usr/bin/env python3
"""Step 2 (laptop, x86 Linux, rknn-toolkit2): convert model.onnx to .rknn for the RK3588 NPU and
check the quantized model's accuracy in the RKNN simulator.

    python3 convert_rknn.py out/simple_cnn_B1            # int8 (default)
    python3 convert_rknn.py out/simple_cnn_B1 --fp16     # fp16, no quantization, for comparison

Writes out/<model>/model_int8.rknn (or model_fp16.rknn).

The /255 input scaling is folded into the model (mean 0, std 255), so on the board you feed the
H x W uint8 image straight in as NHWC (1, H, W, 1).

The accuracy check compares the simulator against ONNX Runtime (fp32) on the calibration images
and reports the error per output label in physical units (mm / deg / N), decoded like
LabelEncoder.decode_label. It is only meaningful with real calibration images (--calib-dir in
step 1), and ideally with held-out images rather than the calibration set itself.
"""
import argparse
import json
import os

import cv2
import numpy as np
from rknn.api import RKNN

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.normpath(os.path.join(HERE, '..', '..', 'resource', 'models'))


def decoder(label_params):
    """Map raw network outputs (N, out_dim) to {label: physical value}, as LabelEncoder does."""
    names, ll, ul = label_params['label_names'], label_params['llims'], label_params['ulims']
    periodic = set(label_params.get('periodic_label_names', []))
    weights = label_params.get('target_weights', [1] * len(label_params['target_label_names']))

    def decode(out):
        res, i = {}, 0
        for name, wgt in zip(label_params['target_label_names'], weights):
            if name in periodic:
                res[name] = np.degrees(np.arctan2(out[:, i] / wgt, out[:, i + 1] / wgt))
                i += 2
            else:
                k = names.index(name)
                res[name] = (out[:, i] / wgt + 1) / 2 * (ul[k] - ll[k]) + ll[k]
                i += 1
        return res
    return decode


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('out_dir', help='output directory of export_onnx.py for one model')
    ap.add_argument('--fp16', action='store_true', help='no quantization (fp16 on the NPU)')
    ap.add_argument('--n-check', type=int, default=100, help='images used for the accuracy check')
    args = ap.parse_args()

    onnx_path = os.path.join(args.out_dir, 'model.onnx')
    dataset = os.path.join(args.out_dir, 'dataset.txt')
    rknn_path = os.path.join(args.out_dir, 'model_fp16.rknn' if args.fp16 else 'model_int8.rknn')

    rknn = RKNN(verbose=False)
    rknn.config(mean_values=[[0]], std_values=[[255]], target_platform='rk3588')
    assert rknn.load_onnx(model=onnx_path) == 0, "load_onnx failed"
    assert rknn.build(do_quantization=not args.fp16, dataset=dataset) == 0, "build failed"
    assert rknn.export_rknn(rknn_path) == 0, "export_rknn failed"
    print(f"Wrote {rknn_path}")

    # --- accuracy check in the simulator (init_runtime without target = simulator) ---
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed; skipping accuracy check")
        return
    assert rknn.init_runtime() == 0, "simulator init failed"
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    with open(dataset) as f:
        paths = [p.strip() for p in f if p.strip()][:args.n_check]

    ref, got = [], []
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        ref.append(sess.run(None, {'image': (img.astype(np.float32) / 255.0)[None, None]})[0][0])
        got.append(np.asarray(rknn.inference(inputs=[img[None, :, :, None]], data_format='nhwc')[0])
                   .reshape(-1))
    ref, got = np.array(ref), np.array(got)
    rknn.release()

    print(f"\nRaw output max abs diff (normalized units): {np.abs(ref - got).max():.4f}")
    model_name = os.path.basename(os.path.normpath(args.out_dir))
    label_file = os.path.join(MODELS_DIR, model_name, 'model_label_params.json')
    if not os.path.exists(label_file):
        print(f"{label_file} not found; skipping per-label report")
        return
    with open(label_file) as f:
        decode = decoder(json.load(f))
    d_ref, d_got = decode(ref), decode(got)
    print(f"\nPer-label error, simulator vs fp32, {len(paths)} images:")
    print(f"  {'label':10s} {'mean abs':>10s} {'max abs':>10s} {'fp32 range':>22s}")
    for name in d_ref:
        err = d_got[name] - d_ref[name]
        if name.startswith('pose_R'):  # wrap angle differences
            err = (err + 180) % 360 - 180
        err = np.abs(err)
        print(f"  {name:10s} {err.mean():10.4f} {err.max():10.4f}"
              f"   [{d_ref[name].min():8.3f}, {d_ref[name].max():8.3f}]")


if __name__ == '__main__':
    main()
