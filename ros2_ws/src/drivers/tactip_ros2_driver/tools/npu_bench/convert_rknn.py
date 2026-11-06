#!/usr/bin/env python3
"""Step 2 (laptop, x86 Linux, rknn-toolkit2): convert model.onnx to .rknn for the RK3588 NPU and
check the quantized model's accuracy in the RKNN simulator.

    python3 convert_rknn.py out/simple_cnn_B1            # int8 (default)
    python3 convert_rknn.py out/simple_cnn_B1 --fp16     # fp16, no quantization, for comparison

Writes out/<model>/model_int8.rknn (or model_fp16.rknn).

The /255 input scaling is folded into the model (mean 0, std 255), so on the board you feed the
H x W uint8 image straight in as NHWC (1, H, W, 1).

The accuracy check compares the simulator against ONNX Runtime (fp32) and reports the error per
output label in physical units (mm / deg / N), decoded like LabelEncoder.decode_label. By default
it uses the calibration images. With --val-dir (a tactile_data val_data directory: targets.csv +
processed_images/) it uses the held-out validation set instead and also scores both fp32 and the
quantized model against the ground-truth labels:

    python3 convert_rknn.py out/simple_cnn_B1 \
        --val-dir ~/tactip-main/tactile_data_shear/data/ur_aerial-B1/surface_9d/val_data
"""
import argparse
import csv
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
    ap.add_argument('--n-check', type=int, default=100,
                    help='images used for the accuracy check (0 = all)')
    ap.add_argument('--val-dir', help='held-out val_data dir (targets.csv + processed_images/)')
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
    model_name = os.path.basename(os.path.normpath(args.out_dir))
    paths, truth = check_images(args, dataset, model_name)

    ref, got = [], []
    for img in paths:
        ref.append(sess.run(None, {'image': (img.astype(np.float32) / 255.0)[None, None]})[0][0])
        got.append(np.asarray(rknn.inference(inputs=[img[None, :, :, None]], data_format='nhwc')[0])
                   .reshape(-1))
    ref, got = np.array(ref), np.array(got)
    rknn.release()

    print(f"\nRaw output max abs diff (normalized units): {np.abs(ref - got).max():.4f}")
    label_file = os.path.join(MODELS_DIR, model_name, 'model_label_params.json')
    if not os.path.exists(label_file):
        print(f"{label_file} not found; skipping per-label report")
        return
    with open(label_file) as f:
        decode = decoder(json.load(f))
    d_ref, d_got = decode(ref), decode(got)
    quant = 'fp16' if args.fp16 else 'int8'
    src = 'validation' if truth else 'calibration'
    print(f"\nPer-label error, {len(paths)} {src} images (mean abs; max abs in brackets):")
    hdr = f"  {'label':10s} {quant + ' vs fp32':>20s}"
    if truth:
        hdr += f" {'fp32 vs truth':>20s} {quant + ' vs truth':>20s} {'change':>8s}"
    print(hdr)
    for name in d_ref:
        line = f"  {name:10s} {fmt(err(name, d_got[name], d_ref[name])):>20s}"
        if truth:
            e_ref = err(name, d_ref[name], truth[name])
            e_got = err(name, d_got[name], truth[name])
            line += (f" {fmt(e_ref):>20s} {fmt(e_got):>20s}"
                     f" {100 * (e_got.mean() / e_ref.mean() - 1):+7.1f}%")
        print(line)


def err(name, a, b):
    e = a - b
    if name.startswith('pose_R'):  # wrap angle differences
        e = (e + 180) % 360 - 180
    return np.abs(e)


def fmt(e):
    return f"{e.mean():.4f} ({e.max():.3f})"


def check_images(args, dataset, model_name):
    """Images for the accuracy check as model-input uint8 arrays, plus ground truth if --val-dir."""
    n = args.n_check or None
    if not args.val_dir:
        with open(dataset) as f:
            files = [p.strip() for p in f if p.strip()][:n]
        return [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in files], None

    # Same preprocessing as LabelledModel.predict: processed image -> INTER_AREA resize to dims.
    with open(os.path.join(MODELS_DIR, model_name, 'model_image_params.json')) as f:
        dims = tuple(json.load(f)['image_processing']['dims'])
    with open(os.path.join(args.val_dir, 'targets.csv')) as f:
        rows = list(csv.DictReader(f))[:n]
    imgs, truth = [], {}
    for r in rows:
        p = os.path.join(args.val_dir, 'processed_images', os.path.basename(r['sensor_image']))
        imgs.append(cv2.resize(cv2.imread(p, cv2.IMREAD_GRAYSCALE), dims, interpolation=cv2.INTER_AREA))
        for k, v in r.items():
            if k not in ('sensor_image', 'object_label'):
                truth.setdefault(k, []).append(float(v))
    return imgs, {k: np.array(v) for k, v in truth.items()}


if __name__ == '__main__':
    main()
