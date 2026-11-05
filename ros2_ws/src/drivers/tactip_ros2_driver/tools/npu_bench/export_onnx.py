#!/usr/bin/env python3
"""Step 1 (laptop): export a TacTip model to ONNX and build an RKNN calibration set.

    python3 export_onnx.py simple_cnn_B1 --calib-dir /path/to/processed/images

Writes to ./out/<model>/:
    model.onnx      float32, input 'image' (1, 1, H, W) already scaled to [0, 1]
    calib/*.png     H x W uint8 images, exactly what the model sees before the /255
    dataset.txt     list of calib images, for RKNN quantization

--calib-dir should hold processed tactile images (the thresholded frames saved during data
collection, or frames saved with TacTip.process(proc_outfile=...)). They are resized with
INTER_AREA like LabelledModel does. Without it, synthetic images are generated: fine for timing,
meaningless for an accuracy check.
"""
import argparse
import glob
import importlib.util
import json
import os
import sys
import types

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))
MODELS_DIR = os.path.join(PKG_ROOT, 'resource', 'models')
MODELS_PY = os.path.join(PKG_ROOT, 'tactip_ros2_driver', 'dependencies', 'models.py')


def load_models_module():
    """Import the driver's models.py standalone. It imports pytorch_model_summary and vit_pytorch
    at top level (model printout / ViT only); stub them if they are not installed."""
    for name in ['pytorch_model_summary', 'vit_pytorch', 'vit_pytorch.vit']:
        try:
            __import__(name)
        except ImportError:
            stub = types.ModuleType(name)
            stub.summary = lambda *a, **k: ''
            stub.ViT = None
            sys.modules[name] = stub
    spec = importlib.util.spec_from_file_location('tactip_models', MODELS_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_model(model_dir):
    with open(os.path.join(model_dir, 'model_params.json')) as f:
        model_params = json.load(f)
    with open(os.path.join(model_dir, 'model_image_params.json')) as f:
        dims = json.load(f)['image_processing']['dims']
    state = torch.load(os.path.join(model_dir, 'best_model.pth'), map_location='cpu', weights_only=True)
    # out_dim from the last Linear layer (periodic labels take two outputs, so not len(targets))
    fc_weights = sorted((k for k in state if k.startswith('fc.') and k.endswith('.weight')),
                        key=lambda k: int(k.split('.')[1]))
    out_dim = state[fc_weights[-1]].shape[0]
    model = load_models_module().create_model(
        in_dim=dims, in_channels=1, out_dim=out_dim, model_params=model_params,
        saved_model_dir=model_dir, display_model=False, device='cpu')
    model.eval()
    return model, dims, out_dim


def make_calib(out_dir, dims, calib_dir, n_max):
    calib_out = os.path.join(out_dir, 'calib')
    os.makedirs(calib_out, exist_ok=True)
    w, h = dims
    paths = []
    if calib_dir:
        files = sorted(p for ext in ('png', 'jpg', 'jpeg', 'bmp')
                       for p in glob.glob(os.path.join(calib_dir, '**', f'*.{ext}'), recursive=True))
        if not files:
            sys.exit(f"No images found in {calib_dir}")
        rng = np.random.default_rng(0)
        files = [files[i] for i in sorted(rng.choice(len(files), min(n_max, len(files)), replace=False))]
        for i, src in enumerate(files):
            img = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            paths.append(os.path.join(calib_out, f'{i:04d}.png'))
            cv2.imwrite(paths[-1], img)
    else:
        print("WARNING: no --calib-dir, using synthetic images. OK for timing, NOT for accuracy.")
        rng = np.random.default_rng(0)
        for i in range(min(n_max, 50)):
            full = (rng.random((480, 640)) > 0.5).astype(np.uint8) * 255
            img = cv2.resize(full, (w, h), interpolation=cv2.INTER_AREA)
            paths.append(os.path.join(calib_out, f'{i:04d}.png'))
            cv2.imwrite(paths[-1], img)
    with open(os.path.join(out_dir, 'dataset.txt'), 'w') as f:
        f.write('\n'.join(os.path.abspath(p) for p in paths) + '\n')
    return paths


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('model', help='model directory name under resource/models, or a path')
    ap.add_argument('--calib-dir', help='directory of processed tactile images (recursive)')
    ap.add_argument('--n-calib', type=int, default=300, help='max calibration images')
    ap.add_argument('--out', default=os.path.join(HERE, 'out'))
    ap.add_argument('--opset', type=int, default=13)
    args = ap.parse_args()

    model_dir = args.model if os.path.isdir(args.model) else os.path.join(MODELS_DIR, args.model)
    name = os.path.basename(os.path.normpath(model_dir))
    out_dir = os.path.join(args.out, name)
    os.makedirs(out_dir, exist_ok=True)

    model, dims, out_dim = load_model(model_dir)
    w, h = dims
    x = torch.zeros(1, 1, h, w)
    onnx_path = os.path.join(out_dir, 'model.onnx')
    kwargs = dict(input_names=['image'], output_names=['out'], opset_version=args.opset,
                  do_constant_folding=True)
    try:
        torch.onnx.export(model, x, onnx_path, dynamo=False, **kwargs)
    except TypeError:  # torch < 2.1 has no dynamo argument
        torch.onnx.export(model, x, onnx_path, **kwargs)
    print(f"Exported {name}: input (1, 1, {h}, {w}), {out_dim} outputs -> {onnx_path}")

    paths = make_calib(out_dir, dims, args.calib_dir, args.n_calib)
    print(f"Wrote {len(paths)} calibration images and dataset.txt to {out_dir}")

    # Sanity check: ONNX Runtime must reproduce PyTorch (fp32, so ~1e-6)
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed; skipping PyTorch vs ONNX check")
        return
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    worst = 0.0
    with torch.inference_mode():
        for p in paths[:20]:
            inp = (cv2.imread(p, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0)[None, None]
            ref = model(torch.from_numpy(inp)).numpy()
            got = sess.run(None, {'image': inp})[0]
            worst = max(worst, float(np.abs(ref - got).max()))
    print(f"PyTorch vs ONNX Runtime max abs diff over {min(20, len(paths))} images: {worst:.2e}")


if __name__ == '__main__':
    main()
