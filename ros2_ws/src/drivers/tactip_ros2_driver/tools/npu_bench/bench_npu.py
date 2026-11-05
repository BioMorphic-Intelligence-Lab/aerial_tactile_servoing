#!/usr/bin/env python3
"""Step 3 (Orange Pi 5, rknn-toolkit-lite2 + librknnrt): time NPU inference.

Single model, one NPU core:
    python3 bench_npu.py simple_cnn_B1/model_int8.rknn

Two sensors at once, each model pinned to its own NPU core (what the driver would do):
    python3 bench_npu.py simple_cnn_B1/model_int8.rknn simple_cnn_B2/model_int8.rknn --cores 0 1

CPU baseline with ONNX Runtime, pinned to the A76 big cores (check `lscpu -e`; usually 4-7):
    python3 bench_npu.py --onnx simple_cnn_B1/model.onnx --threads 4 --affinity 4-7

Timing covers the inference call only (input already a uint8 array), so it is comparable to the
driver's 'fwd' figure. Run it once on an idle board and once with the ROS stack running.
"""
import argparse
import os
import threading
import time

import numpy as np


def parse_cpus(spec):
    cpus = set()
    for part in spec.split(','):
        a, _, b = part.partition('-')
        cpus.update(range(int(a), int(b or a) + 1))
    return cpus


def stats(name, t):
    t = np.array(t) * 1e3
    print(f"{name}: n={len(t)}  mean {t.mean():.2f}  p50 {np.percentile(t, 50):.2f}  "
          f"p90 {np.percentile(t, 90):.2f}  p99 {np.percentile(t, 99):.2f}  max {t.max():.2f} ms")


def npu_runner(path, core):
    from rknnlite.api import RKNNLite
    masks = {'0': RKNNLite.NPU_CORE_0, '1': RKNNLite.NPU_CORE_1, '2': RKNNLite.NPU_CORE_2,
             'auto': RKNNLite.NPU_CORE_AUTO, 'all': RKNNLite.NPU_CORE_0_1_2}
    rk = RKNNLite(verbose=False)
    assert rk.load_rknn(path) == 0, f"load_rknn failed: {path}"
    assert rk.init_runtime(core_mask=masks[core]) == 0, "init_runtime failed (NPU driver / librknnrt?)"
    return rk


def run_loop(fn, x, warmup, iters, out, key):
    for _ in range(warmup):
        fn(x)
    t = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn(x)
        t.append(time.perf_counter() - t0)
    out[key] = t


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('models', nargs='*', help='.rknn file(s); two = run concurrently')
    ap.add_argument('--cores', nargs='*', default=None, help="NPU core per model: 0 1 2 auto all")
    ap.add_argument('--onnx', help='CPU baseline: run this .onnx with ONNX Runtime instead')
    ap.add_argument('--threads', type=int, default=4, help='ONNX Runtime intra-op threads')
    ap.add_argument('--affinity', help='CPU list to pin to, e.g. 4-7')
    ap.add_argument('--size', type=int, nargs=2, default=[128, 128], metavar=('H', 'W'))
    ap.add_argument('--warmup', type=int, default=50)
    ap.add_argument('--iters', type=int, default=1000)
    args = ap.parse_args()

    if args.affinity:
        # Applies to this thread and every thread created after it (ORT's pool, runner threads).
        os.sched_setaffinity(0, parse_cpus(args.affinity))
        print(f"Pinned to CPUs {sorted(os.sched_getaffinity(0))}")

    h, w = args.size
    rng = np.random.default_rng(0)
    img = (rng.random((h, w)) * 255).astype(np.uint8)

    if args.onnx:
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = args.threads
        sess = ort.InferenceSession(args.onnx, so, providers=['CPUExecutionProvider'])
        x = (img.astype(np.float32) / 255.0)[None, None]
        out = {}
        run_loop(lambda a: sess.run(None, {'image': a}), x, args.warmup, args.iters, out, 'cpu')
        stats(f"ORT CPU {os.path.basename(args.onnx)} ({args.threads} threads)", out['cpu'])
        return

    if not args.models:
        ap.error('give one or two .rknn files, or --onnx')
    cores = args.cores or (['0', '1', '2'][:len(args.models)] if len(args.models) > 1 else ['0'])
    if len(cores) != len(args.models):
        ap.error('--cores needs one entry per model')

    x = img[None, :, :, None]  # NHWC uint8; /255 is folded into the .rknn
    runners = [npu_runner(p, c) for p, c in zip(args.models, cores)]
    names = [f"NPU core {c} {os.path.basename(os.path.dirname(os.path.abspath(p)))}"
             for p, c in zip(args.models, cores)]

    out = {}
    if len(runners) == 1:
        run_loop(lambda a: runners[0].inference(inputs=[a]), x, args.warmup, args.iters, out, 0)
        stats(names[0], out[0])
    else:
        # Concurrent: both models hammer the NPU at once. If wall-clock throughput is ~2x a single
        # model, the cores really run in parallel (and the Python binding releases the GIL).
        threads = [threading.Thread(target=run_loop,
                                    args=(lambda a, r=r: r.inference(inputs=[a]), x,
                                          args.warmup, args.iters, out, i))
                   for i, r in enumerate(runners)]
        t0 = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        wall = time.perf_counter() - t0
        for i, n in enumerate(names):
            stats(n, out[i])
        total = len(runners) * (args.iters + args.warmup)
        print(f"Combined throughput: {total / wall:.0f} inferences/s "
              f"(camera needs ~{17.5 * len(runners):.0f}/s)")
    for r in runners:
        r.release()


if __name__ == '__main__':
    main()
