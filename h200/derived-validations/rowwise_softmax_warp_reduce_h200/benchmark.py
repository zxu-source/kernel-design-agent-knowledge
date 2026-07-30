#!/usr/bin/env python3
"""
benchmark.py — Full benchmark harness for FP16 row-wise softmax.

Measures baseline, warp-shuffle candidate, and PyTorch F.softmax on all
8 shapes with >=20 warmups and >=100 CUDA-event-timed iterations per group
across 3 groups. Reports medians, min/max, effective bandwidth, custom-to-
custom speedup, and versus-torch results.

Usage:
    python benchmark.py [--json outputs/benchmark.json] [--csv outputs/benchmark.csv]
"""

import os, sys, json, math, ctypes, argparse, time
import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Load shared library
# ---------------------------------------------------------------------------
SO_PATH = os.path.join(os.path.dirname(__file__), "softmax.so")

lib = ctypes.CDLL(SO_PATH)

lib.launch_softmax_baseline.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
lib.launch_softmax_baseline.restype = None

lib.launch_softmax_warp_shuffle.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
lib.launch_softmax_warp_shuffle.restype = None

lib.allocate_device.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
lib.allocate_device.restype = ctypes.c_int
lib.free_device.argtypes = [ctypes.c_void_p]
lib.free_device.restype = ctypes.c_int
lib.copy_host_to_device.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
lib.copy_host_to_device.restype = ctypes.c_int


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------
SHAPES = [
    (512, 512),
    (1024, 1024),
    (2048, 2048),
    (4096, 4096),
    (8192, 8192),
    (512, 777),
    (768, 3072),
    (1000, 513),
]


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------
def effective_bandwidth_gbs(rows, cols, median_ms):
    bytes_count = 2.0 * rows * cols * 2  # read + write, sizeof(half)=2
    return bytes_count / (median_ms * 1e6)


def benchmark_custom_kernel(kernel_fn, d_input, d_output, rows, cols,
                            warmups=25, iters=120, groups=3):
    """Benchmark a custom kernel using CUDA events."""
    count = rows * cols

    # Allocate device memory
    dev_in = ctypes.c_void_p()
    dev_out = ctypes.c_void_p()
    lib.allocate_device(ctypes.byref(dev_in), count)
    lib.allocate_device(ctypes.byref(dev_out), count)
    lib.copy_host_to_device(ctypes.c_void_p(d_input.data_ptr()), dev_in, count)

    all_times = []

    for g in range(groups):
        # Warmup
        for _ in range(warmups):
            kernel_fn(dev_in, dev_out, rows, cols, None)
        torch.cuda.synchronize()

        # Timed iterations
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)

        for _ in range(iters):
            start_ev.record()
            kernel_fn(dev_in, dev_out, rows, cols, None)
            end_ev.record()
            torch.cuda.synchronize()
            all_times.append(start_ev.elapsed_time(end_ev))

        torch.cuda.synchronize()

    lib.free_device(dev_in)
    lib.free_device(dev_out)

    all_times.sort()
    n = len(all_times)
    return {
        "median_ms": all_times[n // 2],
        "min_ms": all_times[0],
        "max_ms": all_times[-1],
        "all_times_ms": all_times,
    }


def benchmark_torch(input_tensor, warmups=25, iters=120, groups=3):
    """Benchmark torch F.softmax using identical event methodology."""
    all_times = []

    for g in range(groups):
        # Warmup
        for _ in range(warmups):
            _ = F.softmax(input_tensor.float(), dim=1)
        torch.cuda.synchronize()

        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)

        for _ in range(iters):
            start_ev.record()
            _ = F.softmax(input_tensor.float(), dim=1)
            end_ev.record()
            torch.cuda.synchronize()
            all_times.append(start_ev.elapsed_time(end_ev))

        torch.cuda.synchronize()

    all_times.sort()
    n = len(all_times)
    return {
        "median_ms": all_times[n // 2],
        "min_ms": all_times[0],
        "max_ms": all_times[-1],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmups", type=int, default=25)
    parser.add_argument("--iters", type=int, default=120)
    parser.add_argument("--groups", type=int, default=3)
    parser.add_argument("--json", type=str, default="")
    parser.add_argument("--csv", type=str, default="")
    args = parser.parse_args()

    print(f"=== FP16 Row-wise Softmax Benchmark ===")
    print(f"Warmups: {args.warmups}, Iters: {args.iters}, Groups: {args.groups}")
    print(f"Shapes: {len(SHAPES)}\n")

    all_results = []

    for rows, cols in SHAPES:
        print(f"--- Shape [{rows}, {cols}] ---")

        # Generate input once
        torch.manual_seed(42)
        h_input = torch.empty(rows, cols, dtype=torch.float16).uniform_(-1, 1)
        d_input = h_input.cuda()

        # Benchmark baseline
        print("  Baseline...", end=" ", flush=True)
        bl = benchmark_custom_kernel(
            lib.launch_softmax_baseline, h_input, None, rows, cols,
            args.warmups, args.iters, args.groups)
        bw_bl = effective_bandwidth_gbs(rows, cols, bl["median_ms"])
        print(f"{bl['median_ms']:.4f} ms  ({bw_bl:.1f} GB/s)")

        # Benchmark candidate
        print("  Candidate...", end=" ", flush=True)
        ws = benchmark_custom_kernel(
            lib.launch_softmax_warp_shuffle, h_input, None, rows, cols,
            args.warmups, args.iters, args.groups)
        bw_ws = effective_bandwidth_gbs(rows, cols, ws["median_ms"])
        print(f"{ws['median_ms']:.4f} ms  ({bw_ws:.1f} GB/s)")

        # Benchmark torch
        print("  Torch...", end=" ", flush=True)
        th = benchmark_torch(d_input, args.warmups, args.iters, args.groups)
        bw_th = effective_bandwidth_gbs(rows, cols, th["median_ms"])
        print(f"{th['median_ms']:.4f} ms  ({bw_th:.1f} GB/s)")

        speedup_ws_vs_bl = bl["median_ms"] / ws["median_ms"]
        speedup_bl_vs_th = bl["median_ms"] / th["median_ms"]
        speedup_ws_vs_th = ws["median_ms"] / th["median_ms"]

        print(f"  Speedup: warp/baseline={speedup_ws_vs_bl:.3f}x  "
              f"baseline/torch={speedup_bl_vs_th:.3f}x  "
              f"warp/torch={speedup_ws_vs_th:.3f}x\n")

        all_results.append({
            "shape": [rows, cols],
            "baseline": {
                "median_ms": bl["median_ms"],
                "min_ms": bl["min_ms"],
                "max_ms": bl["max_ms"],
                "bandwidth_gbs": bw_bl,
            },
            "c001_warp_shuffle": {
                "median_ms": ws["median_ms"],
                "min_ms": ws["min_ms"],
                "max_ms": ws["max_ms"],
                "bandwidth_gbs": bw_ws,
            },
            "torch": {
                "median_ms": th["median_ms"],
                "min_ms": th["min_ms"],
                "max_ms": th["max_ms"],
                "bandwidth_gbs": bw_th,
            },
            "candidate_to_baseline_speedup": speedup_ws_vs_bl,
            "baseline_to_torch_speedup": speedup_bl_vs_th,
            "candidate_to_torch_speedup": speedup_ws_vs_th,
        })

        del h_input, d_input
        torch.cuda.empty_cache()

    # Summary
    print("=" * 70)
    print(f"{'Shape':<16} {'Baseline(ms)':<14} {'Warp(ms)':<14} {'Torch(ms)':<14} {'WS/BL':<8} {'WS/TH':<8}")
    print("-" * 70)
    for r in all_results:
        s = f"[{r['shape'][0]},{r['shape'][1]}]"
        print(f"{s:<16} {r['baseline']['median_ms']:<14.4f} {r['c001_warp_shuffle']['median_ms']:<14.4f} "
              f"{r['torch']['median_ms']:<14.4f} {r['candidate_to_baseline_speedup']:<8.3f} "
              f"{r['candidate_to_torch_speedup']:<8.3f}")

    # Write outputs
    if args.json:
        with open(args.json, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nJSON written to {args.json}")

    if args.csv:
        with open(args.csv, "w") as f:
            f.write("shape_rows,shape_cols,baseline_ms,candidate_ms,torch_ms,"
                    "candidate_to_baseline_speedup,candidate_to_torch_speedup,"
                    "baseline_bw_gbs,candidate_bw_gbs,torch_bw_gbs\n")
            for r in all_results:
                f.write(f"{r['shape'][0]},{r['shape'][1]},"
                        f"{r['baseline']['median_ms']:.6f},"
                        f"{r['c001_warp_shuffle']['median_ms']:.6f},"
                        f"{r['torch']['median_ms']:.6f},"
                        f"{r['candidate_to_baseline_speedup']:.4f},"
                        f"{r['candidate_to_torch_speedup']:.4f},"
                        f"{r['baseline']['bandwidth_gbs']:.3f},"
                        f"{r['c001_warp_shuffle']['bandwidth_gbs']:.3f},"
                        f"{r['torch']['bandwidth_gbs']:.3f}\n")
        print(f"CSV written to {args.csv}")


if __name__ == "__main__":
    main()
