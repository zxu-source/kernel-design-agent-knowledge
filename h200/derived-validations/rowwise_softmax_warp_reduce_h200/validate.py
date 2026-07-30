#!/usr/bin/env python3
"""
validate.py — Correctness test for H200 FP16 row-wise softmax kernels.

Tests both baseline and warp-shuffle kernels on 8 shapes × 4 input types
against PyTorch F.softmax reference.

Usage: python validate.py [--kernel baseline|c001_warp_shuffle|both]
"""

import os, sys, json, math, ctypes, argparse
import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Load shared library
# ---------------------------------------------------------------------------
SO_PATH = os.path.join(os.path.dirname(__file__), "softmax.so")

lib = ctypes.CDLL(SO_PATH)

# cudaStream_t = void*
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

lib.copy_device_to_host.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
lib.copy_device_to_host.restype = ctypes.c_int


# ---------------------------------------------------------------------------
# Test shapes (5 regular + 3 irregular)
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
# Input generators
# ---------------------------------------------------------------------------
def make_random(rows, cols):
    """Uniform random in [-1, 1]."""
    return torch.empty(rows, cols, dtype=torch.float16).uniform_(-1, 1)

def make_large_signed(rows, cols):
    """Values near FP16 extremes."""
    t = torch.empty(rows, cols, dtype=torch.float32)
    t.uniform_(-65504, 65504)
    return t.clamp(-65504, 65504).to(torch.float16)

def make_identical_row(rows, cols):
    """All rows are identical (tests reduction determinism)."""
    row = torch.empty(1, cols, dtype=torch.float16).uniform_(-1, 1)
    return row.expand(rows, -1).contiguous()

def make_fp16_extremes(rows, cols):
    """Mix of FP16 boundary values."""
    choices = torch.tensor(
        [65504.0, -65504.0, 6.1e-5, -6.1e-5, 0.0, 1.0, -1.0, 100.0, -100.0],
        dtype=torch.float32)
    idx = torch.randint(0, len(choices), (rows, cols))
    return choices[idx].to(torch.float16)


INPUT_GENERATORS = {
    "random": make_random,
    "large_signed": make_large_signed,
    "identical_row": make_identical_row,
    "fp16_extremes": make_fp16_extremes,
}


# ---------------------------------------------------------------------------
# Kernel wrappers
# ---------------------------------------------------------------------------
def _launch(kernel_fn, h_input, rows, cols):
    """Copy input to device, launch kernel, copy output back."""
    count = rows * cols
    # Allocate
    d_input = ctypes.c_void_p()
    d_output = ctypes.c_void_p()
    lib.allocate_device(ctypes.byref(d_input), count)
    lib.allocate_device(ctypes.byref(d_output), count)
    # Copy in
    lib.copy_host_to_device(
        ctypes.c_void_p(h_input.data_ptr()),
        d_input, count)
    # Launch
    kernel_fn(d_input, d_output, rows, cols, None)
    torch.cuda.synchronize()
    # Copy out
    h_output = torch.empty(rows, cols, dtype=torch.float16)
    lib.copy_device_to_host(
        d_output, ctypes.c_void_p(h_output.data_ptr()), count)
    # Free
    lib.free_device(d_input)
    lib.free_device(d_output)
    return h_output


def run_baseline(input_tensor):
    rows, cols = input_tensor.shape
    return _launch(lib.launch_softmax_baseline, input_tensor, rows, cols)

def run_warp_shuffle(input_tensor):
    rows, cols = input_tensor.shape
    return _launch(lib.launch_softmax_warp_shuffle, input_tensor, rows, cols)


# ---------------------------------------------------------------------------
# Torch reference
# ---------------------------------------------------------------------------
def torch_softmax(input_tensor):
    """Reference: upcast to FP32, compute softmax, downcast to FP16."""
    return F.softmax(input_tensor.float(), dim=1).to(torch.float16)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate(kernel_fn, kernel_name, input_tensor, torch_out, verbose=True):
    """Compare kernel output against torch reference."""
    custom_out = kernel_fn(input_tensor)
    torch.cuda.synchronize()

    custom_f32 = custom_out.float().cpu()
    torch_f32  = torch_out.float().cpu()

    diff = (custom_f32 - torch_f32).abs()
    max_abs_err = diff.max().item()

    denom = torch_f32.abs().clamp(min=1e-8)
    rel_err = diff / denom
    max_rel_err = rel_err.max().item()

    nan_count  = torch.isnan(custom_f32).sum().item()
    inf_count  = torch.isinf(custom_f32).sum().item()

    # Row-sum deviation
    row_sums = custom_f32.sum(dim=1)
    row_sum_dev = (row_sums - 1.0).abs().max().item()

    passed = (max_abs_err < 5e-3 and max_rel_err < 1e-2 and
              nan_count == 0 and inf_count == 0 and row_sum_dev < 5e-4)

    if verbose:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {kernel_name}: "
              f"max_abs={max_abs_err:.6f} max_rel={max_rel_err:.6f} "
              f"NaN={nan_count} Inf={inf_count} row_sum_dev={row_sum_dev:.6e}")

    return {
        "kernel": kernel_name,
        "max_abs_err": max_abs_err,
        "max_rel_err": max_rel_err,
        "nan_count": int(nan_count),
        "inf_count": int(inf_count),
        "row_sum_dev": row_sum_dev,
        "passed": passed,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", choices=["baseline", "c001_warp_shuffle", "both"],
                        default="both")
    parser.add_argument("--input-type", choices=list(INPUT_GENERATORS.keys()) + ["all"],
                        default="all")
    parser.add_argument("--shape", type=str, default="all",
                        help="Comma-separated rows,cols or 'all'")
    parser.add_argument("--json", type=str, default="",
                        help="Output JSON file path")
    args = parser.parse_args()

    kernels = []
    if args.kernel in ("baseline", "both"):
        kernels.append(("baseline", run_baseline))
    if args.kernel in ("c001_warp_shuffle", "both"):
        kernels.append(("c001_warp_shuffle", run_warp_shuffle))

    input_types = list(INPUT_GENERATORS.keys()) if args.input_type == "all" \
                  else [args.input_type]

    if args.shape == "all":
        shapes = SHAPES
    else:
        parts = args.shape.split(",")
        shapes = [(int(parts[0]), int(parts[1]))]

    results = []
    all_passed = True

    print(f"=== Correctness Validation ===\n")

    for rows, cols in shapes:
        print(f"Shape [{rows}, {cols}]:")
        for itype in input_types:
            gen_fn = INPUT_GENERATORS[itype]
            input_tensor = gen_fn(rows, cols).cuda()
            torch_out = torch_softmax(input_tensor)

            for kname, kfn in kernels:
                r = validate(kfn, kname, input_tensor, torch_out)
                r["shape"] = [rows, cols]
                r["input_type"] = itype
                results.append(r)
                if not r["passed"]:
                    all_passed = False

            del input_tensor, torch_out
            torch.cuda.empty_cache()
        print()

    # Summary
    total = len(results)
    passed_count = sum(1 for r in results if r["passed"])
    print(f"=== Summary: {passed_count}/{total} PASSED ===")

    if not all_passed:
        print("FAILURES:")
        for r in results:
            if not r["passed"]:
                print(f"  {r['kernel']} shape={r['shape']} type={r['input_type']}: "
                      f"max_abs={r['max_abs_err']:.6f} max_rel={r['max_rel_err']:.6f}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results written to {args.json}")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
