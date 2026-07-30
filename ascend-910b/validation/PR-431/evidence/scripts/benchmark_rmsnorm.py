#!/usr/bin/env python3
"""Repository-local PR-431 RMSNorm microbenchmark on Ascend NPU."""

import csv
import json
import os
import statistics
import time

import torch
import torch.nn.functional as F

from sgl_kernel_npu.norm.rmsnorm_without_weight import fused_rmsnorm_without_weight


CASES = [
    {"name": "upstream", "shape": [1, 130, 2048], "dtype": "float32"},
    {"name": "seq512", "shape": [1, 512, 2048], "dtype": "float32"},
    {"name": "hidden4096", "shape": [1, 128, 4096], "dtype": "float32"},
]
WARMUP = 20
ITERS = 100
EPS = 1e-6


def elapsed_us(fn):
    for _ in range(WARMUP):
        fn()
    torch.npu.synchronize()
    samples = []
    for _ in range(ITERS):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)
    return {
        "median_us": statistics.median(samples),
        "mean_us": statistics.mean(samples),
        "min_us": min(samples),
        "max_us": max(samples),
    }


def main():
    print(json.dumps({"warmup": WARMUP, "iters": ITERS, "eps": EPS, "cases": CASES}))
    results = []
    for case in CASES:
        dtype = getattr(torch, case["dtype"])
        x = torch.randn(*case["shape"], dtype=dtype, device="npu")
        reference = F.rms_norm(x, normalized_shape=(x.shape[-1],), eps=EPS)
        candidate = fused_rmsnorm_without_weight(x, EPS)
        torch.testing.assert_close(candidate, reference, rtol=1e-3, atol=1e-3)
        baseline = elapsed_us(lambda: F.rms_norm(x, normalized_shape=(x.shape[-1],), eps=EPS))
        fused = elapsed_us(lambda: fused_rmsnorm_without_weight(x, EPS))
        result = {
            **case,
            "baseline": baseline,
            "fused": fused,
            "baseline_over_fused_median": baseline["median_us"] / fused["median_us"],
        }
        results.append(result)
        print(json.dumps(result, sort_keys=True))

    output = os.environ.get("PR431_BENCHMARK_JSON")
    if output:
        with open(output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        with open(output.replace(".json", ".csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["name", "shape", "dtype", "baseline_median_us", "fused_median_us", "baseline_over_fused_median"],
            )
            writer.writeheader()
            for row in results:
                writer.writerow({
                    "name": row["name"],
                    "shape": "x".join(map(str, row["shape"])),
                    "dtype": row["dtype"],
                    "baseline_median_us": row["baseline"]["median_us"],
                    "fused_median_us": row["fused"]["median_us"],
                    "baseline_over_fused_median": row["baseline_over_fused_median"],
                })


if __name__ == "__main__":
    main()
