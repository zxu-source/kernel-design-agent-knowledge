# Per-Row Prefix Sum (Triton) on H200
Date: 2026-07-21 (phase2). tl.cumsum (fp32 acc) exclusive per row vs torch.cumsum.
H200, Triton 3.6.0. CUDA events min of 50.

## Purpose: characterization (scan building block).

## Correctness — fp32 PASS (rel 1e-6); bf16 cumsum inherently lossy.

## Latency (fp32)

| M | N | torch/Triton |
|--:|--:|--:|
| 4096 | 4096 | 4.01x |
| 8192 | 8192 | 2.93x |
| 4096 | 32000 | 0.35x |
| 8192 | 32000 | 0.31x |

Moderate N: 2.9-4x faster. N=32000: BLOCK=32768 spills registers (0.31-0.35x);
tile the scan for huge N.

## File
prefix_sum_h200.py — sha256 36c76a30eb5c35c2c3fa20b95b0ddf7b05a85092efa58cc62a35e577c4f84812
