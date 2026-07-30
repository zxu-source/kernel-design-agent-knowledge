# Gradient Norm Clipping (Triton) on H200
Date: 2026-07-21 (phase2). 2-kernel (atomic sum_sq + conditional scale) vs torch
manual (norm + scale). fp32. H200, Triton 3.6.0. CUDA events min of 50.

## Purpose: BOTH — robustness (prevents explosion) + speedup (record-negative).

## Correctness — PASS (norm err ~3e-7, grad err ~5e-10).

## Latency — 0.31x-0.40x torch

| N | torch/Triton |
|--:|--:|
| 1M | 0.40x |
| 4M | 0.31x |
| 16M | 0.34x |
| 33M | 0.35x |

2-kernel approach slower than torch's optimized norm+scale. Use torch.

## File
grad_clip_h200.py — sha256 35313d3f072358533f6811a4fe5f2e1fee7753baccc37f53bf12bf3fefe310cb
