# Scatter-Add (Triton) on H200
Date: 2026-07-21 (phase2). atomic-add scatter vs torch.index_add_. fp32.
H200, Triton 3.6.0. CUDA events min of 50.

## Purpose: SPEEDUP (characterization).

## Correctness — PASS (rel ~6e-8 from atomic-add order).

## Latency — 1.10x-1.44x faster than torch.index_add

| MxVxD | torch/Triton |
|---|--:|
| 8192x128256x4096 | 1.19x |
| 4096x128256x4096 | 1.10x |
| 8192x32000x4096 | 1.44x |
| 16384x128256x4096 | 1.32x |
| 8192x128256x8192 | 1.11x |

## File
scatter_add_h200.py — sha256 18ccdf08eb21cabb7ffa80f46827d1362d0ec1810416381d177fd68603d00a63
