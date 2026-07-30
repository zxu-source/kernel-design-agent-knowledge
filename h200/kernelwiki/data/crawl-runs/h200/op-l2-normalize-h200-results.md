# L2-Normalize (Triton) on H200
Date: 2026-07-21 (phase2). Per-row unit norm (fused) vs torch.nn.functional.normalize.
H200, Triton 3.6.0. CUDA events min of 50.

## Purpose: SPEEDUP (fused reduction+scale).

## Correctness — PASS (fp32 ~1e-8, bf16 ~2e-4).

## Latency — 1.95x-3.31x faster than torch

| MxN | fp32 torch/Triton | bf16 torch/Triton |
|---|--:|--:|
| 4096x4096 | 1.95x | 2.64x |
| 8192x8192 | 2.06x | 3.24x |
| 8192x11008 | 1.99x | 3.13x |
| 8192x14336 | 2.02x | 3.22x |
| 16384x14336 | 2.03x | 3.31x |

## File
l2norm_h200.py — sha256 727c16788653a078f055cbc4a7a5285213cb4ea8e6d6809a3d9a0b0367438d0e
