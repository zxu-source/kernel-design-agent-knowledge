# GELU (tanh) Backward (Triton) on H200
Date: 2026-07-21 (phase2). grad_x=grad_y*gelu'(x) vs torch autograd (backward-only).
H200, Triton 3.6.0. CUDA events min of 50.

## Purpose: SPEEDUP (fused backward).

## Correctness — PASS (fp32 ~1e-6, bf16 ~1e-4).

## Latency — 1.01x-1.27x faster than torch autograd

| MxN | fp32 torch/Triton | bf16 torch/Triton |
|---|--:|--:|
| 4096x4096 | 1.20x | 1.27x |
| 8192x8192 | 1.07x | 1.14x |
| 8192x11008 | 1.04x | 1.12x |
| 8192x14336 | 1.03x | 1.08x |
| 16384x14336 | 1.01x | 1.06x |

## File
gelu_bwd_h200.py — sha256 2330f27dcf2ff4d7bb186396a9ea9531c9a23a89f431c1cae5657a1c683d41c8
