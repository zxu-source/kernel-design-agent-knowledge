# RMSNorm Backward (Triton) on H200
Date: 2026-07-21 (phase2). grad_x + atomic grad_w vs torch autograd. fp32.
H200, Triton 3.6.0. CUDA events min of 50.

## Purpose: SPEEDUP (fused backward).

## Correctness — PASS (grad_x err ~2e-6; grad_w err ~1e-3 from atomic order).

## Latency — 1.34x-1.66x moderate N; 0.28x-0.35x large N (BLOCK spill + atomic)

| M x N | torch/Triton |
|---|--:|
| 4096x4096 | 1.34x |
| 8192x8192 | 1.66x |
| 8192x11008 | 0.28x |
| 8192x14336 | 0.35x |
| 16384x14336 | 0.35x |

## File
rmsnorm_bwd_h200.py — sha256 ef309a26404b8823d08e15f71208bdec3217384b3a8eaae3eba3688f92653f5d
