# LayerNorm Backward (Triton) on H200
Date: 2026-07-21 (phase2). grad_x + atomic grad_w/grad_b vs torch autograd. fp32.
H200, Triton 3.6.0. CUDA events min of 50.

## Purpose: SPEEDUP (marginal).

## Correctness — moderate N PASS (grad_x ~3e-6, grad_w ~1e-3). Large N: ~1e-3 (one-pass variance precision).

## Latency

| M x N | grad_x err | torch/Triton |
|---|--:|--:|
| 4096x4096 | 2.9e-6 | 1.15x |
| 8192x8192 | 2.9e-6 | 1.06x |
| 8192x11008 | 5.7e-3 | 0.21x |
| 8192x14336 | 1.1e-3 | 0.26x |
| 16384x14336 | 1.1e-3 | 0.26x |

Marginal at moderate N; large N slow (BLOCK spill + 3 atomics + one-pass variance precision).

## File
layernorm_bwd_h200.py — sha256 7464bf29e5098c5df8c350e928614948cbe8261c1d45c9cd81d7eea979043ad7
