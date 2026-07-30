# Softmax Backward (Triton) on H200
Date: 2026-07-21 (phase2). grad_x=(grad_y-sum(grad_y*y))*y per row vs torch autograd.
fp32. H200, Triton 3.6.0. CUDA events min of 50.

## Purpose: SPEEDUP (fused backward).

## Correctness — PASS (rel ~7e-8 vs torch autograd).

## Latency — 3.35x-4.04x moderate N; 0.70x N=32000 (BLOCK spill)

| M | N | torch/Triton |
|--:|--:|--:|
| 4096 | 4096 | 3.35x |
| 8192 | 8192 | 4.04x |
| 4096 | 32000 | 0.70x |
| 8192 | 32000 | 0.70x |

## File
softmax_bwd_h200.py — sha256 c883edae2eb83cc407f13895aa4267e579c0123e4d44102725fbe93a0343e32a
