# Causal Mask Gen (Triton) on H200
Date: 2026-07-21 (phase2). Direct 0/-inf write vs torch.triu(ones*-inf,1). fp32.
H200, Triton 3.6.0. CUDA events min of 50.

## Purpose: BOTH — speedup (direct write) + robustness (-inf masking).

## Correctness — PASS (0/-inf positions correct; NaN in diff is -inf-(-inf) artifact).

## Latency — 1.22x-5.13x faster than torch

| N | torch/Triton |
|--:|--:|
| 1024 | 1.22x |
| 2048 | 1.84x |
| 4096 | 3.75x |
| 8192 | 5.13x |

## File
causal_mask_h200.py — sha256 b3b30b39d28f47dbbce8659bf21391be14d406db2b55b8d59ed77d6fc141e4ee
