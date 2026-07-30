# Cross-Entropy Loss (Triton) on H200
Date: 2026-07-21 (phase2). Fused logsumexp + target gather vs torch cross_entropy.
fp32. H200, Triton 3.6.0. CUDA events min of 50.

## Purpose: SPEEDUP (fused loss).

## Correctness — PASS (err 1.9e-6).

## Latency — 2.03x-2.43x moderate vocab; 0.34x D=128K (BLOCK spill)

| M | D | torch/Triton |
|--:|--:|--:|
| 4096 | 4096 | 2.03x |
| 8192 | 32000 | 2.17x |
| 8192 | 128256 | 0.34x |
| 4096 | 128256 | 0.34x |
| 8192 | 4096 | 2.43x |

## File
cross_entropy_h200.py — sha256 ed85b93e237c76e44a3f72041f4ee5477560f829665c3d2f17918d7412c215ee
