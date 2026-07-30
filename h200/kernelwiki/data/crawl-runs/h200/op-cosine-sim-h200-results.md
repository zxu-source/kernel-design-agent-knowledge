# Cosine Similarity (Triton) on H200
Date: 2026-07-21 (phase2). Fused dot+norms vs torch.cosine_similarity. fp32.
H200, Triton 3.6.0. CUDA events min of 50.

## Purpose: SPEEDUP (fused single-pass).

## Correctness — PASS (err ~1e-8).

## Latency — 5.26x-6.67x faster than torch

| M x N | torch/Triton |
|---|--:|
| 4096x4096 | 5.26x |
| 8192x8192 | 6.40x |
| 8192x11008 | 6.65x |
| 8192x14336 | 6.54x |
| 16384x14336 | 6.67x |

## File
cosine_sim_h200.py — sha256 0d373f025987c55a09d0ebd3afbfb15c007861a6e69151fdfa8074020c377e87
