# BCE-with-Logits (Triton) on H200
Date: 2026-07-21 (phase2). Fused numerically-stable BCE vs torch BCE_with_logits.
fp32. H200, Triton 3.6.0. CUDA events min of 50.

## Purpose: SPEEDUP (fusion + numerical stability).

## Correctness — PASS (err ~5e-7).

## Latency — 3.03x-3.74x faster than torch

| M x N | torch/Triton |
|---|--:|
| 4096x4096 | 3.03x |
| 8192x8192 | 3.57x |
| 8192x14336 | 3.66x |
| 16384x14336 | 3.74x |

## File
bce_logits_h200.py — sha256 78545a2cbbcab037ca4401a9cfd71304def302373f66651972e6b19bd2c60c39
