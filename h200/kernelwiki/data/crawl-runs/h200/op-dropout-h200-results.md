# Fused Dropout (Triton) on H200
Date: 2026-07-21 (phase2). Fused Philox+mask+scale (tl.rand) vs torch dropout.
H200, Triton 3.6.0. CUDA events min of 50. p=0.1.

## Purpose: SPEEDUP (record-negative vs torch native dropout).

## Correctness — PASS (statistical): zero_frac=0.1001=p, nonzero=x/(1-p) (val_err=0).

## Latency — 0.54x-0.85x torch

| MxN | fp32 torch/Triton | bf16 torch/Triton |
|---|--:|--:|
| 4096x4096 | 0.83x | 0.58x |
| 8192x8192 | 0.84x | 0.55x |
| 8192x14336 | 0.85x | 0.54x |
| 16384x14336 | 0.85x | 0.54x |

torch's native dropout (optimized Philox) beats Triton tl.rand.

## File
dropout_h200.py — sha256 eef58781276822b5024dd847777a693ec840492495c871916a745db43c16d321
