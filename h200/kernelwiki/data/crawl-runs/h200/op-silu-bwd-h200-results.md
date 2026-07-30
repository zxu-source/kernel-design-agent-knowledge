# SiLU Backward (Triton) on H200
Date: 2026-07-21 (phase2). grad_x=grad_y*silu'(x) vs torch autograd (backward-only).
H200, Triton 3.6.0. CUDA events min of 50.

## Purpose: SPEEDUP (fused backward).

## Correctness — PASS (fp32 ~7e-7, bf16 ~4e-3).

## Latency — 1.02x-1.29x faster than torch autograd

| MxN | fp32 torch/Triton | bf16 torch/Triton |
|---|--:|--:|
| 4096x4096 | 1.20x | 1.29x |
| 8192x8192 | 1.06x | 1.11x |
| 8192x11008 | 1.03x | 1.07x |
| 8192x14336 | 1.04x | 1.10x |
| 16384x14336 | 1.02x | 1.03x |

## File
silu_bwd_h200.py — sha256 83f7ecb01beb37f9277fbc11d9f8f838b3ca4e2def29e42c26b8962581c4dce6
