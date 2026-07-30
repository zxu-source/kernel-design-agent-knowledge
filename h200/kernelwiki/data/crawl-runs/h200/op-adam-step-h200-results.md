# Fused AdamW (Triton) on H200
Date: 2026-07-21 (phase2). Fused m/v/param update vs torch separate ops.
H200, Triton 3.6.0. CUDA events min of 50.

## Purpose: SPEEDUP (fusion).

## Correctness — PASS (fp32 ~3e-8, bf16 ~2e-4).

## Latency — 2.56x-3.05x faster than torch

| N | fp32 torch/Triton | bf16 torch/Triton |
|--:|--:|--:|
| 1M | 2.56x | 2.58x |
| 4M | 2.88x | 2.85x |
| 16M | 3.05x | 3.05x |
| 33M | 3.04x | 3.04x |

## File
adam_w_step_h200.py — sha256 c7729938926f670d56853eab293ee89cac02de693a50d58c8f3a848773e6d025
