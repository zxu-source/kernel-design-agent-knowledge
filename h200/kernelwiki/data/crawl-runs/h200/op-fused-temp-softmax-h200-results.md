# Fused Temperature + Softmax (Triton) on H200
Date: 2026-07-21 (phase2). `y = softmax(x / temp)` fused; ~10% positions set to
-inf to verify masking. vs torch `(x/temp).softmax()`. H200, Triton 3.6.0.

## Purpose: BOTH — speedup (fused scale+softmax) + robustness (-inf mask, overflow guard).

## What it does
Per-row: `y = softmax(x * inv_temp)`, fused one pass, -inf-mask safe.

## Correctness — PASS
err 1e-6..9.8e-4 vs torch, including -inf-masked positions (robustness confirmed).

## Latency — fused 2.0x-5.5x faster than torch

| M | N | torch/Triton (temp=1.0) |
|--:|--:|--:|
| 4096 | 4096 | 3.48x |
| 8192 | 8192 | 5.44x |
| 4096 | 32000 | 2.03x |
| 8192 | 32000 | 2.04x |

(temp 0.7/1.5 track within 0.1x.)

## File
`fused_temp_softmax_h200.py` — sha256 `59dd59fefed82a44182c47d9e8c22a24e514b57450c2f859b86bd2203671b337`
