# Groupwise (G=128) INT8 Quant (Triton) on H200
Date: 2026-07-21 (phase2). Per-128-group amax+scale+round/clamp/cast vs torch
(reshape/amax/div/round/clamp/cast). H200, Triton 3.6.0. CUDA events min of 50.

## Purpose: BOTH — speedup (fused) + accuracy (per-group scales).

## Correctness — PASS (match ~1.0, maxdiff<=1; group scales exact).

## Latency — 2.70x-3.42x faster than torch

| M x N | fp32 torch/Triton | bf16 torch/Triton |
|---|--:|--:|
| 4096x4096 | 2.70x | 3.23x |
| 8192x8192 | 2.79x | 3.38x |
| 8192x11008 | 2.80x | 3.39x |
| 8192x14336 | 2.81x | 3.40x |
| 16384x14336 | 2.82x | 3.42x |

## File
`groupwise_int8_quant_h200.py` — sha256 `fdd4920b523ffd01d37f7609d86fc47f8e4cc2aea344cd8245f9f3daf3d30985`
