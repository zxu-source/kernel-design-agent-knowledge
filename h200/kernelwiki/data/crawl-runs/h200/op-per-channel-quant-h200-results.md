# Per-Channel Symmetric INT8 Quant (Triton) on H200
Date: 2026-07-21 (phase2). Fused per-row amax+scale+round/clamp/cast vs torch
(abs/amax(dim=-1)/div/round/clamp/cast). H200, Triton 3.6.0. CUDA events min of 50.

## Purpose: BOTH — speedup (fused) + accuracy (per-channel scales).

## Correctness — PASS
match ~1.0 (maxdiff<=1, ~0.03% of bf16 elems differ by 1 rounding tie); per-channel scales exact.

## Latency — 6.25x-12.08x faster than torch

| M x N | fp32 torch/Triton | bf16 torch/Triton |
|---|--:|--:|
| 4096x4096 | 6.25x | 7.81x |
| 8192x8192 | 9.07x | 12.08x |
| 8192x11008 | 8.01x | 8.74x |
| 8192x14336 | 9.28x | 10.76x |
| 16384x14336 | 9.90x | 11.44x |

## File
`per_channel_int8_quant_h200.py` — sha256 `e825f5d3721b95de795fa938fce775de99f4242044ffda8736a01a9eae20ecb7`
