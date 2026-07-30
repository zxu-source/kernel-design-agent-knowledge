# INT4 -> BF16 Dequant (W4A16, Triton) on H200
Date: 2026-07-21 (phase2). Unpack packed-uint8 INT4 (two's complement) * per-row
scale -> bf16 vs torch (stack/reshape/where). H200, Triton 3.6.0. CUDA events
min of 50.

## Purpose: SPEEDUP (memory) — 4x smaller weights, on-the-fly dequant for W4A16 GEMM.

## Correctness — PASS (bit-identical, match=1.0, err=0.0)

## Latency — 19x-37x faster than torch

| M x N | torch/Triton |
|---|--:|
| 4096x4096 | 19.26x |
| 8192x8192 | 30.83x |
| 8192x11008 | 31.46x |
| 8192x14336 | 34.03x |
| 16384x14336 | 36.84x |

## File
`int4_dequant_w4a16_h200.py` — sha256 `369fa4385808f59a868c71b3e6b827ba1070295266622017faadf8464604c79a`
