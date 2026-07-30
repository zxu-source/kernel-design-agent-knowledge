# Tiled Transpose (Triton) on H200
Date: 2026-07-21 (phase2). Coalesced [BM,BN] load + tl.trans + coalesced store
(square + non-square) vs torch a.t().contiguous(). H200, Triton 3.6.0. CUDA events
min of 50.

## Purpose: SPEEDUP (coalescing — avoids uncoalesced strided writes of naive transpose).

## Correctness — PASS (bit-identical, err=0.0, square + non-square).

## Latency — 2.79x-6.01x faster than torch

| M x N | fp32 torch/Triton | fp16 torch/Triton |
|---|--:|--:|
| 4096x4096 | 2.79x | 4.38x |
| 8192x8192 | 3.29x | 5.45x |
| 8192x4096 | 3.11x | 4.93x |
| 16384x16384 | 3.50x | 6.01x |
| 4096x14336 | 3.28x | 5.44x |

## File
tiled_transpose_h200.py — sha256 be8ff042307d59fc837d9840d7cc909b20ee2dca9f16a170e54ed244f8029c36
