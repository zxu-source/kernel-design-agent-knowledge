# FA-2 Causal (Triton) on H200
Date: 2026-07-21 (phase2). FA-2 causal forward (lower-triangular mask, N tiles
capped at diagonal) vs torch SDPA(is_causal=True). fp16. H200, Triton 3.6.0.

## Purpose: SPEEDUP (fused causal attention).

## Correctness — PASS (err 3-6e-5 vs torch SDPA causal).

## Latency — 0.36x-0.66x torch SDPA (naive Triton FA-2 slower than torch backend)

| BxHxMxD | sdpa/Triton |
|---|--:|
| 1x8x8192x64 | 0.38x |
| 1x8x8192x128 | 0.62x |
| 4x8x4096x64 | 0.42x |
| 1x4x16384x64 | 0.36x |
| 2x16x2048x128 | 0.66x |

Same pattern as non-causal FA-2: torch SDPA (FA3/cuDNN) beats the naive
Triton FA-2 kernel. Correctness validated; throughput NOT representative of a
production FA kernel.

## File
fa2_causal_h200.py — sha256 6b56816c0930652405102a14c3e95185f9368ede6346c2da6d7629da8af590ec
