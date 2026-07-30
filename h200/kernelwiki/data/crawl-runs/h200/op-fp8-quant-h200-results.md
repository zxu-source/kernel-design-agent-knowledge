# FP8 e4m3 Quant (Triton) on H200
Date: 2026-07-21 (phase2). amax (atomic_max) + clamp/cast to fp8_e4m3 vs torch
(abs().max()+scale+clamp+cast). H200, Triton 3.6.0. CUDA events min of 50.

## Purpose: SPEEDUP (produces fp8 for FP8 GEMM).

## Correctness — effectively PASS
fp32 ~99.996% exact (rare 1-ulp diffs from amax reduction order); bf16 ~0.93-0.97 (bf16 amax precision).

## Latency

| N | dtype | match | torch/Triton |
|--:|---|--:|--:|
| 1M  | fp32 | 1.00 | 0.71x |
| 4M  | fp32 | 1.00 | 0.89x |
| 16M | fp32 | 1.00 | 2.03x |
| 33M | fp32 | 1.00 | 2.50x |
| 33M | bf16 | 0.97 | 3.06x |

Fused Triton 2.0-3.06x faster than torch at large N; small N slower (2-kernel launch overhead).

## File
`fp8_e4m3_quant_h200.py` — sha256 `8c03214e35920187eec7a4132e5deb1446efd3b50b7339f50c61754d936b88cd`
