# Per-Tensor Symmetric INT8 Quant (Triton) on H200
Date: 2026-07-21 (phase2). amax (atomic_max) + round/clamp/cast to int8 vs torch
(abs().max()+scale+round+clamp+cast). H200, Triton 3.6.0. CUDA events min of 50.

## Purpose: SPEEDUP (produces int8 for INT8 GEMM; fused vs torch multi-pass).

## Correctness — PASS
fp32 exact match; bf16 maxdiff<=1 (bf16 amax lower precision; ~95% of elements exact).

## Latency

| N | dtype | match | torch/Triton |
|--:|---|--:|--:|
| 1M  | fp32 | 1.00 | 0.64x |
| 4M  | fp32 | 1.00 | 1.03x |
| 16M | fp32 | 1.00 | 2.68x |
| 33M | fp32 | 1.00 | 3.56x |
| 33M | bf16 | 0.95 | 3.34x |

Fused Triton 2.3-3.6x faster than torch at large N; small N slower (2-kernel launch overhead).

## File
`per_tensor_int8_quant_h200.py` — sha256 `b07932b14523f4b16f2718c325448d28d71d18734610839dfb368c6b1ad59b66`
