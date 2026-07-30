# Fused GEMM+Bias+SiLU (Triton) on H200
Date: 2026-07-21 (phase2). Fused bf16 GEMM+bias+SiLU epilogue vs torch sequential
(silu(a@b+bias)). H200, Triton 3.6.0. CUDA events min of 20 trials.

## Purpose: SPEEDUP (epilogue fusion — bias+act into the GEMM epilogue).

## Correctness — PASS (bf16 acc noise 0.125-0.25)

## Latency — fused 1.04x-1.17x faster than torch

| MxNxK | seq/fused |
|---|--:|
| 2048x2048x2048 | 1.16x |
| 4096x4096x4096 | 1.14x |
| 8192x8192x4096 | 1.15x |
| 8192x8192x8192 | 1.04x |
| 8192x11008x4096 | 1.17x |

Win is smaller than pure elementwise fusions because the GEMM dominates and the
epilogue (bias+silu) is a small fraction of latency.

## File
`fused_gemm_bias_silu_h200.py` — sha256 `f66a662164e483779797e952b98d38274cdbb60890de4d9f551127866a5824d2`
