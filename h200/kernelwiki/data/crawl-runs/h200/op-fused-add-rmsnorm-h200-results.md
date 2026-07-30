# Fused Add+RMSNorm (Triton) on H200

Date: 2026-07-21 (phase2). LLM residual-stream kernel: `residual_out = residual + x`,
`out = rmsnorm(residual_out)`. Fusing add+norm reads `residual` and `x` once and
writes `residual_out` and `out` once (vs torch: `add->tmp`, `rmsnorm(tmp)->out`,
`copy tmp->residual_out` = 3 reads + 3 writes).

H200, Triton 3.6.0, bf16. CUDA events, min of 30 trials.

## Purpose: SPEEDUP (fusion) — fewer launches + less memory traffic.

## What it does
`residual_out = residual + x; out = x * rsqrt(mean((residual+x)^2)+eps) * w` in one kernel.

## Correctness — PASS
out max abs err vs torch: bf16 ~3.1e-2 (dtype precision). residual_out bit-identical (0.0).

## Latency — fused 1.10x-1.45x faster than torch

| M | N | torch/Triton |
|--:|--:|--:|
| 4096 | 4096  | 1.10x |
| 8192 | 4096  | 1.21x |
| 4096 | 8192  | 1.32x |
| 8192 | 8192  | 1.38x |
| 8192 | 11008 | 1.38x |
| 8192 | 14336 | 1.45x |

## File
`fused_add_rmsnorm_h200.py` — sha256 `39eeb27e65df3e8e1a750430374adade1c0113088031a4461677affc14269a11`
