# LayerNorm Forward (Triton) on H200

Date: 2026-07-20 (phase2). `y = (x-mean)*rsqrt(var+eps)*w + b`,
`var=mean((x-mean)^2)`. One fused Triton kernel (load once: sum_x, sum_x2 ->
mean/var via `var=E[x^2]-E[x]^2`). vs `torch.nn.functional.layer_norm`.

H200, Triton 3.6.0, PyTorch 2.11+cu130. CUDA events, min of 30 trials.

## Purpose: BOTH
- **speedup**: fused single-pass vs reduce+center+normalize+scale+bias.
- **robustness**: fp32 reductions + eps.

## What it does
`y = (x - mean) * rsqrt(var + eps) * w + b`, per row, one fused kernel.

## Correctness — PASS
Max abs err vs torch: bf16 ~1.5e-2..3.1e-2, fp16 ~1.9e-3..3.9e-3 (dtype precision).

## Latency — Triton fused 1.10x-1.51x faster than torch

| M | N | dtype | torch/Triton |
|--:|--:|---|--:|
| 4096  | 4096  | bf16 | 1.22x |
| 8192  | 4096  | bf16 | 1.24x |
| 4096  | 8192  | bf16 | 1.39x |
| 8192  | 8192  | bf16 | 1.51x |
| 4096  | 11008 | bf16 | 1.26x |
| 8192  | 14336 | bf16 | 1.25x |
| (fp16 tracks within ~0.1x of bf16) | | | |

## File
`layernorm_fwd_h200.py` — sha256 `840f6c6731bec60882ace2edf4ed906e866a2735085a3084b3beaeb0311d3bbc`
