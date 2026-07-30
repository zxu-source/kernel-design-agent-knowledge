# RMSNorm Forward (Triton) on H200

Date: 2026-07-20 (phase2). Source-informed H200 validation. Canonical RMSNorm
`y = x * rsqrt(mean(x^2)+eps) * w`, the standard LLM block/final norm.

Hardware: NVIDIA H200, 132 SMs, cc 9.0. Triton 3.6.0, PyTorch 2.11.0+cu130.
Kernel: one program per row, fp32 reduction of x^2, rsqrt(+eps), multiply by
weight. vs `torch.nn.functional.rms_norm`. Timing: CUDA events, min of 30 trials.

## Purpose: BOTH
- **speedup**: one fused kernel replaces reduce(x^2) + mean + rsqrt + normalize
  + weight-mul (one read, one write of x; fewer launches + less traffic).
- **robustness**: fp32 accumulation of the variance + the `eps` term avoid
  fp16/bf16 overflow/underflow in the squared-sum reduction.

## What it does
`y = x * rsqrt(mean(x^2, axis=-1) + eps) * w`, per row, in one fused Triton kernel.

## Correctness — PASS
Max abs err vs `torch.nn.functional.rms_norm`: bf16 ~1.5e-2..3.1e-2 (bf16's
~3-decimal precision), fp16 ~1.9e-3 — i.e. exact to dtype precision. Single-row
(M=1, N=11008): err = 0.0 (bit-identical).

## Latency — Triton fused up to 1.40x faster than torch

| M | N | dtype | torch/Triton |
|--:|--:|---|--:|
| 4096  | 4096  | bf16 | 1.01x |
| 8192  | 4096  | bf16 | 1.06x |
| 4096  | 8192  | bf16 | 1.25x |
| 8192  | 8192  | bf16 | 1.36x |
| 4096  | 11008 | bf16 | 1.16x |
| 8192  | 14336 | bf16 | 1.40x |
| (fp16 rows track within 0.02x of bf16; 1-row shape = parity, launch-bound) | | | |

The fused Triton kernel matches torch at small/1-row shapes (launch-bound) and
pulls ahead to ~1.4x as row count grows (launch amortization + memory-traffic
advantage of the single fused pass).

## File

| file | role | sha256 |
|---|---|---|
| `rmsnorm_fwd_h200.py` | runnable harness | `500c986cfd90871e4e947bec662d991c2b55f7d8f275e02ba4ac6f69cf00907c` |
