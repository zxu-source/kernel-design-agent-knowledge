# GELU tanh-approx (Triton) on H200
Date: 2026-07-21 (phase2). `0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))`.
H200, Triton 3.6.0. CUDA events min of 50 trials.

## Purpose: SPEEDUP (marginal — head-to-head vs torch's already-fused gelu)

## What it does
GELU tanh approximation, elementwise (tanh via exp since tl.tanh absent in 3.6).

## Correctness — PASS
bf16 bit-identical, fp16 ~6e-5 vs torch gelu(approximate='tanh').

## Latency — ~parity with torch (0.87x-1.12x)

| M | N | bf16 torch/Triton |
|--:|--:|--:|
| 4096 | 4096 | 0.88x |
| 8192 | 8192 | 1.05x |
| 8192 | 11008 | 1.08x |
| 8192 | 14336 | 1.09x |
| 16384 | 14336 | 1.12x |

gelu is a single fused op in torch, so Triton is ~parity (slight win at large,
slight loss at small/launch-bound). Value: validated self-contained impl.

## File
`gelu_tanh_h200.py` — sha256 `ffe5c0afc2a0546cda9237d0ea07aa741936d0e716cd9c4f2566c7eb5e6f9d21`
