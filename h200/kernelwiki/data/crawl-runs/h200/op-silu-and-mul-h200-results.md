# SiLU-and-Mul (Triton) on H200

Date: 2026-07-21 (phase2). LLM MLP `y = silu(gate) * up`. Fused elementwise vs
torch `silu(gate) * up` (2 launches + intermediate). H200, Triton 3.6.0,
CUDA events min of 50 trials.

## Purpose: SPEEDUP (fusion) — one kernel, no intermediate.

## What it does
`y = (gate / (1 + e^-gate)) * up`, fused elementwise.

## Correctness — PASS
Max abs err vs torch: bf16 ~1.6e-2, fp16 ~1.9e-3 (dtype precision).

## Latency — fused 1.30x-1.69x faster than torch

| M | N | dtype | torch/Triton |
|--:|--:|---|--:|
| 4096  | 4096  | bf16 | 1.30x |
| 8192  | 8192  | bf16 | 1.60x |
| 8192  | 11008 | bf16 | 1.64x |
| 8192  | 14336 | bf16 | 1.64x |
| 16384 | 14336 | bf16 | 1.69x |
| (fp16 tracks within 0.02x) | | | |

## File
`silu_and_mul_h200.py` — sha256 `4d16cacdd9a636f22933d2cf758736bd8377f57aa1cbfcf0788745857456536e`
