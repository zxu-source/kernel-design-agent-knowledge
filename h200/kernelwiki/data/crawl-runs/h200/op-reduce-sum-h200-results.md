# Block+Warp Sum Reduction (Triton) on H200
Date: 2026-07-21 (phase2). Two-stage reduce (per-block tl.sum -> partials ->
partials.sum()) vs torch.sum. H200, Triton 3.6.0. CUDA events min of 50 trials.

## Purpose: characterization (record-negative).

## Correctness — PASS (rel error ~1e-7).

## Latency — 0.62x-0.95x torch.sum

| N | fp32 torch/Triton |
|--:|--:|
| 1M  | 0.64x |
| 4M  | 0.75x |
| 16M | 0.91x |
| 64M | 0.95x |

Triton's standalone two-stage reduce is slower than torch's tuned CUB-style
reduce (launch-bound at small N, ~parity at large N). Use torch/CUB for a
standalone reduce; the per-row tl.sum inside fused kernels (RMSNorm/softmax) is fine.

## File
`reduce_sum_h200.py` — sha256 `1bf65d45c8cf4d22274b293a768737312ce3da8566f431270971c6f68c9747ca`
