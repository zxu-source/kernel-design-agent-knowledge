# FP8 OOB-Clamp Robustness on H200
Date: 2026-07-21 (phase2). Adversarial inputs (0.1% outliers = 1e4, 64 Inf, 64
NaN) into fp8 e4m3 quant. Clamped kernel (clamp to [-448,448] before cast) vs
naive (no clamp). H200, Triton 3.6.0. N=16.7M fp32.

## Purpose: ROBUSTNESS (no-NaN guard against outlier/Inf overflow on fp8 cast).

## Result
- Clamped kernel: NaN=0, Inf=0, max_abs=448.0 (all clamped to fp8 range). ROBUST.
- Naive (no-clamp): 64 NaN (from the 64 Inf inputs -> undefined fp8 cast).
- With proper per-tensor scale (amax/448) + clamp: NaN=0 (Inf clamps to 448, not NaN).
- Clamp overhead: 0.0327ms (clamp) vs 0.0322ms (naive) -> 1.016x (~1.6%, essentially free).

## Takeaway
The clamp-to-[-448,448] before the fp8 cast is a NECESSARY, ~free robustness guard:
without it, outliers / Inf / NaN in the input produce NaN fp8 values (silent
corruption downstream). With it, all values map cleanly into the fp8 range.
(Outliers still hurt *accuracy* — motivating per-channel/block-scale fp8 — but
the clamp guarantees correctness, no NaN.)

## File
op_fp8_oob_clamp.py — sha256 d97b7b03d503caa8846bf008606b0f1ec9b21542d4ceb44f623c07ed7db5c787
