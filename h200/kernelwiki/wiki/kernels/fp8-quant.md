---
id: kernel-fp8-quant
title: FP8 e4m3 Quantization (Hopper / H200)
type: kernel
architectures:
- sm90
tags:
- quantization
- quant
- fp8
- reduction
confidence: experimental
reproducibility: benchmarked
kernel_types:
- quantization
- fused-kernel
languages:
- triton
- python
related:
- kernel-per-tensor-quant
- kernel-fp8-block-scale-gemm
sources:
- doc-nvidia-tuning-guide
performance_claims: []
blackwell_relevance: Same fp8 e4m3 quant on SM100; per-tensor activation quant (DeepGEMM/vLLM).
operator_purpose: speedup
what_it_does: 'Per-tensor FP8 e4m3 quant: amax -> scale=amax/448 -> clamp/cast to
  fp8_e4m3.'
h200_validation:
  date: 2026-07-21 (phase2; self-reported summary)
  evidence_file: data/crawl-runs/h200/op-fp8-quant-h200-results.md
  harness_dir: artifacts/kernels/fp8-quant/variants
  gpu: H200
  arch: sm90
  toolchain: Triton 3.6.0, PyTorch 2.11.0+cu130
  correctness: PASS — fp32 ~99.996% exact (rare 1-ulp diffs from amax reduction order);
    bf16 ~0.93-0.97 (bf16 amax precision).
  result: 2.0x-3.06x faster than torch at large N (small N 0.71-0.89x, 2-kernel launch
    overhead). Enables FP8 GEMM.
  scope: fp32/bf16 tensors up to 33M elements, on H200.
evidence_basis: H200 benchmark replay on 2026-07-21. The original harness completed
  on H200 with return code 0; its raw stdout/stderr is in the replay archive. This
  remains a local derived implementation, not an upstream-source performance claim.
---
> **Evidence boundary.** This is a locally derived canonical implementation, not a captured upstream source file. Any H200 latency/correctness text below comes from a self-reported run summary; retain it for hypothesis generation, but replay it with raw benchmark output before treating it as benchmark evidence.


## Overview

Per-tensor FP8 e4m3 quantization: find the tensor's abs-max, `scale = amax/448`
(fp8e4m3 max representable), clamp, and cast to `float8_e4m3fn`. The fp8 output
feeds an FP8 GEMM. Two kernels: an `atomic_max` amax (fp32) + elementwise
clamp/cast.

```python
@triton.jit
def fp8_quant_k(x_ptr, o_ptr, inv_scale, N, BLOCK: tl.constexpr):
    ...
    q=x*inv_scale
    q=tl.minimum(tl.maximum(q, -448.0), 448.0)     # fp8e4m3 range
    tl.store(o_ptr+offs, q.to(tl.float8e4nv), mask=mask)
```

## Purpose: SPEEDUP
Produces fp8 that enables FP8 GEMM; **2.0x-3.06x faster** than torch's multi-pass
at large N. Small N is 0.71x-0.89x (two-kernel launch overhead).
[`data/crawl-runs/h200/op-fp8-quant-h200-results.md`](../../data/crawl-runs/h200/op-fp8-quant-h200-results.md).

## Robustness: the clamp is a necessary ~free no-NaN guard (validated)
On adversarial inputs (0.1% outliers = 1e4, 64 Inf, 64 NaN), the `clamp(q, -448,
448)` before the fp8 cast is what keeps the output NaN-free: the **clamped**
kernel produced **0 NaN** (max 448.0), while a **naive no-clamp** cast produced
**64 NaN** (from the Inf inputs -> undefined fp8). Clamp overhead is **~1.6%**
(0.0327ms vs 0.0322ms naive — essentially free). Outliers still hurt *accuracy*
(motivating per-channel/block-scale fp8), but the clamp guarantees correctness
(no NaN). Details:
[`data/crawl-runs/h200/op-fp8-oob-clamp-h200-results.md`](../../data/crawl-runs/h200/op-fp8-oob-clamp-h200-results.md),
harness `artifacts/kernels/fp8-oob-clamp/variants/`.

## H200 measured

| N | dtype | match | torch/Triton |
|--:|---|--:|--:|
| 1M  | fp32 | 1.00 | 0.71x |
| 4M  | fp32 | 1.00 | 0.89x |
| 16M | fp32 | 1.00 | 2.03x |
| 33M | fp32 | 1.00 | 2.50x |
| 33M | bf16 | 0.97 | 3.06x |


## H200 benchmark replay (2026-07-21)

Original harness: `op_fp8_quant.py`. The original harness completed on H200 with return code 0; its raw stdout/stderr is in the replay archive. Evidence: [`replay-2026-07-21-all-experimental.md`](../../data/crawl-runs/h200/replay-2026-07-21-all-experimental.md). All speed ratios are shape- and reference-specific.
