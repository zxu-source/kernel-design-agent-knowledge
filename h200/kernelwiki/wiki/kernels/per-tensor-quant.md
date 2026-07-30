---
id: kernel-per-tensor-quant
title: Per-Tensor Symmetric INT8 Quantization (Hopper / H200)
type: kernel
architectures:
- sm90
tags:
- quantization
- quant
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
- kernel-block-reduce
- kernel-fp8-block-scale-gemm
sources:
- doc-nvidia-tuning-guide
performance_claims: []
blackwell_relevance: Same per-tensor quant on SM100; standard W8A8 INT8 quant (TensorRT-LLM
  / vLLM).
operator_purpose: speedup
what_it_does: 'Per-tensor symmetric INT8 quant: amax (atomic_max) -> scale=amax/127
  -> round/clamp/cast to int8.'
h200_validation:
  date: 2026-07-21 (phase2; self-reported summary)
  evidence_file: data/crawl-runs/h200/op-per-tensor-quant-h200-results.md
  harness_dir: artifacts/kernels/per-tensor-quant/variants
  gpu: H200
  arch: sm90
  toolchain: Triton 3.6.0, PyTorch 2.11.0+cu130
  correctness: PASS — fp32 exact match; bf16 maxdiff<=1 (bf16 amax precision).
  result: 2.3x-3.6x faster than torch at large N (fused; small N 0.64x from 2-kernel
    launch overhead). Enables INT8 GEMM.
  scope: fp32/bf16 tensors up to 33M elements, on H200.
evidence_basis: H200 benchmark replay on 2026-07-21. The original harness completed
  on H200 with return code 0; its raw stdout/stderr is in the replay archive. This
  remains a local derived implementation, not an upstream-source performance claim.
---
> **Evidence boundary.** This is a locally derived canonical implementation, not a captured upstream source file. Any H200 latency/correctness text below comes from a self-reported run summary; retain it for hypothesis generation, but replay it with raw benchmark output before treating it as benchmark evidence.


## Overview

Per-tensor symmetric INT8 quantization (W8A8): find the tensor's absolute max,
`scale = amax/127`, then quantize `round(clamp(x*scale, -128, 127))` to int8.
The int8 output feeds a faster INT8 GEMM. Two kernels: an `atomic_max` amax
reduction (fp32) + an elementwise round/clamp/cast.

```python
@triton.jit
def amax_kernel(x_ptr, out_ptr, N, BLOCK):
    ... m=tl.max(tl.abs(x),axis=0).to(tl.float32); tl.atomic_max(out_ptr, m)
@triton.jit
def quant_kernel(x_ptr, o_ptr, inv_scale, N, BLOCK):
    ... q=tl.extra.libdevice.llrint(x*inv_scale); q=clamp(q,-128,127); store int8
```

## Purpose: SPEEDUP
Produces int8 that enables INT8 GEMM; the fused kernels are **2.3x-3.6x faster**
than torch's multi-pass (`abs().max()` + `/scale` + `round` + `clamp` + `cast`)
at large N. Small N is 0.64x (two-kernel launch overhead).
[`data/crawl-runs/h200/op-per-tensor-quant-h200-results.md`](../../data/crawl-runs/h200/op-per-tensor-quant-h200-results.md).

## H200 measured

| N | dtype | match | torch/Triton |
|--:|---|--:|--:|
| 1M  | fp32 | 1.00 | 0.64x |
| 4M  | fp32 | 1.00 | 1.03x |
| 16M | fp32 | 1.00 | 2.68x |
| 33M | fp32 | 1.00 | 3.56x |
| (bf16 ~0.95 exact, maxdiff 1) | | | |


## H200 benchmark replay (2026-07-21)

Original harness: `op_per_tensor_quant.py`. The original harness completed on H200 with return code 0; its raw stdout/stderr is in the replay archive. Evidence: [`replay-2026-07-21-all-experimental.md`](../../data/crawl-runs/h200/replay-2026-07-21-all-experimental.md). All speed ratios are shape- and reference-specific.
