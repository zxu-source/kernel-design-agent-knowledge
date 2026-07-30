---
id: kernel-fused-gemm-bias-act
title: Fused GEMM + Bias + Activation (Hopper / H200)
type: kernel
architectures:
- sm90
tags:
- gemm
- epilogue-fusion
- silu
- kernel-fusion
confidence: experimental
reproducibility: benchmarked
kernel_types:
- gemm
- fused-kernel
languages:
- triton
- python
related:
- kernel-bf16-gemm-hopper
- kernel-silu-and-mul
sources:
- doc-nvidia-tuning-guide
performance_claims: []
blackwell_relevance: Same epilogue-fusion pattern on SM100; standard MLP/attention
  projection (CUTLASS epilogue).
operator_purpose: speedup
what_it_does: Fused BF16 GEMM + per-row bias + SiLU epilogue in one kernel (vs torch
  a@b+bias then silu).
h200_validation:
  date: 2026-07-21 (phase2; self-reported summary)
  evidence_file: data/crawl-runs/h200/op-fused-gemm-bias-act-h200-results.md
  harness_dir: artifacts/kernels/fused-gemm-bias-act/variants
  gpu: H200
  arch: sm90
  toolchain: Triton 3.6.0, PyTorch 2.11.0+cu130
  correctness: PASS — bf16 acc noise 0.125-0.25 vs torch sequential.
  result: Fused 1.04x-1.17x faster than torch (a@b + bias then silu).
  scope: bf16, MLP shapes, on H200. Epilogue fusion; GEMM dominates so the win is
    smaller than pure elementwise fusions.
evidence_basis: H200 benchmark replay on 2026-07-21. The original harness completed
  on H200 with return code 0; its raw stdout/stderr is in the replay archive. This
  remains a local derived implementation, not an upstream-source performance claim.
---
> **Evidence boundary.** This is a locally derived canonical implementation, not a captured upstream source file. Any H200 latency/correctness text below comes from a self-reported run summary; retain it for hypothesis generation, but replay it with raw benchmark output before treating it as benchmark evidence.


## Overview

Epilogue fusion: bf16 GEMM accumulates to fp32, then adds a per-row bias and
applies SiLU in the same kernel — one read of A/B and one write of out, vs
torch's `silu(a@b + bias)` (GEMM + bias-add + silu = 3 launches + 2 intermediates).

```python
@triton.jit
def gemm_bias_silu(...):
    ...
    acc = tl.zeros((BM,BN), dtype=tl.float32)
    for k0 in range(0,K,BK):
        ...
        acc += tl.dot(a, b)
    bias = tl.load(bias_ptr+on, mask=on<N, other=0.0).to(tl.float32)
    acc = acc + bias[None,:]                       # fused bias
    silu = acc / (1.0 + tl.exp(-acc))              # fused activation
    tl.store(o_ptr+..., silu.to(o_ptr.dtype.element_ty), mask=...)
```

## Purpose: SPEEDUP (epilogue fusion)
Fuses bias-add + activation into the GEMM epilogue: 1.04x-1.17x faster than torch
sequential. The win is smaller than pure elementwise fusions (silu-and-mul ~1.6x)
because the GEMM dominates latency and the epilogue is a small fraction.
[`data/crawl-runs/h200/op-fused-gemm-bias-act-h200-results.md`](../../data/crawl-runs/h200/op-fused-gemm-bias-act-h200-results.md).

## H200 measured

| MxNxK | seq/fused |
|---|--:|
| 2048x2048x2048 | 1.16x |
| 4096x4096x4096 | 1.14x |
| 8192x8192x4096 | 1.15x |
| 8192x8192x8192 | 1.04x |
| 8192x11008x4096 | 1.17x |


## H200 benchmark replay (2026-07-21)

Original harness: `op_fused_gemm_bias_act.py`. The original harness completed on H200 with return code 0; its raw stdout/stderr is in the replay archive. Evidence: [`replay-2026-07-21-all-experimental.md`](../../data/crawl-runs/h200/replay-2026-07-21-all-experimental.md). All speed ratios are shape- and reference-specific.
