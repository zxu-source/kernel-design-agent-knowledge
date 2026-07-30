---
id: kernel-gelu-bwd
title: GELU (tanh) Backward (Hopper / H200)
type: kernel
architectures:
- sm90
tags:
- gelu
- activation
- kernel-fusion
confidence: experimental
reproducibility: benchmarked
kernel_types:
- fused-kernel
languages:
- triton
- python
related:
- kernel-gelu
- kernel-silu-and-mul
sources:
- doc-nvidia-tuning-guide
performance_claims: []
blackwell_relevance: Same GELU backward on SM100; training gradient.
operator_purpose: speedup
what_it_does: 'GELU tanh backward: grad_x=grad_y*gelu''(x) (tanh via exp); fused elementwise.'
h200_validation:
  date: 2026-07-21 (phase2; self-reported summary)
  evidence_file: data/crawl-runs/h200/op-gelu-bwd-h200-results.md
  harness_dir: artifacts/kernels/gelu-bwd/variants
  gpu: H200
  arch: sm90
  toolchain: Triton 3.6.0, PyTorch 2.11.0+cu130
  correctness: PASS — fp32 ~1e-6, bf16 ~1e-4 (tanh/exp bf16 noise) vs torch autograd.
  result: 1.01x-1.27x faster than torch autograd (backward-only timed).
  scope: fp32/bf16, MLP shapes, on H200.
evidence_basis: H200 benchmark replay on 2026-07-21. The original harness completed
  on H200 with return code 0; its raw stdout/stderr is in the replay archive. This
  remains a local derived implementation, not an upstream-source performance claim.
---
> **Evidence boundary.** This is a locally derived canonical implementation, not a captured upstream source file. Any H200 latency/correctness text below comes from a self-reported run summary; retain it for hypothesis generation, but replay it with raw benchmark output before treating it as benchmark evidence.


## Overview

GELU tanh-approx backward (training gradient): `grad_x = grad_y * gelu'(x)`,
`gelu'(x) = 0.5*(1+tanh(g)) + 0.5*x*(1-tanh(g)^2)*g'(x)`, with
`g(x) = sqrt(2/pi)*(x + 0.044715*x^3)` and `g'(x) = sqrt(2/pi)*(1 + 0.134145*x^2)`.
Tanh via `(e^2g - 1)/(e^2g + 1)` (tl.tanh absent in 3.6).

```python
@triton.jit
def gelu_bwd(x_ptr, gy_ptr, gx_ptr, total, BLOCK):
    pid=tl.program_id(0); offs=pid*BLOCK+tl.arange(0,BLOCK); mask=offs<total
    x=tl.load(x_ptr+offs, mask=mask).to(tl.float32); gy=tl.load(gy_ptr+offs, mask=mask).to(tl.float32)
    c=0.7978845608028654; inner=c*(x+0.044715*x*x*x)
    e2=tl.exp(2.0*inner); tanh=(e2-1.0)/(e2+1.0); gp=c*(1.0+0.134145*x*x)
    dgelu=0.5*(1.0+tanh) + 0.5*x*(1.0-tanh*tanh)*gp
    tl.store(gx_ptr+offs, (gy*dgelu).to(gx_ptr.dtype.element_ty), mask=mask)
```

## Purpose: SPEEDUP
1.01x-1.27x faster than torch autograd (backward-only timed; bf16 gains more).
[`data/crawl-runs/h200/op-gelu-bwd-h200-results.md`](../../data/crawl-runs/h200/op-gelu-bwd-h200-results.md).

## H200 measured

| MxN | fp32 torch/Triton | bf16 torch/Triton |
|---|--:|--:|
| 4096x4096 | 1.20x | 1.27x |
| 8192x8192 | 1.07x | 1.14x |
| 8192x11008 | 1.04x | 1.12x |
| 8192x14336 | 1.03x | 1.08x |
| 16384x14336 | 1.01x | 1.06x |


## H200 benchmark replay (2026-07-21)

Original harness: `op_gelu_bwd.py`. The original harness completed on H200 with return code 0; its raw stdout/stderr is in the replay archive. Evidence: [`replay-2026-07-21-all-experimental.md`](../../data/crawl-runs/h200/replay-2026-07-21-all-experimental.md). All speed ratios are shape- and reference-specific.
