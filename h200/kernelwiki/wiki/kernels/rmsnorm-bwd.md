---
id: kernel-rmsnorm-bwd
title: RMSNorm Backward (Hopper / H200)
type: kernel
architectures:
- sm90
tags:
- rmsnorm
- normalization
- reduction
- kernel-fusion
confidence: experimental
reproducibility: benchmarked
kernel_types:
- fused-kernel
languages:
- triton
- python
related:
- kernel-rmsnorm-hopper
sources:
- doc-nvidia-tuning-guide
performance_claims: []
blackwell_relevance: Same RMSNorm backward on SM100; training gradient.
operator_purpose: speedup
what_it_does: 'RMSNorm backward: grad_x=rrms*(w*gy - x*rrms^2*mean(gy*w*x)); grad_w=sum_rows(gy*x*rrms).'
h200_validation:
  date: 2026-07-21 (phase2; self-reported summary)
  evidence_file: data/crawl-runs/h200/op-rmsnorm-bwd-h200-results.md
  harness_dir: artifacts/kernels/rmsnorm-bwd/variants
  gpu: H200
  arch: sm90
  toolchain: Triton 3.6.0, PyTorch 2.11.0+cu130
  correctness: PASS — grad_x err ~2e-6; grad_w err ~1e-3 (atomic-add accumulation
    order).
  result: 1.34x-1.66x faster than torch at moderate N; 0.28x-0.35x at large N (BLOCK
    spill + atomic grad_w).
  scope: fp32, hidden 4096..14336, on H200.
evidence_basis: H200 benchmark replay on 2026-07-21. The original harness completed
  on H200 with return code 0; its raw stdout/stderr is in the replay archive. This
  remains a local derived implementation, not an upstream-source performance claim.
---
> **Evidence boundary.** This is a locally derived canonical implementation, not a captured upstream source file. Any H200 latency/correctness text below comes from a self-reported run summary; retain it for hypothesis generation, but replay it with raw benchmark output before treating it as benchmark evidence.


## Overview

RMSNorm training gradient (one fused pass per row + atomic-add grad_w):
`grad_x = rrms*(w*grad_y - x*rrms^2 * mean(grad_y*w*x))`,
`grad_w += sum(grad_y*x*rrms)`. Note: the variance term (`x*rrms^2*c`) carries
**no** `w` (a common bug — the `w` only multiplies `grad_y`).

```python
@triton.jit
def rmsnorm_bwd(x_ptr, w_ptr, gy_ptr, gx_ptr, gw_ptr, eps, M, N, BLOCK_N):
    row=tl.program_id(0); offs=tl.arange(0, BLOCK_N); mask=offs<N
    x=tl.load(x_ptr+row*N+offs, mask=mask).to(tl.float32)
    gy=tl.load(gy_ptr+row*N+offs, mask=mask).to(tl.float32)
    w=tl.load(w_ptr+offs, mask=mask).to(tl.float32)
    var=tl.sum(x*x,axis=0)/N; rrms=tl.rsqrt(var+eps)
    g2w=gy*w; c=tl.sum(g2w*x,axis=0)/N
    gx=rrms*(w*gy - x*rrms*rrms*c)            # variance term has NO w
    tl.store(gx_ptr+row*N+offs, gx.to(gx_ptr.dtype.element_ty), mask=mask)
    tl.atomic_add(gw_ptr+offs, (gy*x*rrms), mask=mask)
```

## Purpose: SPEEDUP
1.34x-1.66x faster than torch autograd at moderate N. 0.28x-0.35x at large N
(N>=11008): BLOCK=16384 spills registers and the atomic-add grad_w adds overhead.
[`data/crawl-runs/h200/op-rmsnorm-bwd-h200-results.md`](../../data/crawl-runs/h200/op-rmsnorm-bwd-h200-results.md).

## H200 measured

| M x N | torch/Triton |
|---|--:|
| 4096x4096 | 1.34x |
| 8192x8192 | 1.66x |
| 8192x11008 | 0.28x |
| 8192x14336 | 0.35x |
| 16384x14336 | 0.35x |


## H200 benchmark replay (2026-07-21)

Original harness: `op_rmsnorm_bwd.py`. The original harness completed on H200 with return code 0; its raw stdout/stderr is in the replay archive. Evidence: [`replay-2026-07-21-all-experimental.md`](../../data/crawl-runs/h200/replay-2026-07-21-all-experimental.md). All speed ratios are shape- and reference-specific.
