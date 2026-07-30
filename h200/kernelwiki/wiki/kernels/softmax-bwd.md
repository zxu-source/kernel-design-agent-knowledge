---
id: kernel-softmax-bwd
title: Softmax Backward (Hopper / H200)
type: kernel
architectures:
- sm90
tags:
- softmax
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
- kernel-online-softmax
sources:
- doc-nvidia-tuning-guide
performance_claims: []
blackwell_relevance: Same softmax backward on SM100; training gradient.
operator_purpose: speedup
what_it_does: 'Softmax backward: grad_x = (grad_y - sum(grad_y*y)) * y, one fused
  pass per row.'
h200_validation:
  date: 2026-07-21 (phase2; self-reported summary)
  evidence_file: data/crawl-runs/h200/op-softmax-bwd-h200-results.md
  harness_dir: artifacts/kernels/softmax-bwd/variants
  gpu: H200
  arch: sm90
  toolchain: Triton 3.6.0, PyTorch 2.11.0+cu130
  correctness: PASS — rel ~7e-8 vs torch autograd.
  result: 3.35x-4.04x faster than torch at moderate N; 0.70x at N=32000 (BLOCK=32768
    spill).
  scope: fp32, vocab up to 32000, on H200.
evidence_basis: H200 benchmark replay on 2026-07-21. The original harness completed
  on H200 with return code 0; its raw stdout/stderr is in the replay archive. This
  remains a local derived implementation, not an upstream-source performance claim.
---
> **Evidence boundary.** This is a locally derived canonical implementation, not a captured upstream source file. Any H200 latency/correctness text below comes from a self-reported run summary; retain it for hypothesis generation, but replay it with raw benchmark output before treating it as benchmark evidence.


## Overview

Softmax backward (training gradient): `grad_x = (grad_y - sum(grad_y * y)) * y`
where `y = softmax(x)`. One fused pass per row.

```python
@triton.jit
def softmax_bwd(y_ptr, gy_ptr, gx_ptr, N, BLOCK_N: tl.constexpr):
    row=tl.program_id(0)
    offs=tl.arange(0, BLOCK_N); mask=offs<N
    y=tl.load(y_ptr+row*N+offs, mask=mask).to(tl.float32)
    gy=tl.load(gy_ptr+row*N+offs, mask=mask).to(tl.float32)
    s=tl.sum(gy*y, axis=0)
    tl.store(gx_ptr+row*N+offs, ((gy-s)*y).to(gx_ptr.dtype.element_ty), mask=mask)
```

## Purpose: SPEEDUP
3.35x-4.04x faster than torch autograd at moderate N. 0.70x at N=32000
(BLOCK=32768 register spill — same large-N issue as forward softmax; tile for
huge N).
[`data/crawl-runs/h200/op-softmax-bwd-h200-results.md`](../../data/crawl-runs/h200/op-softmax-bwd-h200-results.md).

## H200 measured

| M | N | torch/Triton |
|--:|--:|--:|
| 4096 | 4096 | 3.35x |
| 8192 | 8192 | 4.04x |
| 4096 | 32000 | 0.70x |
| 8192 | 32000 | 0.70x |


## H200 benchmark replay (2026-07-21)

Original harness: `op_softmax_bwd.py`. The original harness completed on H200 with return code 0; its raw stdout/stderr is in the replay archive. Evidence: [`replay-2026-07-21-all-experimental.md`](../../data/crawl-runs/h200/replay-2026-07-21-all-experimental.md). All speed ratios are shape- and reference-specific.
