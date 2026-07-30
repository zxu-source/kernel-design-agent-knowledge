---
id: kernel-sigmoid
title: Sigmoid Forward (Hopper / H200)
type: kernel
architectures:
- sm90
tags:
- activation
confidence: experimental
reproducibility: benchmarked
kernel_types:
- fused-kernel
languages:
- triton
- python
related:
- kernel-relu
- kernel-silu-and-mul
sources:
- doc-nvidia-tuning-guide
performance_claims: []
blackwell_relevance: Same sigmoid on SM100; basic activation.
operator_purpose: speedup
what_it_does: 'Sigmoid: y=1/(1+exp(-x)); fused elementwise.'
h200_validation:
  date: 2026-07-21 (phase2; self-reported summary)
  evidence_file: data/crawl-runs/h200/op-sigmoid-h200-results.md
  gpu: H200
  arch: sm90
  correctness: PASS — err 1.2e-7 fp32 / bit-identical bf16.
  result: record-negative — 0.85x-0.99x torch (torch.sigmoid already a single fused
    op; ~parity).
evidence_basis: H200 benchmark replay on 2026-07-21. The original harness completed
  on H200 with return code 0; its raw stdout/stderr is in the replay archive. This
  remains a local derived implementation, not an upstream-source performance claim.
---
> **Evidence boundary.** This is a locally derived canonical implementation, not a captured upstream source file. Any H200 latency/correctness text below comes from a self-reported run summary; retain it for hypothesis generation, but replay it with raw benchmark output before treating it as benchmark evidence.


## Overview
Sigmoid: `y = 1/(1+exp(-x))`. A basic activation. torch.sigmoid is already a
single fused elementwise op, so the Triton kernel is ~parity.

```python
@triton.jit
def sigmoid_fwd(x_ptr, o_ptr, total, BLOCK):
    pid=tl.program_id(0); offs=pid*BLOCK+tl.arange(0,BLOCK); mask=offs<total
    x=tl.load(x_ptr+offs, mask=mask).to(tl.float32)
    tl.store(o_ptr+offs, (1.0/(1.0+tl.exp(-x))).to(o_ptr.dtype.element_ty), mask=mask)
```

## H200 measured: 0.85x-0.99x torch (~parity; torch.sigmoid already fused).


## H200 benchmark replay (2026-07-21)

Original harness: `op_groupnorm_sigmoid.py`. The original harness completed on H200 with return code 0; its raw stdout/stderr is in the replay archive. Evidence: [`replay-2026-07-21-all-experimental.md`](../../data/crawl-runs/h200/replay-2026-07-21-all-experimental.md). All speed ratios are shape- and reference-specific.
