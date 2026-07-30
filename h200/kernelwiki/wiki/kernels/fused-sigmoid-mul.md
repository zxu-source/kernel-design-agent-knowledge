---
id: kernel-fused-sigmoid-mul
title: Fused Sigmoid * Mul (GRU/LSTM Gate, Hopper / H200)
type: kernel
architectures:
- sm90
tags:
- kernel-fusion
- activation
confidence: experimental
reproducibility: benchmarked
kernel_types:
- fused-kernel
languages:
- triton
- python
related:
- kernel-silu-and-mul
- kernel-sigmoid
sources:
- doc-nvidia-tuning-guide
performance_claims: []
blackwell_relevance: Same sigmoid*mul on SM100; GRU/LSTM gating mechanism.
operator_purpose: speedup
what_it_does: 'Fused sigmoid*mul: out = x * sigmoid(gate); GRU/LSTM gate mechanism.'
h200_validation:
  date: 2026-07-21 (phase2; self-reported summary)
  correctness: PASS — err ~2e-7 fp32 / ~4e-3 bf16.
  result: 1.37x-1.67x faster than torch (x * sigmoid(gate), 2 ops fused).
evidence_basis: H200 benchmark replay on 2026-07-21. The original harness completed
  on H200 with return code 0; its raw stdout/stderr is in the replay archive. This
  remains a local derived implementation, not an upstream-source performance claim.
---
> **Evidence boundary.** This is a locally derived canonical implementation, not a captured upstream source file. Any H200 latency/correctness text below comes from a self-reported run summary; retain it for hypothesis generation, but replay it with raw benchmark output before treating it as benchmark evidence.


## Overview
GRU/LSTM gate: `out = x * sigmoid(gate)`. Fuses the sigmoid + elementwise mul
into one kernel (2 reads + 1 write, no intermediate sigmoid tensor) vs torch's
separate sigmoid then mul.

```python
@triton.jit
def fused_sigmoid_mul(x_ptr, g_ptr, o_ptr, total, BLOCK):
    pid=tl.program_id(0); offs=pid*BLOCK+tl.arange(0,BLOCK); mask=offs<total
    x=tl.load(x_ptr+offs,mask=mask).to(tl.float32); g=tl.load(g_ptr+offs,mask=mask).to(tl.float32)
    tl.store(o_ptr+offs, (x*(1.0/(1.0+tl.exp(-g)))).to(o_ptr.dtype.element_ty), mask=mask)
```

## H200 measured: 1.37x-1.67x faster than torch.


## H200 benchmark replay (2026-07-21)

Original harness: `op_fused_qkt_sigmul.py`. The original harness completed on H200 with return code 0; its raw stdout/stderr is in the replay archive. Evidence: [`replay-2026-07-21-all-experimental.md`](../../data/crawl-runs/h200/replay-2026-07-21-all-experimental.md). All speed ratios are shape- and reference-specific.
