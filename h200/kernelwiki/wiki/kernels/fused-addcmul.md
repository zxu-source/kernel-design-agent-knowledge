---
id: kernel-fused-addcmul
title: Fused Addcmul (c+a*b, Hopper / H200)
type: kernel
architectures:
- sm90
tags:
- kernel-fusion
confidence: experimental
reproducibility: benchmarked
kernel_types:
- fused-kernel
languages:
- triton
- python
related:
- kernel-adam-step
- kernel-silu-and-mul
sources:
- doc-nvidia-tuning-guide
performance_claims: []
blackwell_relevance: Same addcmul on SM100; FMA elementwise (optimizer/interpolation).
operator_purpose: speedup
what_it_does: 'Fused addcmul: out = c + a*b; FMA elementwise (optimizer helper, interpolation).'
h200_validation:
  date: 2026-07-21 (phase2; self-reported summary)
  correctness: PASS — bit-identical (err=0).
  result: record-negative — 0.86x-0.98x torch (torch.addcmul already a fused single
    op; ~parity).
evidence_basis: H200 benchmark replay on 2026-07-21. The original harness completed
  on H200 with return code 0; its raw stdout/stderr is in the replay archive. This
  remains a local derived implementation, not an upstream-source performance claim.
---
> **Evidence boundary.** This is a locally derived canonical implementation, not a captured upstream source file. Any H200 latency/correctness text below comes from a self-reported run summary; retain it for hypothesis generation, but replay it with raw benchmark output before treating it as benchmark evidence.


## Overview
Fused multiply-add: `out = c + a*b` (the FMA pattern; used in optimizer steps,
linear interpolation). torch.addcmul is already a single fused op, so this is
~parity — the value is the validated self-contained impl.

```python
@triton.jit
def fused_addcmul(c_ptr, a_ptr, b_ptr, o_ptr, total, BLOCK):
    pid=tl.program_id(0); offs=pid*BLOCK+tl.arange(0,BLOCK); mask=offs<total
    c=tl.load(c_ptr+offs,mask=mask).to(tl.float32)
    a=tl.load(a_ptr+offs,mask=mask).to(tl.float32)
    b=tl.load(b_ptr+offs,mask=mask).to(tl.float32)
    tl.store(o_ptr+offs, (c+a*b).to(o_ptr.dtype.element_ty), mask=mask)
```

## H200 measured: 0.86x-0.98x torch (~parity; torch.addcmul already fused).


## H200 benchmark replay (2026-07-21)

Original harness: `op_fused_addcmul_clampscale.py`. The original harness completed on H200 with return code 0; its raw stdout/stderr is in the replay archive. Evidence: [`replay-2026-07-21-all-experimental.md`](../../data/crawl-runs/h200/replay-2026-07-21-all-experimental.md). All speed ratios are shape- and reference-specific.
