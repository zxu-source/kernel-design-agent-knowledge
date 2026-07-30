---
id: kernel-fused-add-silu
title: Fused Add + SiLU (Hopper / H200)
type: kernel
architectures:
- sm90
tags:
- silu
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
- kernel-fused-prenorm
sources:
- doc-nvidia-tuning-guide
performance_claims: []
blackwell_relevance: Same fused add+silu on SM100; transformer MLP elementwise fusion.
operator_purpose: speedup
what_it_does: 'Fused add+SiLU: out = silu(a + b); two-input elementwise fusion.'
h200_validation:
  date: 2026-07-21 (phase2; self-reported summary)
  correctness: PASS — err ~2e-7 fp32 / ~1e-2 bf16.
  result: 1.36x-1.66x faster than torch (add then silu, 2 ops fused).
evidence_basis: H200 benchmark replay on 2026-07-21. The original harness completed
  on H200 with return code 0; its raw stdout/stderr is in the replay archive. This
  remains a local derived implementation, not an upstream-source performance claim.
---
> **Evidence boundary.** This is a locally derived canonical implementation, not a captured upstream source file. Any H200 latency/correctness text below comes from a self-reported run summary; retain it for hypothesis generation, but replay it with raw benchmark output before treating it as benchmark evidence.


## Overview
Fused two-input elementwise: `out = silu(a + b)`. Common in transformer MLP
where a residual is added then activated. Fused into one kernel (2 reads + 1
write, no intermediate) vs torch's separate add + silu (2 passes + intermediate).

```python
@triton.jit
def fused_add_silu(a_ptr, b_ptr, o_ptr, total, BLOCK):
    pid=tl.program_id(0); offs=pid*BLOCK+tl.arange(0,BLOCK); mask=offs<total
    a=tl.load(a_ptr+offs,mask=mask).to(tl.float32); b=tl.load(b_ptr+offs,mask=mask).to(tl.float32)
    s=a+b; sig=1.0/(1.0+tl.exp(-s))
    tl.store(o_ptr+offs, (s*sig).to(o_ptr.dtype.element_ty), mask=mask)
```

## H200 measured: 1.36x-1.66x faster than torch.


## H200 benchmark replay (2026-07-21)

Original harness: `op_fused_prenorm_addsilu.py`. The original harness completed on H200 with return code 0; its raw stdout/stderr is in the replay archive. Evidence: [`replay-2026-07-21-all-experimental.md`](../../data/crawl-runs/h200/replay-2026-07-21-all-experimental.md). All speed ratios are shape- and reference-specific.
