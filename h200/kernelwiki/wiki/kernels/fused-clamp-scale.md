---
id: kernel-fused-clamp-scale
title: Fused Clamp + Scale (Hopper / H200)
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
- kernel-fp8-quant
- kernel-per-tensor-quant
sources:
- doc-nvidia-tuning-guide
performance_claims: []
blackwell_relevance: Same clamp+scale on SM100; mixed-precision safety (clamp before
  quant).
operator_purpose: both
what_it_does: 'Fused clamp+scale: out = clamp(x, lo, hi) * scale; mixed-precision
  safety pattern.'
h200_validation:
  date: 2026-07-21 (phase2; self-reported summary)
  correctness: PASS — bit-identical (err=0).
  result: 1.61x-1.86x faster than torch (clamp then *scale, 2 ops fused).
evidence_basis: H200 benchmark replay on 2026-07-21. The original harness completed
  on H200 with return code 0; its raw stdout/stderr is in the replay archive. This
  remains a local derived implementation, not an upstream-source performance claim.
---
> **Evidence boundary.** This is a locally derived canonical implementation, not a captured upstream source file. Any H200 latency/correctness text below comes from a self-reported run summary; retain it for hypothesis generation, but replay it with raw benchmark output before treating it as benchmark evidence.


## Overview
Fused clamp + scale: `out = clamp(x, lo, hi) * scale`. Used in mixed-precision
safety paths (clamp activations to a safe range before quantizing to fp8/int8).

```python
@triton.jit
def fused_clamp_scale(x_ptr, o_ptr, lo, hi, scale, total, BLOCK):
    pid=tl.program_id(0); offs=pid*BLOCK+tl.arange(0,BLOCK); mask=offs<total
    x=tl.load(x_ptr+offs,mask=mask).to(tl.float32)
    clamped=tl.minimum(tl.maximum(x, lo), hi)
    tl.store(o_ptr+offs, (clamped*scale).to(o_ptr.dtype.element_ty), mask=mask)
```

## H200 measured: 1.61x-1.86x faster than torch.


## H200 benchmark replay (2026-07-21)

Original harness: `op_fused_addcmul_clampscale.py`. The original harness completed on H200 with return code 0; its raw stdout/stderr is in the replay archive. Evidence: [`replay-2026-07-21-all-experimental.md`](../../data/crawl-runs/h200/replay-2026-07-21-all-experimental.md). All speed ratios are shape- and reference-specific.
