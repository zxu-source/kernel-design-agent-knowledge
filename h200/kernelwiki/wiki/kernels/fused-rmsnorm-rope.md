---
id: kernel-fused-rmsnorm-rope
title: Fused RMSNorm + RoPE (Hopper / H200)
type: kernel
architectures:
- sm90
tags:
- rmsnorm
- normalization
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
- kernel-rope
sources:
- doc-nvidia-tuning-guide
performance_claims: []
blackwell_relevance: Same fused pattern on SM100; transformer pre-attention fusion.
operator_purpose: speedup
what_it_does: 'Fused RMSNorm+RoPE: normalize then positional-rotate in one kernel
  (vs torch 2-op).'
h200_validation:
  date: 2026-07-21 (phase2; self-reported summary)
  correctness: PASS — err ~1e-6 vs torch(rms_norm+rope).
  result: 1.77x-2.74x faster than torch (2 ops fused into 1 pass).
evidence_basis: H200 benchmark replay on 2026-07-21. The original harness completed
  on H200 with return code 0; its raw stdout/stderr is in the replay archive. This
  remains a local derived implementation, not an upstream-source performance claim.
---
> **Evidence boundary.** This is a locally derived canonical implementation, not a captured upstream source file. Any H200 latency/correctness text below comes from a self-reported run summary; retain it for hypothesis generation, but replay it with raw benchmark output before treating it as benchmark evidence.


## Overview
Fuses RMSNorm (squared-mean reduction, normalize, weight-scale) and RoPE
(rotary position embedding, rotate-half) into one kernel — one read + one write
of x, vs torch's separate `rms_norm` then `rope` (2 passes + intermediate).

```python
@triton.jit
def fused_rmsnorm_rope(x_ptr, w_ptr, cos_ptr, sin_ptr, o_ptr, eps, NH, S, D, stride_bh, stride_row, BLOCK_D):
    row=tl.program_id(0); d=tl.arange(0, BLOCK_D); mask=d<D; half=D//2
    x=tl.load(x_ptr+row*D+d, mask=mask).to(tl.float32)
    var=tl.sum(x*x,axis=0)/D; rrms=tl.rsqrt(var+eps)
    w=tl.load(w_ptr+d,mask=mask).to(tl.float32); xn=x*rrms*w     # RMSNorm
    s=row%S
    cos=tl.load(cos_ptr+s*half+(d%half),mask=mask).to(tl.float32)
    sin=tl.load(sin_ptr+s*half+(d%half),mask=mask).to(tl.float32)
    lo=d<half; partner=tl.where(lo, d+half, d-half)               # RoPE
    xp=tl.load(x_ptr+row*D+partner,mask=mask).to(tl.float32)*rrms*tl.load(w_ptr+partner,mask=mask).to(tl.float32)
    tl.store(o_ptr+row*D+d, tl.where(lo, xn*cos-xp*sin, xn*cos+xp*sin).to(o_ptr.dtype.element_ty), mask=mask)
```
## H200 measured: 1.77x-2.74x faster than torch.


## H200 benchmark replay (2026-07-21)

Original harness: `op_fused_ln_gelu_rmsnorm_rope.py`. The original harness completed on H200 with return code 0; its raw stdout/stderr is in the replay archive. Evidence: [`replay-2026-07-21-all-experimental.md`](../../data/crawl-runs/h200/replay-2026-07-21-all-experimental.md). All speed ratios are shape- and reference-specific.
