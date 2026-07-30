---
id: kernel-fused-prenorm
title: Fused Pre-Norm Residual (x + LayerNorm(x), Hopper / H200)
type: kernel
architectures:
- sm90
tags:
- layernorm
- kernel-fusion
- normalization
confidence: experimental
reproducibility: benchmarked
kernel_types:
- fused-kernel
languages:
- triton
- python
related:
- kernel-layernorm-hopper
- kernel-fused-add-rmsnorm
sources:
- doc-nvidia-tuning-guide
performance_claims: []
blackwell_relevance: Same pre-norm residual on SM100; GPT-style transformer block.
operator_purpose: speedup
what_it_does: 'Fused pre-norm residual: out = x + LayerNorm(x, w, b); one pass (vs
  torch 2-op).'
h200_validation:
  date: 2026-07-21 (phase2; self-reported summary)
  correctness: PASS — err ~1e-6.
  result: 2.23x-2.86x faster than torch (x + layer_norm(x), 2 ops fused).
evidence_basis: H200 benchmark replay on 2026-07-21. The original harness completed
  on H200 with return code 0; its raw stdout/stderr is in the replay archive. This
  remains a local derived implementation, not an upstream-source performance claim.
---
> **Evidence boundary.** This is a locally derived canonical implementation, not a captured upstream source file. Any H200 latency/correctness text below comes from a self-reported run summary; retain it for hypothesis generation, but replay it with raw benchmark output before treating it as benchmark evidence.


## Overview
Pre-norm residual pattern (GPT-style transformers): `out = x + LayerNorm(x, w, b)`.
The residual `x` is added AFTER the norm. Fused into one kernel (load x once,
norm, add residual, store) vs torch's separate `layer_norm` + add (2 passes +
intermediate).

```python
@triton.jit
def fused_prenorm(x_ptr, w_ptr, b_ptr, o_ptr, eps, N, BLOCK_N):
    row=tl.program_id(0); offs=tl.arange(0, BLOCK_N); mask=offs<N
    x=tl.load(x_ptr+row*N+offs, mask=mask).to(tl.float32)
    mean=tl.sum(x,axis=0)/N; var=tl.sum((x-mean)*(x-mean),axis=0)/N; rrms=tl.rsqrt(var+eps)
    w=tl.load(w_ptr+offs,mask=mask).to(tl.float32); b=tl.load(b_ptr+offs,mask=mask).to(tl.float32)
    tl.store(o_ptr+row*N+offs, (x+(x-mean)*rrms*w+b).to(o_ptr.dtype.element_ty), mask=mask)
```

## H200 measured: 2.23x-2.86x faster than torch.


## H200 benchmark replay (2026-07-21)

Original harness: `op_fused_prenorm_addsilu.py`. The original harness completed on H200 with return code 0; its raw stdout/stderr is in the replay archive. Evidence: [`replay-2026-07-21-all-experimental.md`](../../data/crawl-runs/h200/replay-2026-07-21-all-experimental.md). All speed ratios are shape- and reference-specific.
