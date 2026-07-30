---
id: kernel-group-norm
title: Group Normalization (Hopper / H200)
type: kernel
architectures:
- sm90
tags:
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
- kernel-layernorm-hopper
- kernel-rmsnorm-hopper
sources:
- doc-nvidia-tuning-guide
performance_claims: []
blackwell_relevance: Same GroupNorm on SM100; diffusion models (Stable Diffusion,
  Flux).
operator_purpose: speedup
what_it_does: 'GroupNorm [N,C] with G groups: normalize within each group per batch;
  fused.'
h200_validation:
  date: 2026-07-21 (phase2; self-reported summary)
  evidence_file: data/crawl-runs/h200/op-group-norm-h200-results.md
  gpu: H200
  arch: sm90
  correctness: PASS — err ~1e-6 vs torch.nn.functional.group_norm.
  result: 1.48x-5.29x faster than torch (bigger groups -> bigger fusion win).
evidence_basis: H200 benchmark replay on 2026-07-21. The original harness completed
  on H200 with return code 0; its raw stdout/stderr is in the replay archive. This
  remains a local derived implementation, not an upstream-source performance claim.
---
> **Evidence boundary.** This is a locally derived canonical implementation, not a captured upstream source file. Any H200 latency/correctness text below comes from a self-reported run summary; retain it for hypothesis generation, but replay it with raw benchmark output before treating it as benchmark evidence.


## Overview
GroupNorm: split C channels into G groups, normalize within each group per batch
element. Used in diffusion models (Stable Diffusion, Flux).

```python
@triton.jit
def group_norm(x_ptr, w_ptr, b_ptr, o_ptr, N, C, G, cg, eps, BLOCK_CG):
    ng=tl.program_id(0); n=ng//G; g=ng%G
    offs=tl.arange(0, BLOCK_CG); mask=offs<cg
    base=n*C + g*cg + offs
    x=tl.load(x_ptr+base, mask=mask).to(tl.float32)
    mean=tl.sum(x, axis=0)/cg; var=tl.sum((x-mean)*(x-mean), axis=0)/cg
    rrms=tl.rsqrt(var+eps)
    w=tl.load(w_ptr+g*cg+offs, mask=mask).to(tl.float32)
    b=tl.load(b_ptr+g*cg+offs, mask=mask).to(tl.float32)
    tl.store(o_ptr+base, ((x-mean)*rrms*w+b).to(o_ptr.dtype.element_ty), mask=mask)
```

## H200 measured: 1.48x-5.29x faster than torch (bigger groups -> bigger win).


## H200 benchmark replay (2026-07-21)

Original harness: `op_groupnorm_sigmoid.py`. The original harness completed on H200 with return code 0; its raw stdout/stderr is in the replay archive. Evidence: [`replay-2026-07-21-all-experimental.md`](../../data/crawl-runs/h200/replay-2026-07-21-all-experimental.md). All speed ratios are shape- and reference-specific.
