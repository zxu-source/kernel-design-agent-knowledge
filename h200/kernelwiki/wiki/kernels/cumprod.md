---
id: kernel-cumprod
title: Cumulative Product (Hopper / H200)
type: kernel
architectures:
- sm90
tags:
- prefix-sum
- reduction
confidence: experimental
reproducibility: benchmarked
kernel_types:
- fused-kernel
languages:
- triton
- python
related:
- kernel-prefix-scan
sources:
- doc-nvidia-tuning-guide
performance_claims: []
blackwell_relevance: Same cumprod on SM100; scan building block.
operator_purpose: characterization
what_it_does: 'Cumprod: exp(cumsum(log(x))) per row (Triton 3.6 has no tl.cumprod).'
h200_validation:
  date: 2026-07-21 (phase2; self-reported summary)
  evidence_file: data/crawl-runs/h200/op-cumprod-h200-results.md
  gpu: H200
  arch: sm90
  correctness: PASS — rel ~2e-6 (exp(cumsum(log)) drift) vs torch.cumprod.
  result: 1.50x-3.28x faster than torch.cumprod.
evidence_basis: H200 benchmark replay on 2026-07-21. The original harness completed
  on H200 with return code 0; its raw stdout/stderr is in the replay archive. This
  remains a local derived implementation, not an upstream-source performance claim.
---
> **Evidence boundary.** This is a locally derived canonical implementation, not a captured upstream source file. Any H200 latency/correctness text below comes from a self-reported run summary; retain it for hypothesis generation, but replay it with raw benchmark output before treating it as benchmark evidence.


## Overview
Cumulative product per row via `exp(cumsum(log(x)))` (Triton 3.6 has no `tl.cumprod`).

```python
@triton.jit
def cumprod_k(x_ptr, o_ptr, N, BLOCK_N):
    row=tl.program_id(0); offs=tl.arange(0, BLOCK_N); mask=offs<N
    x=tl.load(x_ptr+row*N+offs, mask=mask, other=1.0).to(tl.float32)
    cp=tl.exp(tl.cumsum(tl.log(x), axis=0))
    tl.store(o_ptr+row*N+offs, cp.to(o_ptr.dtype.element_ty), mask=mask)
```

## H200 measured: 1.50x-3.28x faster than torch.cumprod (rel ~2e-6).


## H200 benchmark replay (2026-07-21)

Original harness: `op_kl_div_cumprod.py`. The original harness completed on H200 with return code 0; its raw stdout/stderr is in the replay archive. Evidence: [`replay-2026-07-21-all-experimental.md`](../../data/crawl-runs/h200/replay-2026-07-21-all-experimental.md). All speed ratios are shape- and reference-specific.
