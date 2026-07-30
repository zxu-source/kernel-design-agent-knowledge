---
id: kernel-kl-div
title: KL Divergence Loss (Hopper / H200)
type: kernel
architectures:
- sm90
tags:
- softmax
- kernel-fusion
confidence: experimental
reproducibility: benchmarked
kernel_types:
- fused-kernel
languages:
- triton
- python
related:
- kernel-cross-entropy
- kernel-online-softmax
sources:
- doc-nvidia-tuning-guide
performance_claims: []
blackwell_relevance: Same KL-div on SM100; distillation/sampling loss.
operator_purpose: speedup
what_it_does: 'KL divergence: fused log_softmax + target*(log(target)-log_sm) per
  row.'
h200_validation:
  date: 2026-07-21 (phase2; self-reported summary)
  evidence_file: data/crawl-runs/h200/op-kl-div-h200-results.md
  gpu: H200
  arch: sm90
  correctness: PASS — err ~3e-8 vs torch.nn.functional.kl_div.
  result: 2.96x-3.07x faster than torch at moderate N; 0.55x at N=32K (BLOCK spill).
evidence_basis: H200 benchmark replay on 2026-07-21. The original harness completed
  on H200 with return code 0; its raw stdout/stderr is in the replay archive. This
  remains a local derived implementation, not an upstream-source performance claim.
---
> **Evidence boundary.** This is a locally derived canonical implementation, not a captured upstream source file. Any H200 latency/correctness text below comes from a self-reported run summary; retain it for hypothesis generation, but replay it with raw benchmark output before treating it as benchmark evidence.


## Overview
KL divergence (distillation/sampling loss): `loss = target * (log(target) - log_softmax(logits))`.
Fused log_softmax + target gather.

```python
@triton.jit
def kl_div(logit_ptr, tgt_ptr, o_ptr, N, BLOCK_N):
    row=tl.program_id(0); offs=tl.arange(0, BLOCK_N); mask=offs<N
    x=tl.load(logit_ptr+row*N+offs, mask=mask, other=-1e30).to(tl.float32)
    t=tl.load(tgt_ptr+row*N+offs, mask=mask, other=0.0).to(tl.float32)
    m=tl.max(x, axis=0); e=tl.where(mask, tl.exp(x-m), 0.0); lse=m+tl.log(tl.sum(e, axis=0))
    loss = t * (tl.log(t+1e-30) - (x-lse))
    tl.store(o_ptr+row*N+offs, loss, mask=mask)
```

## H200 measured: 2.96x-3.07x moderate N; 0.55x at N=32K.


## H200 benchmark replay (2026-07-21)

Original harness: `op_kl_div_cumprod.py`. The original harness completed on H200 with return code 0; its raw stdout/stderr is in the replay archive. Evidence: [`replay-2026-07-21-all-experimental.md`](../../data/crawl-runs/h200/replay-2026-07-21-all-experimental.md). All speed ratios are shape- and reference-specific.
