---
id: kernel-cross-entropy
title: Cross-Entropy Loss Forward (Hopper / H200)
type: kernel
architectures:
- sm90
tags:
- softmax
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
- kernel-online-softmax
- kernel-argmax
sources:
- doc-nvidia-tuning-guide
performance_claims: []
blackwell_relevance: Same cross-entropy on SM100; training loss head.
operator_purpose: speedup
what_it_does: 'Cross-entropy forward: loss=logsumexp(logits)-logits[target]; fused
  logsumexp+gather per row.'
h200_validation:
  date: 2026-07-21 (phase2; self-reported summary)
  evidence_file: data/crawl-runs/h200/op-cross-entropy-h200-results.md
  harness_dir: artifacts/kernels/cross-entropy/variants
  gpu: H200
  arch: sm90
  toolchain: Triton 3.6.0, PyTorch 2.11.0+cu130
  correctness: PASS — err 1.9e-6 vs torch.nn.functional.cross_entropy.
  result: 2.03x-2.43x faster than torch at moderate vocab; 0.34x at D=128K (BLOCK=131072
    spill).
  scope: fp32, vocab up to 128K, on H200.
evidence_basis: H200 benchmark replay on 2026-07-21. The original harness completed
  on H200 with return code 0; its raw stdout/stderr is in the replay archive. This
  remains a local derived implementation, not an upstream-source performance claim.
---
> **Evidence boundary.** This is a locally derived canonical implementation, not a captured upstream source file. Any H200 latency/correctness text below comes from a self-reported run summary; retain it for hypothesis generation, but replay it with raw benchmark output before treating it as benchmark evidence.


## Overview

Cross-entropy forward (training loss head): `loss[i] = logsumexp(logits[i]) -
logits[i, target[i]]`. Fuses the logsumexp reduction + the target gather into one
pass per row (vs torch's log_softmax + gather + neg).

```python
@triton.jit
def ce_fwd(logits_ptr, tgt_ptr, loss_ptr, D, BLOCK_D):
    row=tl.program_id(0); offs=tl.arange(0, BLOCK_D); mask=offs<D
    x=tl.load(logits_ptr+row*D+offs, mask=mask, other=-1e30).to(tl.float32)
    m=tl.max(x, axis=0); e=tl.where(mask, tl.exp(x-m), 0.0)
    lse=m+tl.log(tl.sum(e, axis=0))
    tgt=tl.load(tgt_ptr+row); xt=tl.load(logits_ptr+row*D+tgt).to(tl.float32)
    tl.store(loss_ptr+row, lse-xt)
```

## Purpose: SPEEDUP
2.03x-2.43x faster than torch at moderate vocab (fused logsumexp+gather). 0.34x
at D=128K (BLOCK=131072 spills registers — same large-N issue as softmax/scan;
tile the reduction for huge vocab).
[`data/crawl-runs/h200/op-cross-entropy-h200-results.md`](../../data/crawl-runs/h200/op-cross-entropy-h200-results.md).

## H200 measured

| M | D | torch/Triton |
|--:|--:|--:|
| 4096 | 4096 | 2.03x |
| 8192 | 32000 | 2.17x |
| 8192 | 128256 | 0.34x |
| 4096 | 128256 | 0.34x |
| 8192 | 4096 | 2.43x |


## H200 benchmark replay (2026-07-21)

Original harness: `op_cross_entropy.py`. The original harness completed on H200 with return code 0; its raw stdout/stderr is in the replay archive. Evidence: [`replay-2026-07-21-all-experimental.md`](../../data/crawl-runs/h200/replay-2026-07-21-all-experimental.md). All speed ratios are shape- and reference-specific.
