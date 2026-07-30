---
id: kernel-bce-loss
title: BCE-with-Logits Loss (Hopper / H200)
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
- kernel-cross-entropy
- kernel-online-softmax
sources:
- doc-nvidia-tuning-guide
performance_claims: []
blackwell_relevance: Same BCE on SM100; binary classification / multi-label loss.
operator_purpose: speedup
what_it_does: 'BCE-with-logits: loss=max(x,0)-x*t+log(1+exp(-|x|)); numerically stable
  fused elementwise.'
h200_validation:
  date: 2026-07-21 (phase2; self-reported summary)
  evidence_file: data/crawl-runs/h200/op-bce-loss-h200-results.md
  harness_dir: artifacts/kernels/bce-loss/variants
  gpu: H200
  arch: sm90
  toolchain: Triton 3.6.0, PyTorch 2.11.0+cu130
  correctness: PASS — err ~5e-7 vs torch.nn.functional.binary_cross_entropy_with_logits.
  result: 3.03x-3.74x faster than torch (fused numerically-stable form vs multi-op).
  scope: fp32, MLP shapes, on H200.
evidence_basis: H200 benchmark replay on 2026-07-21. The original harness completed
  on H200 with return code 0; its raw stdout/stderr is in the replay archive. This
  remains a local derived implementation, not an upstream-source performance claim.
---
> **Evidence boundary.** This is a locally derived canonical implementation, not a captured upstream source file. Any H200 latency/correctness text below comes from a self-reported run summary; retain it for hypothesis generation, but replay it with raw benchmark output before treating it as benchmark evidence.


## Overview

Binary cross-entropy with logits (numerically stable): `loss = max(x,0) - x*t +
log(1+exp(-|x|))`. Avoids `log(0)` and `sigmoid` overflow. One fused elementwise
pass vs torch's multi-op chain (sigmoid + log + elementwise mul/sub).

```python
@triton.jit
def bce_logits(x_ptr, t_ptr, o_ptr, total, BLOCK):
    pid=tl.program_id(0); offs=pid*BLOCK+tl.arange(0,BLOCK); mask=offs<total
    x=tl.load(x_ptr+offs, mask=mask).to(tl.float32); t=tl.load(t_ptr+offs, mask=mask).to(tl.float32)
    loss = tl.maximum(x, 0.0) - x*t + tl.log(1.0 + tl.exp(-tl.abs(x)))
    tl.store(o_ptr+offs, loss, mask=mask)
```

## Purpose: SPEEDUP (fusion + numerical stability)
3.03x-3.74x faster than torch: the fused numerically-stable form reads `logits`
and `target` once each, vs torch's multi-op (sigmoid + log + mul + sub, each a
separate pass + intermediate). The `max(x,0) + log(1+exp(-|x|))` form also avoids
`log(0)` (robustness).
[`data/crawl-runs/h200/op-bce-loss-h200-results.md`](../../data/crawl-runs/h200/op-bce-loss-h200-results.md).

## H200 measured

| M x N | torch/Triton |
|---|--:|
| 4096x4096 | 3.03x |
| 8192x8192 | 3.57x |
| 8192x14336 | 3.66x |
| 16384x14336 | 3.74x |


## H200 benchmark replay (2026-07-21)

Original harness: `op_bce.py`. The original harness completed on H200 with return code 0; its raw stdout/stderr is in the replay archive. Evidence: [`replay-2026-07-21-all-experimental.md`](../../data/crawl-runs/h200/replay-2026-07-21-all-experimental.md). All speed ratios are shape- and reference-specific.
