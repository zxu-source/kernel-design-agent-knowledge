---
id: kernel-cosine-sim
title: Cosine Similarity (fused dot+norms, Hopper / H200)
type: kernel
architectures:
- sm90
tags:
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
- kernel-l2-normalize
- kernel-block-reduce
sources:
- doc-nvidia-tuning-guide
performance_claims: []
blackwell_relevance: Same cosine similarity on SM100; retrieval/embedding similarity.
operator_purpose: speedup
what_it_does: 'Cosine similarity (fused dot + L2-norms): y=a.b/(||a||||b||+eps) per
  row.'
h200_validation:
  date: 2026-07-21 (phase2; self-reported summary)
  evidence_file: data/crawl-runs/h200/op-cosine-sim-h200-results.md
  harness_dir: artifacts/kernels/cosine-sim/variants
  gpu: H200
  arch: sm90
  toolchain: Triton 3.6.0, PyTorch 2.11.0+cu130
  correctness: PASS — err ~1e-8 vs torch.nn.functional.cosine_similarity.
  result: 5.26x-6.67x faster than torch (fused single-pass dot+2-norms+divide vs multi-op).
  scope: fp32, hidden 4096..14336, on H200.
evidence_basis: H200 benchmark replay on 2026-07-21. The original harness completed
  on H200 with return code 0; its raw stdout/stderr is in the replay archive. This
  remains a local derived implementation, not an upstream-source performance claim.
---
> **Evidence boundary.** This is a locally derived canonical implementation, not a captured upstream source file. Any H200 latency/correctness text below comes from a self-reported run summary; retain it for hypothesis generation, but replay it with raw benchmark output before treating it as benchmark evidence.


## Overview

Cosine similarity per row: `y = (a·b) / (||a|| ||b|| + eps)`. One fused pass per
row computes the dot product and both L2 norms together (reads `a` and `b` once
each), then divides.

```python
@triton.jit
def cos_sim(a_ptr, b_ptr, o_ptr, eps, N, BLOCK_N):
    row=tl.program_id(0); offs=tl.arange(0, BLOCK_N); mask=offs<N
    a=tl.load(a_ptr+row*N+offs, mask=mask).to(tl.float32); b=tl.load(b_ptr+row*N+offs, mask=mask).to(tl.float32)
    dot=tl.sum(a*b, axis=0); na=tl.sqrt(tl.sum(a*a, axis=0)); nb=tl.sqrt(tl.sum(b*b, axis=0))
    tl.store(o_ptr+row, dot/(na*nb+eps))
```

## Purpose: SPEEDUP (fusion)
5.26x-6.67x faster than torch: torch computes dot + norm_a + norm_b + divide as
multiple passes (re-reading a and b); the fused kernel does all four in one pass.
[`data/crawl-runs/h200/op-cosine-sim-h200-results.md`](../../data/crawl-runs/h200/op-cosine-sim-h200-results.md).

## H200 measured

| M x N | torch/Triton |
|---|--:|
| 4096x4096 | 5.26x |
| 8192x8192 | 6.40x |
| 8192x11008 | 6.65x |
| 8192x14336 | 6.54x |
| 16384x14336 | 6.67x |


## H200 benchmark replay (2026-07-21)

Original harness: `op_cos_sim.py`. The original harness completed on H200 with return code 0; its raw stdout/stderr is in the replay archive. Evidence: [`replay-2026-07-21-all-experimental.md`](../../data/crawl-runs/h200/replay-2026-07-21-all-experimental.md). All speed ratios are shape- and reference-specific.
