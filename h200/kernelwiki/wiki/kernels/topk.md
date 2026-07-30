---
id: kernel-topk
title: Per-Row Top-K Selection (Hopper / H200)
type: kernel
architectures:
- sm90
tags:
- topk
- reduction
confidence: experimental
reproducibility: benchmarked
kernel_types:
- fused-kernel
languages:
- triton
- python
related:
- kernel-argmax
- kernel-online-softmax
sources:
- doc-nvidia-tuning-guide
performance_claims: []
blackwell_relevance: Same top-k pattern on SM100; top-k sampling/decoding building
  block.
operator_purpose: speedup
what_it_does: Per-row top-k (k-pass max+argmax+mask) vs torch.topk; values+indices
  exact.
h200_validation:
  date: 2026-07-21 (phase2; self-reported summary)
  evidence_file: data/crawl-runs/h200/op-topk-h200-results.md
  harness_dir: artifacts/kernels/topk/variants
  gpu: H200
  arch: sm90
  toolchain: Triton 3.6.0, PyTorch 2.11.0+cu130
  correctness: PASS — top-k values AND index-set match torch.topk exactly (val_err=0,
    set_match=1.0).
  result: 3.54x-4.14x faster than torch.topk at moderate N; 0.57x-0.58x at N=32000
    (k-pass O(kN) slower than torch partial sort for huge N).
  scope: fp32, K=5, vocab up to 32000, on H200.
evidence_basis: H200 benchmark replay on 2026-07-21. The original harness completed
  on H200 with return code 0; its raw stdout/stderr is in the replay archive. This
  remains a local derived implementation, not an upstream-source performance claim.
---
> **Evidence boundary.** This is a locally derived canonical implementation, not a captured upstream source file. Any H200 latency/correctness text below comes from a self-reported run summary; retain it for hypothesis generation, but replay it with raw benchmark output before treating it as benchmark evidence.


## Overview

Per-row top-k for top-k sampling/decoding: k passes of (row max + argmax +
mask-out the picked element). O(k*N) per row; efficient for small k.

```python
@triton.jit
def topk(x_ptr, val_ptr, idx_ptr, N, K: tl.constexpr, BLOCK_N: tl.constexpr):
    row=tl.program_id(0)
    offs=tl.arange(0, BLOCK_N)
    x=tl.load(x_ptr+row*N+offs, mask=offs<N, other=-1e30).to(tl.float32)
    for k in tl.static_range(K):
        m=tl.max(x, axis=0); sel = x==m
        idx=tl.argmax(tl.where(sel, offs.to(tl.float32), -1e30), axis=0)   # sel->offs else -inf
        tl.store(val_ptr+row*K+k, m); tl.store(idx_ptr+row*K+k, idx.to(tl.int64))
        x=tl.where(sel, -1e30, x)                                          # mask out picked
```

## Purpose: SPEEDUP
3.54x-4.14x faster than torch.topk at moderate N (few passes, fully parallel).
0.57x-0.58x at N=32000: the k-pass is O(k*N) and torch.topk uses an efficient
partial-sort for huge N. Use this for small-to-moderate N, small k.
[`data/crawl-runs/h200/op-topk-h200-results.md`](../../data/crawl-runs/h200/op-topk-h200-results.md).

## H200 measured (K=5, fp32)

| M | N | torch/Triton |
|--:|--:|--:|
| 4096 | 4096 | 4.14x |
| 8192 | 8192 | 3.54x |
| 4096 | 32000 | 0.58x |
| 8192 | 32000 | 0.57x |


## H200 benchmark replay (2026-07-21)

Original harness: `op_topk.py`. The original harness completed on H200 with return code 0; its raw stdout/stderr is in the replay archive. Evidence: [`replay-2026-07-21-all-experimental.md`](../../data/crawl-runs/h200/replay-2026-07-21-all-experimental.md). All speed ratios are shape- and reference-specific.
