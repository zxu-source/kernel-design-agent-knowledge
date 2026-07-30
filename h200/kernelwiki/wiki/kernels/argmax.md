---
id: kernel-argmax
title: Per-Row Argmax (Sampling top-1, Hopper / H200)
type: kernel
architectures:
- sm90
tags:
- reduction
- topk
confidence: experimental
reproducibility: benchmarked
kernel_types:
- fused-kernel
languages:
- triton
- python
related:
- kernel-block-reduce
- kernel-online-softmax
sources:
- doc-nvidia-tuning-guide
performance_claims: []
blackwell_relevance: Same per-row argmax on SM100; sampling top-1 building block.
operator_purpose: speedup
what_it_does: 'Per-row argmax (sampling top-1): block tl.argmax per row.'
h200_validation:
  date: 2026-07-21 (phase2; self-reported summary)
  evidence_file: data/crawl-runs/h200/op-argmax-h200-results.md
  harness_dir: artifacts/kernels/argmax/variants
  gpu: H200
  arch: sm90
  toolchain: Triton 3.6.0, PyTorch 2.11.0+cu130
  correctness: PASS — 100% match vs torch.argmax (agree=1.0).
  result: 1.10x-1.76x faster than torch.argmax.
  scope: fp32, vocab up to 128K, on H200.
evidence_basis: H200 benchmark replay on 2026-07-21. The original harness completed
  on H200 with return code 0; its raw stdout/stderr is in the replay archive. This
  remains a local derived implementation, not an upstream-source performance claim.
---
> **Evidence boundary.** This is a locally derived canonical implementation, not a captured upstream source file. Any H200 latency/correctness text below comes from a self-reported run summary; retain it for hypothesis generation, but replay it with raw benchmark output before treating it as benchmark evidence.


## Overview

Per-row argmax (greedy-decoding / sampling top-1): for each row find the index of
the maximum value.

```python
@triton.jit
def argmax_row(x_ptr, idx_ptr, N, BLOCK_N: tl.constexpr):
    row=tl.program_id(0)
    offs=tl.arange(0, BLOCK_N); mask=offs<N
    x=tl.load(x_ptr+row*N+offs, mask=mask, other=-float('inf')).to(tl.float32)
    i=tl.argmax(x, axis=0)
    tl.store(idx_ptr+row, i.to(tl.int64))
```

## Purpose: SPEEDUP
1.10x-1.76x faster than `torch.argmax`. Handles large N (128K vocab) without the
register-spill regression seen in the naive one-program-per-row softmax
(argmax is a single scalar output per row, no exp/div).
[`data/crawl-runs/h200/op-argmax-h200-results.md`](../../data/crawl-runs/h200/op-argmax-h200-results.md).

## H200 measured

| M | N | torch/Triton |
|--:|--:|--:|
| 4096 | 4096 | 1.10x |
| 8192 | 8192 | 1.76x |
| 4096 | 32000 | 1.22x |
| 8192 | 32000 | 1.23x |
| 8192 | 128256 | 1.16x |


## H200 benchmark replay (2026-07-21)

Original harness: `op_argmax.py`. The original harness completed on H200 with return code 0; its raw stdout/stderr is in the replay archive. Evidence: [`replay-2026-07-21-all-experimental.md`](../../data/crawl-runs/h200/replay-2026-07-21-all-experimental.md). All speed ratios are shape- and reference-specific.
