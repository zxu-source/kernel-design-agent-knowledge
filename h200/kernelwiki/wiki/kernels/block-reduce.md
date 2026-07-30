---
id: kernel-block-reduce
title: Block+Warp Sum Reduction (Hopper / H200)
type: kernel
architectures:
- sm90
tags:
- reduction
confidence: experimental
reproducibility: benchmarked
kernel_types:
- fused-kernel
languages:
- triton
- python
related:
- kernel-rmsnorm-hopper
- kernel-online-softmax
sources:
- doc-nvidia-tuning-guide
performance_claims: []
blackwell_relevance: Same reduction pattern on SM100; a building block for norms/softmax/sampling.
operator_purpose: speedup
what_it_does: Block+warp tree sum reduction (tl.sum per block -> partials -> 2nd-stage
  reduce).
h200_validation:
  date: 2026-07-21 (phase2; self-reported summary)
  evidence_file: data/crawl-runs/h200/op-reduce-sum-h200-results.md
  harness_dir: artifacts/kernels/block-reduce/variants
  gpu: H200
  arch: sm90
  toolchain: Triton 3.6.0, PyTorch 2.11.0+cu130
  correctness: PASS — rel error ~1e-7 vs torch.sum.
  result: record-negative — 0.62x-0.95x torch.sum (torch's CUB-style reduce is highly
    optimized; launch-bound at small N, ~parity at large N).
  scope: 1D fp32/bf16 reduction, N up to 64M, on H200.
evidence_basis: H200 benchmark replay on 2026-07-21. The original harness completed
  on H200 with return code 0; its raw stdout/stderr is in the replay archive. This
  remains a local derived implementation, not an upstream-source performance claim.
---
> **Evidence boundary.** This is a locally derived canonical implementation, not a captured upstream source file. Any H200 latency/correctness text below comes from a self-reported run summary; retain it for hypothesis generation, but replay it with raw benchmark output before treating it as benchmark evidence.


## Overview

A standalone sum-reduction building block (the inner reduction used by RMSNorm /
LayerNorm / softmax / sampling). Two-stage: each block tree-reduces a chunk
(`tl.sum` -> warp shuffle), stores a partial, then a tiny second-stage reduce
sums the partials.

```python
@triton.jit
def reduce_sum(x_ptr, partials_ptr, N, BLOCK: tl.constexpr):
    pid=tl.program_id(0)
    offs=pid*BLOCK+tl.arange(0,BLOCK); mask=offs<N
    x=tl.load(x_ptr+offs, mask=mask, other=0.0).to(tl.float32)
    s=tl.sum(x, axis=0)                       # tree reduce -> scalar
    tl.store(partials_ptr+pid, s)
# then: partials.sum()
```

## Purpose: SPEEDUP (characterization; record-negative)
Correctness is excellent (rel ~1e-7), but Triton's standalone two-stage reduce
is **0.62x-0.95x torch.sum** — torch's reduction is a heavily-tuned CUB-style
kernel. The value is the validated building-block pattern; for a standalone
full-array reduce, use torch/CUB. (Inside a fused kernel like RMSNorm, the
per-row `tl.sum` is fine because it is not a separate launch.)
[`data/crawl-runs/h200/op-reduce-sum-h200-results.md`](../../data/crawl-runs/h200/op-reduce-sum-h200-results.md).

## H200 measured

| N | dtype | torch/Triton |
|--:|---|--:|
| 1M    | fp32 | 0.64x |
| 4M    | fp32 | 0.75x |
| 16M   | fp32 | 0.91x |
| 64M   | fp32 | 0.95x |
| (bf16 tracks within 0.05x) | | |

## Related
- [rmsnorm-hopper](rmsnorm-hopper.md) / [online-softmax](online-softmax.md) — fused kernels that embed this reduction.


## H200 benchmark replay (2026-07-21)

Original harness: `op_reduce_sum.py`. The original harness completed on H200 with return code 0; its raw stdout/stderr is in the replay archive. Evidence: [`replay-2026-07-21-all-experimental.md`](../../data/crawl-runs/h200/replay-2026-07-21-all-experimental.md). All speed ratios are shape- and reference-specific.
