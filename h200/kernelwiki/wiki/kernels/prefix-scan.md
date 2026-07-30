---
id: kernel-prefix-scan
title: Per-Row Prefix Sum / Scan (Hopper / H200)
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
- kernel-block-reduce
- kernel-online-softmax
sources:
- doc-nvidia-tuning-guide
performance_claims: []
blackwell_relevance: Same parallel scan on SM100; cumsum building block (sampling
  CDF, segment offsets).
operator_purpose: speedup
what_it_does: Per-row exclusive prefix sum (tl.cumsum, fp32 acc) vs torch.cumsum.
h200_validation:
  date: 2026-07-21 (phase2; self-reported summary)
  evidence_file: data/crawl-runs/h200/op-prefix-sum-h200-results.md
  harness_dir: artifacts/kernels/prefix-scan/variants
  gpu: H200
  arch: sm90
  toolchain: Triton 3.6.0, PyTorch 2.11.0+cu130
  correctness: PASS for fp32 (rel 1e-6 vs torch.cumsum). bf16 cumsum inherently lossy
    (running-sum rounding); use fp32 accumulation.
  result: 2.9x-4.0x faster than torch at moderate N (4096..8192); 0.31x-0.35x at N=32000
    (BLOCK=32768 register spill — tile the scan for huge N).
  scope: 1D per-row exclusive scan, N up to 32000, on H200.
evidence_basis: H200 benchmark replay on 2026-07-21. The original harness completed
  on H200 with return code 0; its raw stdout/stderr is in the replay archive. This
  remains a local derived implementation, not an upstream-source performance claim.
---
> **Evidence boundary.** This is a locally derived canonical implementation, not a captured upstream source file. Any H200 latency/correctness text below comes from a self-reported run summary; retain it for hypothesis generation, but replay it with raw benchmark output before treating it as benchmark evidence.


## Overview

Parallel prefix sum (exclusive cumsum) per row via `tl.cumsum`. A scan building
block (sampling CDF, segment-offset cumsum, attention cumsum).

```python
@triton.jit
def prefix_sum(x_ptr, o_ptr, N, BLOCK_N: tl.constexpr):
    row=tl.program_id(0)
    offs=tl.arange(0, BLOCK_N); mask=offs<N
    x=tl.load(x_ptr+row*N+offs, mask=mask, other=0.0).to(tl.float32)
    s=tl.cumsum(x, axis=0)            # inclusive scan
    s=s-x                             # exclusive (shift)
    tl.store(o_ptr+row*N+offs, s.to(o_ptr.dtype.element_ty), mask=mask)
```

## Purpose: SPEEDUP (characterization)
2.9x-4.0x faster than torch at moderate N. **0.31x-0.35x at N=32000**: the
one-program-per-row `BLOCK_N=32768` tile spills registers (the same large-N issue
as the naive softmax). Tile the scan (multi-pass with block carries) for huge N.
Correctness PASS for fp32 (rel 1e-6); bf16 cumsum is inherently lossy (running-sum
rounding dominates) — accumulate in fp32.
[`data/crawl-runs/h200/op-prefix-sum-h200-results.md`](../../data/crawl-runs/h200/op-prefix-sum-h200-results.md).

## H200 measured (fp32)

| M | N | torch/Triton |
|--:|--:|--:|
| 4096 | 4096 | 4.01x |
| 8192 | 8192 | 2.93x |
| 4096 | 32000 | 0.35x |
| 8192 | 32000 | 0.31x |


## H200 benchmark replay (2026-07-21)

Original harness: `op_prefix_sum.py`. The original harness completed on H200 with return code 0; its raw stdout/stderr is in the replay archive. Evidence: [`replay-2026-07-21-all-experimental.md`](../../data/crawl-runs/h200/replay-2026-07-21-all-experimental.md). All speed ratios are shape- and reference-specific.
