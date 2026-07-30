---
id: kernel-dropout
title: Fused Dropout (Hopper / H200)
type: kernel
architectures:
- sm90
tags:
- activation
confidence: experimental
reproducibility: benchmarked
kernel_types:
- fused-kernel
languages:
- triton
- python
related:
- kernel-gelu
- kernel-relu
sources:
- doc-nvidia-tuning-guide
performance_claims: []
blackwell_relevance: Same dropout on SM100; training regularization.
operator_purpose: speedup
what_it_does: 'Fused dropout: Philox RNG (tl.rand) + Bernoulli mask + scale 1/(1-p).'
h200_validation:
  date: 2026-07-21 (phase2; self-reported summary)
  evidence_file: data/crawl-runs/h200/op-dropout-h200-results.md
  harness_dir: artifacts/kernels/dropout/variants
  gpu: H200
  arch: sm90
  toolchain: Triton 3.6.0, PyTorch 2.11.0+cu130
  correctness: PASS (statistical) — zero_frac=0.1001 (=p=0.1); nonzero values exactly
    x/(1-p) (val_err=0).
  result: record-negative — 0.54x-0.85x torch (torch's native dropout with optimized
    Philox RNG is faster).
  scope: fp32/bf16, p=0.1, MLP shapes, on H200.
evidence_basis: H200 benchmark replay on 2026-07-21. The original harness completed
  on H200 with return code 0; its raw stdout/stderr is in the replay archive. This
  remains a local derived implementation, not an upstream-source performance claim.
---
> **Evidence boundary.** This is a locally derived canonical implementation, not a captured upstream source file. Any H200 latency/correctness text below comes from a self-reported run summary; retain it for hypothesis generation, but replay it with raw benchmark output before treating it as benchmark evidence.


## Overview

Fused dropout: `y = x * mask / (1-p)`, `mask ~ Bernoulli(1-p)`. The RNG
(`tl.rand` = Philox), the Bernoulli compare, and the scale are fused into one
elementwise kernel.

```python
@triton.jit
def dropout(x_ptr, o_ptr, p, seed, total, BLOCK):
    pid=tl.program_id(0); offs=pid*BLOCK+tl.arange(0,BLOCK); mask=offs<total
    x=tl.load(x_ptr+offs, mask=mask).to(tl.float32)
    rand=tl.rand(seed, offs)                       # Philox uniform [0,1)
    keep = rand >= p                               # Bernoulli(1-p)
    tl.store(o_ptr+offs, tl.where(keep, x/(1.0-p), 0.0).to(o_ptr.dtype.element_ty), mask=mask)
```

## Purpose: SPEEDUP (record-negative)
Correctness is **statistical** (torch uses a different RNG so exact values differ):
**zero_frac = 0.1001 = p** ✓ and nonzero values are **exactly x/(1-p)** ✓.
(`mean_ratio` is unstable for zero-mean randn input — E[x]≈0 → 0/0 — not a bug.)
**0.54x-0.85x torch**: torch's native dropout uses an optimized Philox path that
beats the Triton `tl.rand` elementwise kernel. Use torch dropout; this is a
validated self-contained impl.
[`data/crawl-runs/h200/op-dropout-h200-results.md`](../../data/crawl-runs/h200/op-dropout-h200-results.md).

## H200 measured

| MxN | fp32 torch/Triton | bf16 torch/Triton |
|---|--:|--:|
| 4096x4096 | 0.83x | 0.58x |
| 8192x8192 | 0.84x | 0.55x |
| 8192x14336 | 0.85x | 0.54x |
| 16384x14336 | 0.85x | 0.54x |


## H200 benchmark replay (2026-07-21)

Original harness: `op_dropout.py`. The original harness completed on H200 with return code 0; its raw stdout/stderr is in the replay archive. Evidence: [`replay-2026-07-21-all-experimental.md`](../../data/crawl-runs/h200/replay-2026-07-21-all-experimental.md). All speed ratios are shape- and reference-specific.
