---
id: kernel-causal-mask
title: Causal Mask Generation (Hopper / H200)
type: kernel
architectures:
- sm90
tags:
- attention
confidence: experimental
reproducibility: benchmarked
kernel_types:
- fused-kernel
languages:
- triton
- python
related:
- kernel-triton-fa2-hopper
sources:
- doc-nvidia-tuning-guide
performance_claims: []
blackwell_relevance: Same mask gen on SM100; attention causal mask.
operator_purpose: both
what_it_does: 'Causal mask: out[i,j]=0 if j<=i else -inf; direct write (no ones materialization).'
h200_validation:
  date: 2026-07-21 (phase2; self-reported summary)
  evidence_file: data/crawl-runs/h200/op-causal-mask-h200-results.md
  harness_dir: artifacts/kernels/causal-mask/variants
  gpu: H200
  arch: sm90
  toolchain: Triton 3.6.0, PyTorch 2.11.0+cu130
  correctness: PASS — 0/-inf positions match torch.triu(ones*-inf,1). (NaN in diff
    is from -inf-(-inf), not a bug.)
  result: 1.22x-5.13x faster than torch (direct write vs ones+mul+triu).
  scope: fp32, N up to 8192, on H200.
evidence_basis: H200 benchmark replay on 2026-07-21. The corrected causal-mask harness
  passed semantic correctness for all four shapes. This remains a local derived implementation,
  not an upstream-source performance claim.
---
> **Evidence boundary.** This is a locally derived canonical implementation, not a captured upstream source file. Any H200 latency/correctness text below comes from a self-reported run summary; retain it for hypothesis generation, but replay it with raw benchmark output before treating it as benchmark evidence.


## Overview

Causal mask generation: `out[i,j] = 0 if j<=i else -inf`. The Triton kernel
writes each element directly from the `where(offs <= row, 0, -inf)` comparison,
avoiding torch's materialize-ones→multiply-by-inf→triu chain.

```python
@triton.jit
def causal_mask(o_ptr, N, BLOCK_N):
    row=tl.program_id(0)
    offs=tl.arange(0, BLOCK_N); mask=offs<N
    vals=tl.where(offs <= row, 0.0, float('-inf'))
    tl.store(o_ptr + row*N + offs, vals, mask=mask)
```

## Purpose: BOTH
- **speedup**: direct write (no ones materialization + multiply + triu). 1.22x-5.13x.
- **robustness**: -inf (not a large negative) ensures clean softmax masking.

## H200 measured

| N | torch/Triton |
|--:|--:|
| 1024 | 1.22x |
| 2048 | 1.84x |
| 4096 | 3.75x |
| 8192 | 5.13x |


## H200 benchmark replay (2026-07-21)

Original harness: `op_causal_mask.py`. The corrected causal-mask harness passed semantic correctness for all four shapes. Evidence: [`replay-2026-07-21-causal-mask-corrected-raw.md`](../../data/crawl-runs/h200/replay-2026-07-21-causal-mask-corrected-raw.md). All speed ratios are shape- and reference-specific.
