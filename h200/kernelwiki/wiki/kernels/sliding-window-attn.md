---
id: kernel-sliding-window-attn
title: Sliding-Window Attention (Hopper / H200)
type: kernel
architectures:
- sm90
tags:
- attention
- flash-attention
confidence: experimental
reproducibility: benchmarked
kernel_types:
- attention
- fused-kernel
languages:
- triton
- python
related:
- kernel-triton-fa2-hopper
- kernel-gqa-mqa-attn
sources:
- doc-nvidia-tuning-guide
performance_claims: []
blackwell_relevance: Same SWA pattern on SM100; Mistral/Qwen long-context sliding
  window.
operator_purpose: both
what_it_does: 'Sliding-Window Attention: FA-2 with causal+window mask (query i attends
  to [i-W+1,i]); finite-neg masking avoids -inf-(-inf) NaN.'
h200_validation:
  date: 2026-07-21 (phase2; self-reported summary)
  evidence_file: data/crawl-runs/h200/op-swa-attn-h200-results.md
  harness_dir: artifacts/kernels/sliding-window-attn/variants
  gpu: H200
  arch: sm90
  toolchain: Triton 3.6.0, PyTorch 2.11.0+cu130
  correctness: PASS — err 2.4e-4 vs full-materialized reference.
  result: 6.9x-14.3x faster than the naive O(M^2) reference (flash-attention no-materialization
    win). Naive version loops full N (window = accuracy, not speed); window-capping
    deferred.
  scope: fp16, windows 512/1024, seq up to 16384, on H200.
evidence_basis: H200 benchmark replay on 2026-07-21. The original harness completed
  on H200 with return code 0; its raw stdout/stderr is in the replay archive. This
  remains a local derived implementation, not an upstream-source performance claim.
---
> **Evidence boundary.** This is a locally derived canonical implementation, not a captured upstream source file. Any H200 latency/correctness text below comes from a self-reported run summary; retain it for hypothesis generation, but replay it with raw benchmark output before treating it as benchmark evidence.


## Overview

Sliding-Window Attention: query `i` attends only to keys in `[max(0, i-W+1), i]`
(causal + window). Limits context to `W` tokens (Mistral/Qwen long-context). Built
on FA-2 with a windowed mask in the online softmax.

```python
for start_n in range(0, M, BLOCK_N):
    ...
    qk = tl.dot(q, k) * sm_scale
    n_offs = start_n + tl.arange(0, BLOCK_N)
    mask = (n_offs[None,:] <= offs_m[:,None]) & (n_offs[None,:] > offs_m[:,None] - WINDOW)
    qk = tl.where(mask, qk, -1e30)            # FINITE neg (not -inf): avoids -inf-(-inf)=NaN
    m_ij = tl.maximum(m_i, tl.max(qk,1)); p = tl.exp(qk - m_ij[:,None])
    ... online softmax update ...
```

## Purpose: BOTH
- **speedup**: fused, no O(M^2) materialization — 6.9x-14.3x faster than the
  naive full-materialization reference (the flash-attention memory win).
- **accuracy/context**: the window limits each query's receptive field to W
  tokens (bounded attention cost for long contexts).

**Robustness note**: masked positions use a large **finite** negative (`-1e30`),
not `-inf` — `-inf - (-inf)` is NaN, which corrupts the online softmax when a
tile is fully masked for a row whose window does not reach tile 0. Finite-neg
masking avoids this. **Deferred**: this naive version loops the full N range, so
the window affects accuracy but not speed; window-capping (skipping
out-of-window tiles) needs dynamic loop bounds (advanced scheduling).
[`data/crawl-runs/h200/op-swa-attn-h200-results.md`](../../data/crawl-runs/h200/op-swa-attn-h200-results.md).

## H200 measured

| BxHxMxD (W) | naive-ref/Triton |
|---|--:|
| 1x8x8192x64 (512) | 14.27x |
| 1x8x8192x128 (1024) | 7.70x |
| 4x8x4096x64 (512) | 13.04x |
| 1x4x16384x64 (1024) | 13.30x |
| 2x16x2048x128 (512) | 6.88x |


## H200 benchmark replay (2026-07-21)

Original harness: `op_swa_attn.py`. The original harness completed on H200 with return code 0; its raw stdout/stderr is in the replay archive. Evidence: [`replay-2026-07-21-all-experimental.md`](../../data/crawl-runs/h200/replay-2026-07-21-all-experimental.md). All speed ratios are shape- and reference-specific.
