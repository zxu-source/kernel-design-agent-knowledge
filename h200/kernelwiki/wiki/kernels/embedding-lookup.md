---
id: kernel-embedding-lookup
title: Embedding Lookup (gather, Hopper / H200)
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
- kernel-concat-split
- kernel-moe-permute
sources:
- doc-nvidia-tuning-guide
performance_claims: []
blackwell_relevance: Same gather on SM100; token-embedding lookup.
operator_purpose: speedup
what_it_does: 'Embedding lookup: out[i,:]=table[idx[i],:] (gather, one program/row).'
h200_validation:
  date: 2026-07-21 (phase2; self-reported summary)
  evidence_file: data/crawl-runs/h200/op-embedding-lookup-h200-results.md
  harness_dir: artifacts/kernels/embedding-lookup/variants
  gpu: H200
  arch: sm90
  toolchain: Triton 3.6.0, PyTorch 2.11.0+cu130
  correctness: PASS — bit-identical to torch (err=0).
  result: record-negative — ~parity (0.82x-1.02x; gather is memcpy-bound, torch indexing
    near-optimal).
  scope: fp32/bf16, vocab up to 128K, hidden 4096..8192, on H200.
evidence_basis: H200 benchmark replay on 2026-07-21. The original harness completed
  on H200 with return code 0; its raw stdout/stderr is in the replay archive. This
  remains a local derived implementation, not an upstream-source performance claim.
---
> **Evidence boundary.** This is a locally derived canonical implementation, not a captured upstream source file. Any H200 latency/correctness text below comes from a self-reported run summary; retain it for hypothesis generation, but replay it with raw benchmark output before treating it as benchmark evidence.


## Overview

Token-embedding lookup: `out[i,:] = table[idx[i],:]` (gather by token id). One
program per row (token). Memory-bound.

```python
@triton.jit
def embed_lookup(tbl_ptr, idx_ptr, o_ptr, V, D, BLOCK_D):
    i=tl.program_id(0); src=tl.load(idx_ptr+i)
    d=tl.arange(0, BLOCK_D); mask=d<D
    v=tl.load(tbl_ptr + src*D + d, mask=mask, other=0.0)
    tl.store(o_ptr + i*D + d, v, mask=mask)
```

## Purpose: SPEEDUP (record-negative)
Gather is memcpy-bound and torch indexing is near-optimal, so the Triton gather is
**~parity** (0.82x-1.02x). Use torch for standalone embedding lookup.
[`data/crawl-runs/h200/op-embedding-lookup-h200-results.md`](../../data/crawl-runs/h200/op-embedding-lookup-h200-results.md).

## H200 measured

| MxVxD | fp32 torch/Triton | bf16 torch/Triton |
|---|--:|--:|
| 8192x128256x4096 | 0.96x | 0.90x |
| 4096x128256x4096 | 0.88x | 0.82x |
| 8192x32000x4096 | 0.95x | 0.89x |
| 8192x128256x8192 | 1.00x | 0.95x |
| 16384x128256x4096 | 1.02x | 0.97x |


## H200 benchmark replay (2026-07-21)

Original harness: `op_embed.py`. The original harness completed on H200 with return code 0; its raw stdout/stderr is in the replay archive. Evidence: [`replay-2026-07-21-all-experimental.md`](../../data/crawl-runs/h200/replay-2026-07-21-all-experimental.md). All speed ratios are shape- and reference-specific.
