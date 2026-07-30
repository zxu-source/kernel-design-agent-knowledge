---
id: kernel-scatter-add
title: Scatter-Add / index_add (Hopper / H200)
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
- kernel-embedding-lookup
- kernel-moe-permute
sources:
- doc-nvidia-tuning-guide
performance_claims: []
blackwell_relevance: Same scatter-add on SM100; embedding-grad / MoE-combine.
operator_purpose: speedup
what_it_does: 'Scatter-add (index_add): out[idx[i],:] += values[i,:] (atomic-add,
  embedding-grad/MoE-combine).'
h200_validation:
  date: 2026-07-21 (phase2; self-reported summary)
  evidence_file: data/crawl-runs/h200/op-scatter-add-h200-results.md
  harness_dir: artifacts/kernels/scatter-add/variants
  gpu: H200
  arch: sm90
  toolchain: Triton 3.6.0, PyTorch 2.11.0+cu130
  correctness: PASS — rel ~6e-8 vs torch.index_add (atomic-add order).
  result: 1.10x-1.44x faster than torch.index_add.
  scope: fp32, vocab up to 128K, hidden 4096..8192, on H200.
evidence_basis: H200 benchmark replay on 2026-07-21. The original harness completed
  on H200 with return code 0; its raw stdout/stderr is in the replay archive. This
  remains a local derived implementation, not an upstream-source performance claim.
---
> **Evidence boundary.** This is a locally derived canonical implementation, not a captured upstream source file. Any H200 latency/correctness text below comes from a self-reported run summary; retain it for hypothesis generation, but replay it with raw benchmark output before treating it as benchmark evidence.


## Overview

Scatter-add (`index_add`): `out[idx[i], :] += values[i, :]` — accumulate token
rows into vocab/expert bins. Used for embedding gradient and MoE combine.

```python
@triton.jit
def scatter_add(val_ptr, idx_ptr, out_ptr, D, BLOCK_D):
    i=tl.program_id(0); dst=tl.load(idx_ptr+i)
    d=tl.arange(0, BLOCK_D); mask=d<D
    v=tl.load(val_ptr+i*D+d, mask=mask, other=0.0).to(tl.float32)
    tl.atomic_add(out_ptr + dst*D + d, v, mask=mask)
```

## Purpose: SPEEDUP
1.10x-1.44x faster than torch.index_add_ (the atomic-add scatter amortizes over
the D-vector). Correctness within atomic-order tolerance (rel ~6e-8).
[`data/crawl-runs/h200/op-scatter-add-h200-results.md`](../../data/crawl-runs/h200/op-scatter-add-h200-results.md).

## H200 measured

| MxVxD | torch/Triton |
|---|--:|
| 8192x128256x4096 | 1.19x |
| 4096x128256x4096 | 1.10x |
| 8192x32000x4096 | 1.44x |
| 16384x128256x4096 | 1.32x |
| 8192x128256x8192 | 1.11x |


## H200 benchmark replay (2026-07-21)

Original harness: `op_scatter_add.py`. The original harness completed on H200 with return code 0; its raw stdout/stderr is in the replay archive. Evidence: [`replay-2026-07-21-all-experimental.md`](../../data/crawl-runs/h200/replay-2026-07-21-all-experimental.md). All speed ratios are shape- and reference-specific.
