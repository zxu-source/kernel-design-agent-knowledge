---
id: kernel-block-sparse-matmul
title: Block-Sparse Matmul (spMM, Hopper / H200)
type: kernel
architectures:
- sm90
tags:
- gemm
- sparse-attention
confidence: experimental
reproducibility: benchmarked
kernel_types:
- gemm
- sparse-attention
languages:
- triton
- python
related:
- kernel-bf16-gemm-hopper
- kernel-grouped-gemm-hopper
sources:
- doc-nvidia-tuning-guide
performance_claims: []
blackwell_relevance: Same block-sparse pattern on SM100; sparse attention / MoE expert
  routing.
operator_purpose: speedup
what_it_does: 'Block-sparse matmul: grid over nonzero output tiles only (skips zero
  tiles).'
h200_validation:
  date: 2026-07-21 (phase2; self-reported summary)
  evidence_file: data/crawl-runs/h200/op-block-sparse-matmul-h200-results.md
  harness_dir: artifacts/kernels/block-sparse-matmul/variants
  gpu: H200
  arch: sm90
  toolchain: Triton 3.6.0, PyTorch 2.11.0+cu130
  correctness: PASS — nonzero tiles match dense-masked exactly (err=0.0).
  result: record-negative — naive random-tile block-sparse is ~0.01x dense (L2 locality
    destroyed). Sparse only wins with locality-aware tile sorting.
  scope: bf16, 128x128 tiles, density 10-25%, on H200.
evidence_basis: H200 benchmark replay on 2026-07-21. The original harness completed
  on H200 with return code 0; its raw stdout/stderr is in the replay archive. This
  remains a local derived implementation, not an upstream-source performance claim.
---
> **Evidence boundary.** This is a locally derived canonical implementation, not a captured upstream source file. Any H200 latency/correctness text below comes from a self-reported run summary; retain it for hypothesis generation, but replay it with raw benchmark output before treating it as benchmark evidence.


## Overview

Block-sparse matmul (spMM): given a block-sparsity mask, launch one program per
NONZERO output tile (skipping zero tiles). Theoretical speedup ~ total_tiles /
nonzero_tiles.

```python
@triton.jit
def bs_gemm(a_ptr,b_ptr,o_ptr,mb_ptr,nb_ptr, ..., BM, BN, BK):
    pid=tl.program_id(0)
    mb=tl.load(mb_ptr+pid); nb=tl.load(nb_ptr+pid)     # nonzero tile coords (indirect)
    om=mb*BM+tl.arange(0,BM); on=nb*BN+tl.arange(0,BN)
    acc=tl.zeros((BM,BN),dtype=tl.float32)
    for kk in range(0,K,BK):
        ... load A[om,:], B[:,on] ...; acc+=tl.dot(a,b)
    tl.store(o_ptr+..., acc, ...)
```

## Purpose: SPEEDUP (record-negative for the naive version)
Correctness PASS, but the **naive** version (1D grid over the mask's nonzero
tiles in arbitrary order) measured **~0.01x dense** (100x slower). The random
tile access order destroys L2 locality — consecutive programs hit unrelated A
rows and B columns, thrashing L2, so the kernel becomes memory-latency-bound at
tiny effective bandwidth. **Lesson**: block-sparse only beats dense when the tile
order is **locality-aware** (sort tiles to reuse A rows / B columns across
consecutive SMs), as in cuSPARSELt, FlashInfer block-sparse, or Triton's
`BlockSparse`. The naive "skip zero tiles" approach is a net loss on H200.
[`data/crawl-runs/h200/op-block-sparse-matmul-h200-results.md`](../../data/crawl-runs/h200/op-block-sparse-matmul-h200-results.md).

## H200 measured

| MxNxK (density, tiles) | dense/sparse (naive) |
|---|--:|
| 4096³ (25%, 266/1024) | 0.012x |
| 4096³ (10%, 96/1024) | 0.012x |
| 2048³ (25%, 71/256) | 0.009x |
| 8192x8192x4096 (25%, 1060/4096) | 0.012x |


## H200 benchmark replay (2026-07-21)

Original harness: `op_block_sparse_mm.py`. The original harness completed on H200 with return code 0; its raw stdout/stderr is in the replay archive. Evidence: [`replay-2026-07-21-all-experimental.md`](../../data/crawl-runs/h200/replay-2026-07-21-all-experimental.md). All speed ratios are shape- and reference-specific.
