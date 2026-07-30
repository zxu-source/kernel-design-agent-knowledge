---
id: kernel-moe-permute
title: MoE Permute / Unpermute (dispatch gather+scatter, Hopper / H200)
type: kernel
architectures:
- sm90
tags:
- moe
- moe-gating
confidence: experimental
reproducibility: benchmarked
kernel_types:
- moe
- fused-kernel
languages:
- triton
- python
related:
- kernel-moe-gating
- kernel-grouped-gemm-hopper
sources:
- doc-nvidia-tuning-guide
performance_claims: []
blackwell_relevance: Same dispatch permute on SM100; MoE expert dispatch + combine.
operator_purpose: speedup
what_it_does: MoE dispatch permute (gather by sort index -> expert-contiguous) + unpermute
  (scatter back).
h200_validation:
  date: 2026-07-21 (phase2; self-reported summary)
  evidence_file: data/crawl-runs/h200/op-moe-permute-h200-results.md
  harness_dir: artifacts/kernels/moe-permute/variants
  gpu: H200
  arch: sm90
  toolchain: Triton 3.6.0, PyTorch 2.11.0+cu130
  correctness: PASS — bit-identical gather (permute) + scatter (unpermute) vs torch
    indexing.
  result: record-negative — ~parity with torch gather (0.83x-1.06x; gather/scatter
    is memcpy-bound, torch indexing near-optimal).
  scope: fp32/bf16, M up to 16384, hidden 4096..14336, on H200.
evidence_basis: H200 benchmark replay on 2026-07-21. The original harness completed
  on H200 with return code 0; its raw stdout/stderr is in the replay archive. This
  remains a local derived implementation, not an upstream-source performance claim.
---
> **Evidence boundary.** This is a locally derived canonical implementation, not a captured upstream source file. Any H200 latency/correctness text below comes from a self-reported run summary; retain it for hypothesis generation, but replay it with raw benchmark output before treating it as benchmark evidence.


## Overview

MoE dispatch: permute tokens into expert-contiguous order (gather by a sort index
so each expert's tokens are contiguous for its grouped GEMM), then after the
expert GEMMs, unpermute back to the original token order (scatter).

```python
@triton.jit
def permute_gather(tok_ptr, ord_ptr, out_ptr, M, D, BLOCK_D):     # permuted[i] = tokens[order[i]]
    i=tl.program_id(0); src=tl.load(ord_ptr+i)
    d=tl.arange(0,BLOCK_D)
    v=tl.load(tok_ptr+src*D+d, mask=d<D)
    tl.store(out_ptr+i*D+d, v, mask=d<D)
@triton.jit
def unpermute_scatter(perm_ptr, ord_ptr, out_ptr, M, D, BLOCK_D):  # out[order[i]] = permuted[i]
    i=tl.program_id(0); dst=tl.load(ord_ptr+i)
    d=tl.arange(0,BLOCK_D)
    v=tl.load(perm_ptr+i*D+d, mask=d<D)
    tl.store(out_ptr+dst*D+d, v, mask=d<D)
```

## Purpose: SPEEDUP (record-negative)
Gather/scatter is **memcpy-bound** and torch fancy indexing is near-optimal, so
the Triton kernel is **~parity** (0.83x-1.06x). Use torch indexing for standalone
permute/unpermute; the value here is the validated MoE dispatch building block
(often fused with the expert GEMM or gating in production).
[`data/crawl-runs/h200/op-moe-permute-h200-results.md`](../../data/crawl-runs/h200/op-moe-permute-h200-results.md).

## H200 measured (gather/permute)

| M x D | fp32 torch/Triton | bf16 torch/Triton |
|---|--:|--:|
| 4096x4096 | 0.90x | 0.83x |
| 8192x8192 | 1.01x | 0.97x |
| 8192x11008 | 1.04x | 1.03x |
| 8192x14336 | 1.03x | 1.02x |
| 16384x14336 | 1.06x | 1.05x |


## H200 benchmark replay (2026-07-21)

Original harness: `op_moe_permute.py`. The original harness completed on H200 with return code 0; its raw stdout/stderr is in the replay archive. Evidence: [`replay-2026-07-21-all-experimental.md`](../../data/crawl-runs/h200/replay-2026-07-21-all-experimental.md). All speed ratios are shape- and reference-specific.
