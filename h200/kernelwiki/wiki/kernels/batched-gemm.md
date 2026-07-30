---
id: kernel-batched-gemm
title: Batched Matmul (bmm, Hopper / H200)
type: kernel
architectures:
- sm90
tags:
- gemm
- batched-gemv
confidence: experimental
reproducibility: benchmarked
kernel_types:
- gemm
languages:
- triton
- python
related:
- kernel-bf16-gemm-hopper
- kernel-grouped-gemm-hopper
sources:
- doc-nvidia-tuning-guide
performance_claims: []
blackwell_relevance: Same batched matmul on SM100; attention batched projections.
operator_purpose: speedup
what_it_does: 'Batched matmul (bmm): [B,M,K]@[B,K,N]->[B,M,N], grid (B,M-tiles,N-tiles).'
h200_validation:
  date: 2026-07-21 (phase2; self-reported summary)
  evidence_file: data/crawl-runs/h200/op-batched-gemm-h200-results.md
  harness_dir: artifacts/kernels/batched-gemm/variants
  gpu: H200
  arch: sm90
  toolchain: Triton 3.6.0, PyTorch 2.11.0+cu130
  correctness: PASS — err=0.0 vs fp32 reference.
  result: record-negative — 0.49x-0.69x cuBLAS torch.bmm (naive Triton bmm vs tuned
    strided-batched cuBLAS).
  scope: bf16, batch 8..32, on H200.
evidence_basis: H200 benchmark replay on 2026-07-21. The original harness completed
  on H200 with return code 0; its raw stdout/stderr is in the replay archive. This
  remains a local derived implementation, not an upstream-source performance claim.
---
> **Evidence boundary.** This is a locally derived canonical implementation, not a captured upstream source file. Any H200 latency/correctness text below comes from a self-reported run summary; retain it for hypothesis generation, but replay it with raw benchmark output before treating it as benchmark evidence.


## Overview

Batched matmul: `[B,M,K] @ [B,K,N] -> [B,M,N]`. Grid `(B, M-tiles, N-tiles)`;
each program handles one batch's tile (strided by batch).

```python
@triton.jit
def bmm(...):
    pbb=tl.program_id(0); pm=tl.program_id(1); pn=tl.program_id(2)
    om=pm*BM+tl.arange(0,BM); on=pn*BN+tl.arange(0,BN)
    acc=tl.zeros((BM,BN),dtype=tl.float32)
    a_base=a_ptr+pbb*sab; b_base=b_ptr+pbb*sbb
    for k0 in range(0,K,BK):
        ... a=load(a_base+...); b=load(b_base+...); acc+=tl.dot(a,b)
    tl.store(o_ptr+pbb*sob + ..., acc, mask=...)
```

## Purpose: SPEEDUP (record-negative)
Correctness PASS, but **0.49x-0.69x cuBLAS** — `torch.bmm` uses cuBLAS strided-
batched GEMM (highly tuned), and the naive Triton bmm attains only 144-538 TF vs
cuBLAS's 234-783 TF. Use torch.bmm for production; this is a validated self-contained impl.
[`data/crawl-runs/h200/op-batched-gemm-h200-results.md`](../../data/crawl-runs/h200/op-batched-gemm-h200-results.md).

## H200 measured

| BxMxNxK | Triton TF | cuBLAS TF | cuBLAS/Triton |
|---|--:|--:|--:|
| 16x512³ | 144 | 234 | 0.61x |
| 32x1024³ | 310 | 635 | 0.49x |
| 16x2048³ | 393 | 767 | 0.51x |
| 8x2048x2048x4096 | 538 | 783 | 0.69x |
| 32x1024x1024x2048 | 354 | 714 | 0.50x |


## H200 benchmark replay (2026-07-21)

Original harness: `op_bmm.py`. The original harness completed on H200 with return code 0; its raw stdout/stderr is in the replay archive. Evidence: [`replay-2026-07-21-all-experimental.md`](../../data/crawl-runs/h200/replay-2026-07-21-all-experimental.md). All speed ratios are shape- and reference-specific.
