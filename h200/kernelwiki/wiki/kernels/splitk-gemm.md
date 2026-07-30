---
id: kernel-splitk-gemm
title: Split-K GEMM (Hopper / H200)
type: kernel
architectures:
- sm90
tags:
- gemm
- tile-scheduling
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
blackwell_relevance: Split-K is architecture-agnostic; same pattern on SM100. Use
  when M*N tile grid under-fills the GPU.
operator_purpose: speedup
what_it_does: 'Split-K GEMM: split K across blocks, each atomicAdds a partial to fp32
  output; more parallelism for tall-K/small-MN.'
h200_validation:
  date: 2026-07-21 (phase2; self-reported summary)
  evidence_file: data/crawl-runs/h200/op-splitk-gemm-h200-results.md
  harness_dir: artifacts/kernels/splitk-gemm/variants
  gpu: H200
  arch: sm90
  toolchain: Triton 3.6.0, PyTorch 2.11.0+cu130
  correctness: PASS — err=0.0 vs fp32 reference (both split-K and single).
  result: 1.26x-2.20x faster for tall-K/small-MN (best 128x128 K=16384 SPLIT=16 ->
    2.20x); 0.77x at 512x512 (atomic overhead when the GEMM already fills the GPU).
  scope: bf16 GEMM, tall-K shapes, SPLIT 4..16, on H200.
evidence_basis: H200 benchmark replay on 2026-07-21. The original harness completed
  on H200 with return code 0; its raw stdout/stderr is in the replay archive. This
  remains a local derived implementation, not an upstream-source performance claim.
---
> **Evidence boundary.** This is a locally derived canonical implementation, not a captured upstream source file. Any H200 latency/correctness text below comes from a self-reported run summary; retain it for hypothesis generation, but replay it with raw benchmark output before treating it as benchmark evidence.


## Overview

Split-K GEMM: split the K reduction across `SPLIT` blocks (3D grid
`(M_tiles, N_tiles, SPLIT)`); each block computes a partial `[BM,BN]` for its
K-chunk and `atomicAdd`s it to an fp32 output. This adds a grid dimension so a
tall-K / small-MN GEMM (whose M*N tile grid under-fills the GPU) can use more SMs.

```python
@triton.jit
def splitk_gemm(...,BM,BN,BK,SPLIT):
    pm=program_id(0); pn=program_id(1); ps=program_id(2)
    Kc=tl.cdiv(K,SPLIT); k0=ps*Kc
    acc=tl.zeros((BM,BN),dtype=tl.float32)
    for kk in range(k0,k0+Kc,BK):
        ... a,b loads (mask ok<K) ...
        acc+=tl.dot(a,b)
    tl.atomic_add(c_ptr+..., acc, mask=...)
```

## Purpose: SPEEDUP (occupancy for tall-K)
**1.26x-2.20x faster** than single-GEMM for tall-K/small-MN — best at
128x128 x K=16384, SPLIT=16 (2.20x). **0.77x at 512x512**: when the M*N grid
already fills the GPU, the atomic-add overhead makes split-K slower. Crossover:
use split-K when `M_tiles * N_tiles << num_SMs`.
[`data/crawl-runs/h200/op-splitk-gemm-h200-results.md`](../../data/crawl-runs/h200/op-splitk-gemm-h200-results.md).

## H200 measured

| MxNxK (SPLIT) | single/split-K |
|---|--:|
| 128x128x8192 (8) | 1.29x |
| 256x256x8192 (8) | 1.26x |
| 128x128x16384 (16) | 2.20x |
| 256x128x8192 (8) | 1.28x |
| 512x512x4096 (4) | 0.77x |


## H200 benchmark replay (2026-07-21)

Original harness: `op_splitk_gemm.py`. The original harness completed on H200 with return code 0; its raw stdout/stderr is in the replay archive. Evidence: [`replay-2026-07-21-all-experimental.md`](../../data/crawl-runs/h200/replay-2026-07-21-all-experimental.md). All speed ratios are shape- and reference-specific.
