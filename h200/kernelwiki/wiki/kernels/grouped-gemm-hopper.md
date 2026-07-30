---
id: kernel-grouped-gemm-hopper
title: Grouped GEMM (MoE expert GEMM, Hopper / H200)
type: kernel
architectures:
- sm90
tags:
- gemm
- grouped-gemm
- kernel-fusion
- moe
confidence: experimental
reproducibility: benchmarked
kernel_types:
- grouped-gemm
- gemm
languages:
- triton
- python
related:
- kernel-bf16-gemm-hopper
- kernel-fused-gemm-bias-act
sources:
- doc-nvidia-tuning-guide
performance_claims: []
blackwell_relevance: Same grouped-GEMM pattern on SM100; MoE expert GEMM (DeepGEMM/CUTLASS).
operator_purpose: speedup
what_it_does: 'Grouped GEMM: GROUP expert A_g[Mg,K]@B_g[K,N] in one launch (2D tile
  grid + group lookup).'
h200_validation:
  date: 2026-07-21 (phase2; self-reported summary)
  evidence_file: data/crawl-runs/h200/op-grouped-gemm-h200-results.md
  harness_dir: artifacts/kernels/grouped-gemm-hopper/variants
  gpu: H200
  arch: sm90
  toolchain: Triton 3.6.0, PyTorch 2.11.0+cu130
  correctness: PASS — bit-identical to torch loop (err=0.0) across all shapes.
  result: 'Modest speedup 0.85x-1.39x vs torch loop (biggest at many small expert
    groups: 32x128 -> 1.39x launch amortization; ~parity when GEMM dominates).'
  scope: Uniform-Mg grouped bf16 GEMM, GROUP 8..32 experts, on H200. Naive version;
    production grouped-GEMM wins need token sorting + no-padding.
evidence_basis: H200 benchmark replay on 2026-07-21. The original harness completed
  on H200 with return code 0; its raw stdout/stderr is in the replay archive. This
  remains a local derived implementation, not an upstream-source performance claim.
---
> **Evidence boundary.** This is a locally derived canonical implementation, not a captured upstream source file. Any H200 latency/correctness text below comes from a self-reported run summary; retain it for hypothesis generation, but replay it with raw benchmark output before treating it as benchmark evidence.


## Overview

Grouped GEMM (MoE expert GEMM): `GROUP` independent GEMMs `A_g[Mg,K] @ B_g[K,N]`
in a single launch. A 2D tile grid covers `(total_M = GROUP*Mg, N)`; each tile
looks up its group from `pid_m` and loads the correct expert weight `B_g`. Avoids
per-expert launch overhead and can fill the GPU better when individual experts are
small (MoE dispatch with few tokens per expert).

```python
@triton.jit
def grouped_gemm(...,BM,BN,BK,TILES_PER_GROUP):
    pid_m=tl.program_id(0); pid_n=tl.program_id(1)
    g = pid_m // TILES_PER_GROUP            # expert index
    local = pid_m % TILES_PER_GROUP
    om = g*Mg + local*BM + tl.arange(0,BM)
    on = pid_n*BN + tl.arange(0,BN)
    acc=tl.zeros((BM,BN),dtype=tl.float32)
    b_base = b_ptr + g*sbgrp                # expert g's B[K,N]
    for k0 in range(0,K,BK):
        ...
        acc+=tl.dot(a, b)
    tl.store(o_ptr+..., acc, mask=...)
```

## Purpose: SPEEDUP
One launch vs a torch loop of `GROUP` cuBLAS calls. **Correctness bit-identical**
(err=0.0). Modest speedup **0.85x-1.39x**: the biggest win is at many small
expert groups (32 experts x 128 tokens -> 1.39x, where launch amortization
matters), and ~parity when the per-expert GEMM dominates (cuBLAS per-call overhead
is small relative to the work). Production grouped-GEMM wins (DeepGEMM/CUTLASS)
come from **token sorting + no-padding** dispatch, not this naive uniform-Mg version.
[`data/crawl-runs/h200/op-grouped-gemm-h200-results.md`](../../data/crawl-runs/h200/op-grouped-gemm-h200-results.md).

## H200 measured

| GROUP x Mg x N x K | loop/grouped |
|---|--:|
| 8 x 512 x 4096 x 4096 | 0.93x |
| 8 x 1024 x 4096 x 4096 | 0.85x |
| 16 x 256 x 4096 x 4096 | 1.09x |
| 32 x 128 x 4096 x 4096 | 1.39x |
| 8 x 2048 x 8192 x 4096 | 1.02x |


## H200 benchmark replay (2026-07-21)

Original harness: `op_grouped_gemm.py`. The original harness completed on H200 with return code 0; its raw stdout/stderr is in the replay archive. Evidence: [`replay-2026-07-21-all-experimental.md`](../../data/crawl-runs/h200/replay-2026-07-21-all-experimental.md). All speed ratios are shape- and reference-specific.
