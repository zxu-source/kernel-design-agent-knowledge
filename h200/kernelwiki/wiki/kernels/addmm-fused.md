---
id: kernel-addmm-fused
title: Fused AddMM (alpha*A@B + beta*C, Hopper / H200)
type: kernel
architectures:
- sm90
tags:
- gemm
- epilogue-fusion
confidence: experimental
reproducibility: benchmarked
kernel_types:
- gemm
- fused-kernel
languages:
- triton
- python
related:
- kernel-bf16-gemm-hopper
- kernel-fused-gemm-bias-act
sources:
- doc-nvidia-tuning-guide
performance_claims: []
blackwell_relevance: Same epilogue-fused GEMM on SM100; Linear+residual (alpha/beta).
operator_purpose: speedup
what_it_does: 'Fused AddMM: out = alpha*A@B + beta*C (residual add in the GEMM epilogue).'
h200_validation:
  date: 2026-07-21 (phase2; self-reported summary)
  evidence_file: data/crawl-runs/h200/op-addmm-fused-h200-results.md
  harness_dir: artifacts/kernels/addmm-fused/variants
  gpu: H200
  arch: sm90
  toolchain: Triton 3.6.0, PyTorch 2.11.0+cu130
  correctness: PASS — bf16 accumulation noise vs torch.addmm.
  result: 0.86x-0.95x cuBLAS (torch.addmm is already a fused cuBLAS op; Triton ~0.9x
    like the bf16 GEMM baseline).
  scope: bf16, MLP shapes, on H200.
evidence_basis: H200 benchmark replay on 2026-07-21. The original harness completed
  on H200 with return code 0; its raw stdout/stderr is in the replay archive. This
  remains a local derived implementation, not an upstream-source performance claim.
---
> **Evidence boundary.** This is a locally derived canonical implementation, not a captured upstream source file. Any H200 latency/correctness text below comes from a self-reported run summary; retain it for hypothesis generation, but replay it with raw benchmark output before treating it as benchmark evidence.


## Overview

`out = alpha*A@B + beta*C`: a residual add fused into the GEMM epilogue (vs a
separate add). Standard Linear+residual pattern.

```python
@triton.jit
def addmm(a_ptr,b_ptr,c_ptr,o_ptr,alpha,beta,...,BM,BN,BK):
    ... acc=tl.zeros((BM,BN),dtype=tl.float32)
    for k0 in range(0,K,BK):
        ... a,b loads ...; acc+=tl.dot(a,b)
    c=tl.load(c_ptr+..., mask=mask).to(tl.float32)
    out=alpha*acc + beta*c                    # residual fused in epilogue
    tl.store(o_ptr+..., out.to(o_ptr.dtype.element_ty), mask=mask)
```

## Purpose: SPEEDUP (characterization)
Correctness PASS, but **0.86x-0.95x cuBLAS** — `torch.addmm` is already a single
fused cuBLAS op, so there's no separate-add to eliminate; Triton's fused addmm
tracks the bf16-GEMM baseline (~0.9x cuBLAS). The fusion is "free" (no extra pass
over the output), but cuBLAS's GEMM+epilogue is still faster than the naive
Triton tile. Use cuBLAS addmm for production; this is a validated self-contained impl.
[`data/crawl-runs/h200/op-addmm-fused-h200-results.md`](../../data/crawl-runs/h200/op-addmm-fused-h200-results.md).

## H200 measured

| MxNxK | Triton TF | cuBLAS TF | cuBLAS/Triton |
|---|--:|--:|--:|
| 2048³ | 385 | 449 | 0.86x |
| 4096³ | 631 | 689 | 0.92x |
| 8192x8192x4096 | 662 | 702 | 0.94x |
| 8192³ | 613 | 663 | 0.92x |
| 8192x11008x4096 | 598 | 632 | 0.95x |


## H200 benchmark replay (2026-07-21)

Original harness: `op_addmm.py`. The original harness completed on H200 with return code 0; its raw stdout/stderr is in the replay archive. Evidence: [`replay-2026-07-21-all-experimental.md`](../../data/crawl-runs/h200/replay-2026-07-21-all-experimental.md). All speed ratios are shape- and reference-specific.
