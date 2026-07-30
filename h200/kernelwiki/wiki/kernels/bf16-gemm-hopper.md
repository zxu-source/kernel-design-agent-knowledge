---
id: kernel-bf16-gemm-hopper
title: BF16 GEMM (Triton vs cuBLAS) on H200
type: kernel
architectures:
- sm90
tags:
- gemm
- kernel-fusion
confidence: experimental
reproducibility: benchmarked
kernel_types:
- gemm
languages:
- triton
- python
related:
- kernel-triton-fa2-hopper
- kernel-fp8-block-scale-gemm
sources:
- doc-nvidia-tuning-guide
performance_claims: []
blackwell_relevance: Same tiled bf16 GEMM on SM100; baseline characterization for
  the kernel-design-agent.
operator_purpose: speedup
what_it_does: BF16 GEMM (Triton tl.dot bf16->fp32 acc) characterized vs cuBLAS on
  H200; ~72% of bf16 peak.
h200_validation:
  date: 2026-07-21 (phase2; self-reported summary)
  evidence_file: data/crawl-runs/h200/op-bf16-gemm-h200-results.md
  harness_dir: artifacts/kernels/bf16-gemm-hopper/variants
  gpu: H200
  arch: sm90
  toolchain: Triton 3.6.0, PyTorch 2.11.0+cu130 (cuBLAS, TF32 off)
  correctness: PASS — vs fp32 reference, bf16 accumulation noise 0.5-1.0 (expected
    at large K).
  result: Triton 674-715 TFLOPS (~72% of H200 bf16 peak ~989 TF); 0.89-0.99x cuBLAS
    at large shapes (launch-bound at 1024^3).
  scope: Square/large bf16 GEMM, M=N=K up to 8192, on H200.
evidence_basis: H200 benchmark replay on 2026-07-21. The original harness completed
  on H200 with return code 0; its raw stdout/stderr is in the replay archive. This
  remains a local derived implementation, not an upstream-source performance claim.
---
> **Evidence boundary.** This is a locally derived canonical implementation, not a captured upstream source file. Any H200 latency/correctness text below comes from a self-reported run summary; retain it for hypothesis generation, but replay it with raw benchmark output before treating it as benchmark evidence.


## Overview

Baseline H200 BF16 GEMM via Triton (`tl.dot` bf16 inputs -> fp32 accumulator),
characterized against cuBLAS (`torch.matmul`, TF32 disabled for true bf16).
This establishes the achievable bf16 throughput on H200 for the kernel-design-agent.

```python
@triton.jit
def matmul_bf16(a_ptr,b_ptr,c_ptr,M,N,K,...,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    pm=tl.program_id(0); pn=tl.program_id(1)
    om=pm*BM+tl.arange(0,BM); on=pn*BN+tl.arange(0,BN)
    acc=tl.zeros((BM,BN),dtype=tl.float32)
    for k0 in range(0,K,BK):
        ok=k0+tl.arange(0,BK)
        a=tl.load(...); b=tl.load(...)
        acc+=tl.dot(a,b)               # bf16 x bf16 -> fp32
    tl.store(c_ptr+..., acc, mask=...)
```
Tile config: BM=128, BN=256, BK=64, num_warps=8, num_stages=3.

## Purpose: SPEEDUP baseline / characterization
Triton's tiled bf16 GEMM reaches ~72% of H200 bf16 peak and is within ~10% of
cuBLAS at large shapes — a solid baseline for sizing fused/quantized variants.
[`data/crawl-runs/h200/op-bf16-gemm-h200-results.md`](../../data/crawl-runs/h200/op-bf16-gemm-h200-results.md).

## H200 measured

| M=N=K | Triton TF | Triton util | cuBLAS TF | cuBLAS/Triton |
|--:|--:|--:|--:|--:|
| 1024 | 74  | 7%  | 144 | 0.51x |
| 2048 | 427 | 43% | 522 | 0.82x |
| 4096 | 674 | 68% | 758 | 0.89x |
| 8192 | 715 | 72% | 725 | 0.99x |

(H200 bf16 peak ~989 TFLOPS.)


## Robustness: fp32 accumulator is mandatory (validated)

A focused test compares the bf16 GEMM's fp32 accumulator against a bf16
accumulator. **fp32** stays at rel err ~1e-5 across K; **bf16** accumulator
degrades from 2.6% (K=2048) to **12.9%** (K=32768) — **~3000-7000x worse**, and
it overflows to inf for large-magnitude inputs. **Always accumulate bf16/fp16
GEMM in fp32** (the kernel above uses `acc = tl.zeros(..., dtype=tl.float32)`).
Details: [`data/crawl-runs/h200/op-mixed-prec-gemm-accum-h200-results.md`](../../data/crawl-runs/h200/op-mixed-prec-gemm-accum-h200-results.md).


## H200 benchmark replay (2026-07-21)

Original harness: `op_bf16_gemm.py`. The original harness completed on H200 with return code 0; its raw stdout/stderr is in the replay archive. Evidence: [`replay-2026-07-21-all-experimental.md`](../../data/crawl-runs/h200/replay-2026-07-21-all-experimental.md). All speed ratios are shape- and reference-specific.
