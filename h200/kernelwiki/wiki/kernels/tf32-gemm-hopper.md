---
id: kernel-tf32-gemm-hopper
title: TF32 GEMM (Triton vs cuBLAS) on H200
type: kernel
architectures:
- sm90
tags:
- gemm
confidence: experimental
reproducibility: benchmarked
kernel_types:
- gemm
languages:
- triton
- python
related:
- kernel-bf16-gemm-hopper
- kernel-fp8-block-scale-gemm
sources:
- doc-nvidia-tuning-guide
performance_claims: []
blackwell_relevance: Same TF32 path on SM100; naive Triton TF32 underperforms cuBLAS
  on both.
operator_purpose: speedup
what_it_does: TF32 GEMM (Triton fp32 tl.dot) characterized vs cuBLAS; naive kernel
  ~8% peak (record-negative).
h200_validation:
  date: 2026-07-21 (phase2; self-reported summary)
  evidence_file: data/crawl-runs/h200/op-tf32-gemm-h200-results.md
  harness_dir: artifacts/kernels/tf32-gemm-hopper/variants
  gpu: H200
  arch: sm90
  toolchain: Triton 3.6.0, PyTorch 2.11.0+cu130 (cuBLAS allow_tf32=True)
  correctness: PASS — TF32 rounding 0.04-0.12 vs cuBLAS TF32.
  result: record-negative — naive Triton TF32 ~82 TFLOPS (8% peak), 0.2-0.3x cuBLAS
    (391-417 TF, 42%). Use cuBLAS or a tuned Triton TF32 kernel.
  scope: fp32/TF32 matmul, M=N=K up to 8192, on H200. input_precision='tf32' explicit
    did not change the result.
evidence_basis: H200 benchmark replay on 2026-07-21. The original harness completed
  on H200 with return code 0; its raw stdout/stderr is in the replay archive. This
  remains a local derived implementation, not an upstream-source performance claim.
---
> **Evidence boundary.** This is a locally derived canonical implementation, not a captured upstream source file. Any H200 latency/correctness text below comes from a self-reported run summary; retain it for hypothesis generation, but replay it with raw benchmark output before treating it as benchmark evidence.


## Overview

TF32 GEMM (fp32 inputs, TF32 tensor cores — 10-bit mantissa). Compared Triton
`tl.dot` (fp32 inputs, `input_precision='tf32'`) against cuBLAS (`torch.matmul`,
`allow_tf32=True`).

```python
@triton.jit
def matmul_tf32(...,BM,BN,BK):
    ...
    acc += tl.dot(a, b, input_precision='tf32')   # fp32 inputs -> TF32 MMA
    ...
```
Tile config BM=128, BN=128, BK=32, num_warps=4, num_stages=3.

## Purpose: SPEEDUP baseline (record-negative)
The naive Triton TF32 kernel is **correct** (TF32 rounding) but achieves only
~8% of peak throughput — it does not hit the fast TF32 wgmma path with this
plain tile config. `input_precision='tf32'` did not change the result. Use
cuBLAS (or a tuned Triton TF32 kernel with TMA / proper K-tile) for real TF32
throughput. Contrast with the bf16 GEMM baseline where Triton hits ~72% of peak.
[`data/crawl-runs/h200/op-tf32-gemm-h200-results.md`](../../data/crawl-runs/h200/op-tf32-gemm-h200-results.md).

## H200 measured

| M=N=K | Triton TF | Triton util | cuBLAS TF | cuBLAS/Triton |
|--:|--:|--:|--:|--:|
| 1024 | 28 | 3%  | 87  | 0.32x |
| 2048 | 75 | 8%  | 291 | 0.26x |
| 4096 | 81 | 8%  | 392 | 0.21x |
| 8192 | 82 | 8%  | 416 | 0.20x |

## Related
- [bf16-gemm-hopper](bf16-gemm-hopper.md) — bf16 hits ~72% peak (the good baseline).


## H200 benchmark replay (2026-07-21)

Original harness: `op_tf32_gemm.py`. The original harness completed on H200 with return code 0; its raw stdout/stderr is in the replay archive. Evidence: [`replay-2026-07-21-all-experimental.md`](../../data/crawl-runs/h200/replay-2026-07-21-all-experimental.md). All speed ratios are shape- and reference-specific.
