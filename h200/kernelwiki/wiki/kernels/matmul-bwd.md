---
id: kernel-matmul-bwd
title: Matmul Backward (grad_a, grad_b, Hopper / H200)
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
- kernel-addmm-fused
sources:
- doc-nvidia-tuning-guide
performance_claims: []
blackwell_relevance: Same matmul backward on SM100; training gradient of Linear.
operator_purpose: speedup
what_it_does: 'Matmul backward: grad_a=grad_c@B^T, grad_b=A^T@grad_c (2 strided GEMMs,
  one generic kernel).'
h200_validation:
  date: 2026-07-21 (phase2; self-reported summary)
  evidence_file: data/crawl-runs/h200/op-matmul-bwd-h200-results.md
  harness_dir: artifacts/kernels/matmul-bwd/variants
  gpu: H200
  arch: sm90
  toolchain: Triton 3.6.0, PyTorch 2.11.0+cu130
  correctness: OK — bf16 accumulation noise (err 0.25-0.50 vs torch autograd; reduction-order
    difference).
  result: ~0.83x-1.05x cuBLAS (parity; 2 strided GEMMs match the bf16 GEMM baseline).
  scope: bf16, square/large GEMM backward, on H200.
evidence_basis: H200 benchmark replay on 2026-07-21. The original harness completed
  on H200 with return code 0; its raw stdout/stderr is in the replay archive. This
  remains a local derived implementation, not an upstream-source performance claim.
---
> **Evidence boundary.** This is a locally derived canonical implementation, not a captured upstream source file. Any H200 latency/correctness text below comes from a self-reported run summary; retain it for hypothesis generation, but replay it with raw benchmark output before treating it as benchmark evidence.


## Overview

Matmul backward (Linear training gradient): for `C = A@B` (`A[M,K]`, `B[K,N]`),
`grad_a[M,K] = grad_c @ B^T` and `grad_b[K,N] = A^T @ grad_c` — two GEMMs with
one transposed operand each. Implemented with one **generic strided GEMM** kernel
(transpose handled by strides, no materialization).

```python
@triton.jit
def gemm_g(a_ptr,b_ptr,c_ptr,OI,OJ,R,a_si,a_sr,b_sr,b_sj,BM,BN,BK):
    pi=tl.program_id(0); pj=tl.program_id(1)
    oi=pi*BM+tl.arange(0,BM); oj=pj*BN+tl.arange(0,BN)
    acc=tl.zeros((BM,BN),dtype=tl.float32)
    for r0 in range(0,R,BK):
        r=r0+tl.arange(0,BK)
        a=tl.load(a_ptr+oi[:,None]*a_si+r[None,:]*a_sr, mask=...)   # [BM,BR]
        b=tl.load(b_ptr+r[:,None]*b_sr+oj[None,:]*b_sj, mask=...)   # [BR,BN]
        acc+=tl.dot(a,b)
    tl.store(c_ptr+oi[:,None]*OJ+oj[None,:], acc, mask=...)
# grad_a = gemm_g(grad_c, B,  M,K,N, a_si=N,a_sr=1, b_sr=1,b_sj=N)  # B[r=n,out=k]=B[k,n]
# grad_b = gemm_g(A, grad_c, K,N,M, a_si=1,a_sr=K, b_sr=N,b_sj=1)   # A[out=k,r=m]=A[m,k]
```

## Purpose: SPEEDUP (characterization)
Correctness OK (bf16 accumulation noise). **~0.83x-1.05x cuBLAS** (parity): two
strided GEMMs match the bf16 GEMM baseline; cuBLAS's backward GEMMs are slightly
faster. Use cuBLAS autograd for production; this is a validated self-contained impl.
**Measurement note**: the torch baseline must be timed **backward-only** (build the
forward graph once, time `.backward`); timing forward+backward inflates the ratio.
[`data/crawl-runs/h200/op-matmul-bwd-h200-results.md`](../../data/crawl-runs/h200/op-matmul-bwd-h200-results.md).

## H200 measured

| MxKxN | torch/Triton (backward-only) |
|---|--:|
| 2048³ | 1.05x |
| 4096³ | 0.83x |
| 8192x8192x4096 | 0.94x |
| 8192x4096x8192 | 0.94x |
| 4096x4096x8192 | 0.93x |


## H200 benchmark replay (2026-07-21)

Original harness: `op_matmul_bwd.py`. The original harness completed on H200 with return code 0; its raw stdout/stderr is in the replay archive. Evidence: [`replay-2026-07-21-all-experimental.md`](../../data/crawl-runs/h200/replay-2026-07-21-all-experimental.md). All speed ratios are shape- and reference-specific.
