---
id: kernel-fused-gated-mlp
title: Fused Gated MLP (gate-up projection, Hopper / H200)
type: kernel
architectures:
- sm90
tags:
- epilogue-fusion
- silu
- kernel-fusion
- gemm
confidence: experimental
reproducibility: benchmarked
kernel_types:
- gemm
- fused-kernel
languages:
- triton
- python
related:
- kernel-silu-and-mul
- kernel-fused-gemm-bias-act
sources:
- doc-nvidia-tuning-guide
performance_claims: []
blackwell_relevance: Same fused gate-up MLP on SM100; Llama/Qwen MLP.
operator_purpose: speedup
what_it_does: 'Fused gated MLP: concat gate|up -> 1 GEMM x@Wu[2N] -> split -> silu(gate)*up
  (vs 2 GEMMs).'
h200_validation:
  date: 2026-07-21 (phase2; self-reported summary)
  evidence_file: data/crawl-runs/h200/op-fused-gated-mlp-h200-results.md
  harness_dir: artifacts/kernels/fused-gated-mlp/variants
  gpu: H200
  arch: sm90
  toolchain: Triton 3.6.0, PyTorch 2.11.0+cu130
  correctness: OK — bf16 accumulation noise (err 1-2, silu*mul amplifies rounding)
    vs torch sequential.
  result: record-negative — 0.80x-0.90x torch sequential (2 cuBLAS GEMMs); cuBLAS
    per-GEMM efficiency outweighs the x-read-once fusion benefit.
  scope: bf16, LLM MLP shapes, on H200.
evidence_basis: H200 benchmark replay on 2026-07-21. The original harness completed
  on H200 with return code 0; its raw stdout/stderr is in the replay archive. This
  remains a local derived implementation, not an upstream-source performance claim.
---
> **Evidence boundary.** This is a locally derived canonical implementation, not a captured upstream source file. Any H200 latency/correctness text below comes from a self-reported run summary; retain it for hypothesis generation, but replay it with raw benchmark output before treating it as benchmark evidence.


## Overview

LLM fused gated MLP: concatenate `gate_W | up_W` into `Wu [K, 2N]`, do ONE GEMM
`x @ Wu -> [M, 2N]`, split, then `silu(gate) * up`. Reads `x` once vs twice (two
separate GEMMs). Standard Llama/Qwen MLP.

```python
@triton.jit
def gemm_2n(a_ptr, wu_ptr, o_ptr, M, K, N2, sam, BM, BN, BK):
    pm=tl.program_id(0); pn=tl.program_id(1)
    om=pm*BM+tl.arange(0,BM); on=pn*BN+tl.arange(0,BN)
    acc=tl.zeros((BM,BN),dtype=tl.float32)
    for k0 in range(0,K,BK):
        ok=k0+tl.arange(0,BK)
        a=tl.load(a_ptr+om[:,None]*sam+ok[None,:], mask=(om[:,None]<M)&(ok[None,:]<K), other=0.0)
        w=tl.load(wu_ptr+ok[:,None]*N2+on[None,:], mask=(ok[:,None]<K)&(on[None,:]<N2), other=0.0)
        acc += tl.dot(a, w)
    tl.store(o_ptr+om[:,None]*N2+on[None,:], acc.to(o_ptr.dtype.element_ty), mask=(om[:,None]<M)&(on[None,:]<N2))
# then: silu_and_mul(o2[:,:N], o2[:,N:], out) -> out[M,N]
```

## Purpose: SPEEDUP (record-negative)
The fusion reads `x` once (vs twice), but the naive Triton GEMM@2N is ~0.9x
cuBLAS per-FLOP, and torch's two cuBLAS GEMMs are each highly optimized — so
**0.80x-0.90x torch sequential** (cuBLAS per-GEMM efficiency outweighs the
x-read-once benefit). The fusion WOULD win if the per-GEMM efficiencies were
equal (use a cuBLAS-grade fused gate-up, or a single CUTLASS GEMM@2N, to realize
it). Correctness is bf16-accurate.
[`data/crawl-runs/h200/op-fused-gated-mlp-h200-results.md`](../../data/crawl-runs/h200/op-fused-gated-mlp-h200-results.md).

## H200 measured

| MxKxN | seq/fused |
|---|--:|
| 2048x4096x4096 | 0.80x |
| 4096x4096x4096 | 0.81x |
| 8192x4096x11008 | 0.87x |
| 8192x8192x14336 | 0.90x |
| 8192x4096x14336 | 0.81x |


## H200 benchmark replay (2026-07-21)

Original harness: `op_fused_gated_mlp.py`. The original harness completed on H200 with return code 0; its raw stdout/stderr is in the replay archive. Evidence: [`replay-2026-07-21-all-experimental.md`](../../data/crawl-runs/h200/replay-2026-07-21-all-experimental.md). All speed ratios are shape- and reference-specific.
