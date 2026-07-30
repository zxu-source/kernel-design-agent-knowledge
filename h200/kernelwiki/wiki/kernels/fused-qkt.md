---
id: kernel-fused-qkt
title: Fused QK^T + Scale + Mask (Attention Score, Hopper / H200)
type: kernel
architectures:
- sm90
tags:
- gemm
- epilogue-fusion
- attention
confidence: experimental
reproducibility: benchmarked
kernel_types:
- gemm
- fused-kernel
languages:
- triton
- python
related:
- kernel-fused-gemm-bias-act
- kernel-triton-fa2-hopper
sources:
- doc-nvidia-tuning-guide
performance_claims: []
blackwell_relevance: Same attention-score fusion on SM100; pre-softmax QK^T GEMM epilogue.
operator_purpose: speedup
what_it_does: 'Fused QK^T*scale+mask: attention score GEMM with scale+causal-mask
  epilogue (vs torch 3-op).'
h200_validation:
  date: 2026-07-21 (phase2; self-reported summary)
  correctness: PASS — err_finite ~7e-8 (excludes -inf mask positions).
  result: 2.15x-5.49x faster than torch (Q@K^T then *scale then +mask, 3 ops fused).
evidence_basis: H200 benchmark replay on 2026-07-21. The original harness completed
  on H200 with return code 0; its raw stdout/stderr is in the replay archive. This
  remains a local derived implementation, not an upstream-source performance claim.
---
> **Evidence boundary.** This is a locally derived canonical implementation, not a captured upstream source file. Any H200 latency/correctness text below comes from a self-reported run summary; retain it for hypothesis generation, but replay it with raw benchmark output before treating it as benchmark evidence.

## H200 replay evidence (2026-07-21)

The compact replay passed correctness (`err_finite` 4.47e-8–6.71e-8) and measured
`torch_ms / triton_ms` of 1.95x, 2.00x, and 3.23x for 512², 1024², and 2048².
Raw stdout: [`replay-2026-07-21-three-operators-raw.md`](../../data/crawl-runs/h200/replay-2026-07-21-three-operators-raw.md).


## Overview
Attention score computation: `scores = (Q @ K^T) * scale + mask`. The GEMM
epilogue applies temperature scale and additive causal mask (0 or -inf) in the
same pass — vs torch's 3 separate ops (GEMM, scale, mask-add).

```python
@triton.jit
def qkt_scale_mask(q_ptr, k_ptr, m_ptr, o_ptr, M, N, D, scale, ..., BM, BN, BD):
    pm=tl.program_id(0); pn=tl.program_id(1)
    om=pm*BM+tl.arange(0,BM); on=pn*BN+tl.arange(0,BN)
    acc=tl.zeros((BM,BN),dtype=tl.float32)
    for d0 in range(0,D,BD):
        ... q,k loads ...; acc+=tl.dot(q, tl.trans(k))    # Q @ K^T
    acc=acc*scale                                           # epilogue: scale
    mask=tl.load(m_ptr+...,mask=...).to(tl.float32)
    tl.store(o_ptr+..., acc+mask, mask=...)                 # epilogue: +mask
```

## H200 measured: 2.15x-5.49x faster than torch (3-op fused into GEMM+epilogue).


## H200 benchmark replay (2026-07-21)

Original harness: `op_fused_qkt_sigmul.py`. The original harness completed on H200 with return code 0; its raw stdout/stderr is in the replay archive. Evidence: [`replay-2026-07-21-all-experimental.md`](../../data/crawl-runs/h200/replay-2026-07-21-all-experimental.md). All speed ratios are shape- and reference-specific.
