---
id: kernel-triton-fa2-hopper
title: Triton Flash-Attention-2 Forward (Hopper / H200)
type: kernel
architectures:
- sm90
tags:
- attention
- flash-attention
- pipeline-stages
- warp-specialization
- gemm
confidence: experimental
reproducibility: benchmarked
kernel_types:
- fused-kernel
- attention
languages:
- triton
- python
related:
- technique-pipeline-stages
- technique-warp-specialization
- kernel-flash-attention-4
sources:
- pr-triton-6660
blackwell_relevance: 'FA-2 is the immediate predecessor of FA-4 on Blackwell; the
  Triton compiler pipelining/WS that PR #6660 improves for attention on Hopper is
  the same machinery carried into SM100 schedules. The H200 characterization establishes
  the Hopper baseline against which Blackwell FA-4 gains are measured.

  '
performance_claims: []
artifact_dir: artifacts/kernels/triton-fa2-hopper/variants
h200_validation:
  date: 2026-07-20 (overnight; self-reported summary)
  evidence_file: data/crawl-runs/h200/triton6660-flash-attention-h200-results.md
  gpu: H200
  arch: sm90
  toolchain: Triton 3.6.0, PyTorch 2.11.0+cu130
  correctness: PASS — Triton FA-2 forward matches torch SDPA within fp16 noise (max
    err 7.6e-6..1.5e-5)
  result: 'The local plain-Triton harness reports 215-429 TFLOPS; torch SDPA reports
    ~1.5-2.1x higher throughput (448-653 TFLOPS). This comparison does not isolate
    the compiler changes in PR #6660.'
  scope: From-scratch FA-2 forward characterization on H200; compiler-internal pipeliner/WS
    not individually isolated (no user toggle in Triton 3.6.0).
evidence_basis: H200 benchmark replay on 2026-07-21. The non-causal FA2 harness passed
  correctness; torch SDPA was faster for every tested shape. This remains a local
  derived implementation, not an upstream-source performance claim.
---

> **Evidence boundary.** This page's implementation is locally derived, not a
> captured PR #6660 source file. The local FA-2-versus-SDPA comparison does not
> isolate the compiler changes in that PR.

## Overview

PR #6660 improves Triton's compiler Pipeliner and warp-specialization passes for
attention (assigned-stage mechanism composing WS with software pipelining,
producing CUTLASS-level FMHA schedules). Those passes are compiler-internal with
no user-facing toggle in Triton 3.6.0, so this page records a from-scratch Triton
**flash-attention-2 forward** kernel — the kernel class the PR optimizes — and
its measured behavior on H200.

## Kernel structure

Standard FA-2 forward with online (tiling) softmax, written with Triton block
pointers:

```python
@triton.jit
def attn_fwd(Q, K, V, O, sm_scale, stride_bh, stride_row,
             M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr):
    pid_m = tl.program_id(0); pid_bh = tl.program_id(1)
    Qb = tl.make_block_ptr(base=Q + pid_bh*stride_bh, shape=(M, HEAD_DIM),
                           strides=(stride_row, 1), offsets=(pid_m*BLOCK_M, 0),
                           block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0))
    Kb = tl.make_block_ptr(base=K + pid_bh*stride_bh, shape=(HEAD_DIM, N),
                           strides=(1, stride_row), offsets=(0, 0),
                           block_shape=(HEAD_DIM, BLOCK_N), order=(0, 1))
    Vb = tl.make_block_ptr(base=V + pid_bh*stride_bh, shape=(N, HEAD_DIM),
                           strides=(stride_row, 1), offsets=(0, 0),
                           block_shape=(BLOCK_N, HEAD_DIM), order=(1, 0))
    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.full([BLOCK_M], 1.0, tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    q = tl.load(Qb, boundary_check=(0,))
    for start_n in range(0, N, BLOCK_N):
        k = tl.load(Kb, boundary_check=(1,))
        qk = tl.dot(q, k) * sm_scale
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_ij[:, None])
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v = tl.load(Vb, boundary_check=(0,))
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_ij
        Kb = tl.advance(Kb, (0, BLOCK_N)); Vb = tl.advance(Vb, (BLOCK_N, 0))
    acc = acc / l_i[:, None]
    Ob = tl.make_block_ptr(base=O + pid_bh*stride_bh, shape=(M, HEAD_DIM),
                           strides=(stride_row, 1), offsets=(pid_m*BLOCK_M, 0),
                           block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0))
    tl.store(Ob, acc.to(O.dtype.element_ty), boundary_check=(0,))
```

## H200 self-reported observation

Correctness PASS against `torch.nn.functional.scaled_dot_product_attention`
(max abs err 7.6e-6..1.5e-5 across shapes with D=64/128). Throughput: the plain
Triton FA-2 harness reports **215-429 TFLOPS**; torch's optimized SDPA backend
reports **~1.5x-2.1x higher throughput** (448-653 TFLOPS). This is a useful
baseline observation, but it is not a measurement of the effect of PR #6660:
the compiler-internal pipeliner and warp-specialization changes were not
isolated. Full run summary:
[`data/crawl-runs/h200/triton6660-flash-attention-h200-results.md`](../../data/crawl-runs/h200/triton6660-flash-attention-h200-results.md).

## Related

- [flash-attention-4](flash-attention-4.md) — the Blackwell (SM100) successor.
- [warp-specialization](../techniques/warp-specialization.md) — the WS technique.
- [pipeline-stages](../techniques/pipeline-stages.md) — software pipelining.

## Causal variant (FA-2 CAUSAL)

A causal-mask variant (lower-triangular: `qk = where(col <= row, qk, -inf)`,
N-tiles capped at the M-block diagonal) was also validated on H200:
correctness PASS vs `torch SDPA(is_causal=True)` (err 3-6e-5), throughput
0.36x-0.66x torch SDPA (same backend-wins pattern as the non-causal kernel).
Harness + numbers:
[`data/crawl-runs/h200/op-fa2-causal-h200-results.md`](../../data/crawl-runs/h200/op-fa2-causal-h200-results.md).


## H200 benchmark replay (2026-07-21)

Original harness: `flash_attention_fwd_h200.py`. The non-causal FA2 harness passed correctness; torch SDPA was faster for every tested shape. Evidence: [`replay-2026-07-21-fa2-hopper-raw.md`](../../data/crawl-runs/h200/replay-2026-07-21-fa2-hopper-raw.md). All speed ratios are shape- and reference-specific.
