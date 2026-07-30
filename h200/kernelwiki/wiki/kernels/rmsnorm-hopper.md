---
id: kernel-rmsnorm-hopper
title: RMSNorm Forward (Hopper / H200)
type: kernel
architectures:
- sm90
tags:
- rmsnorm
- normalization
- reduction
- kernel-fusion
confidence: experimental
reproducibility: benchmarked
kernel_types:
- fused-kernel
languages:
- triton
- python
related:
- kernel-layernorm-hopper
- technique-pipeline-stages
sources:
- doc-nvidia-tuning-guide
performance_claims: []
blackwell_relevance: Identical fused-RMSNorm pattern on SM100; fp32 reduction still
  recommended for bf16/fp16 inputs.
operator_purpose: both
what_it_does: 'RMSNorm forward: y = x * rsqrt(mean(x^2)+eps) * w in one fused kernel
  (fp32 reduction).'
h200_validation:
  date: 2026-07-21 (phase2; self-reported summary)
  evidence_file: data/crawl-runs/h200/op-rmsnorm-h200-results.md
  harness_dir: artifacts/kernels/rmsnorm-hopper/variants
  gpu: H200
  arch: sm90
  toolchain: Triton 3.6.0, PyTorch 2.11.0+cu130
  correctness: PASS — matches torch.nn.functional.rms_norm to dtype precision (bf16
    ~1.5e-2..3.1e-2, fp16 ~1.9e-3; 1-row bit-identical)
  result: Fused Triton kernel 1.0x-1.40x faster than torch rms_norm (more at larger
    row counts).
  scope: Canonical RMSNorm forward on H200, bf16/fp16, hidden 4096..14336.
evidence_basis: H200 benchmark replay on 2026-07-21. The original harness completed
  on H200 with return code 0; its raw stdout/stderr is in the replay archive. This
  remains a local derived implementation, not an upstream-source performance claim.
---
> **Evidence boundary.** This is a locally derived canonical implementation, not a captured upstream source file. Any H200 latency/correctness text below comes from a self-reported run summary; retain it for hypothesis generation, but replay it with raw benchmark output before treating it as benchmark evidence.


## Overview

RMSNorm is the standard normalization in modern LLMs (block norm / final norm).
`y = x * rsqrt(mean(x^2, axis=-1) + eps) * w`. The fused forward kernel does the
squared-sum reduction, rsqrt, normalization, and weight multiply in one pass:
one read + one write of `x`, vs four separate ops.

## Purpose

- **speedup**: one fused kernel (vs reduce + rsqrt + normalize + mul) — fewer
  launches and lower memory traffic.
- **robustness**: the variance reduction is accumulated in fp32 and the `eps`
  term guards against division by ~0, avoiding bf16/fp16 overflow/underflow.

## Kernel

```python
@triton.jit
def rmsnorm_fwd(x_ptr, w_ptr, o_ptr, eps, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N); mask = offs < N
    xrow = tl.load(x_ptr + row*N + offs, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(xrow*xrow, axis=0) / N            # fp32 reduction
    rrms = tl.rsqrt(var + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(o_ptr + row*N + offs, (xrow * rrms * w).to(o_ptr.dtype.element_ty), mask=mask)
```

## H200 measured evidence

Correctness PASS vs `torch.nn.functional.rms_norm` to dtype precision. The fused
Triton kernel matches torch at small/1-row shapes (launch-bound) and is up to
**1.40x faster** at larger row counts (8192x14336). Full numbers:
[`data/crawl-runs/h200/op-rmsnorm-h200-results.md`](../../data/crawl-runs/h200/op-rmsnorm-h200-results.md).

## Related

- [layernorm-hopper](layernorm-hopper.md) — LayerNorm (with mean subtract + bias).


## H200 benchmark replay (2026-07-21)

Original harness: `op_rmsnorm.py`. The original harness completed on H200 with return code 0; its raw stdout/stderr is in the replay archive. Evidence: [`replay-2026-07-21-all-experimental.md`](../../data/crawl-runs/h200/replay-2026-07-21-all-experimental.md). All speed ratios are shape- and reference-specific.
