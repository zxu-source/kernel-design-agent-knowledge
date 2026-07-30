---
id: kernel-layernorm-hopper
title: LayerNorm Forward (Hopper / H200)
type: kernel
architectures:
- sm90
tags:
- layernorm
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
- kernel-rmsnorm-hopper
sources:
- doc-nvidia-tuning-guide
performance_claims: []
blackwell_relevance: Same fused-LayerNorm pattern on SM100; fp32 reductions recommended.
operator_purpose: both
what_it_does: 'LayerNorm forward: y=(x-mean)*rsqrt(var+eps)*w+b in one fused kernel
  (fp32 reductions).'
h200_validation:
  date: 2026-07-21 (phase2; self-reported summary)
  evidence_file: data/crawl-runs/h200/op-layernorm-h200-results.md
  harness_dir: artifacts/kernels/layernorm-hopper/variants
  gpu: H200
  arch: sm90
  toolchain: Triton 3.6.0, PyTorch 2.11.0+cu130
  correctness: PASS — matches torch.nn.functional.layer_norm to dtype precision (bf16
    ~1.5e-2..3.1e-2, fp16 ~1.9e-3..3.9e-3)
  result: Fused Triton kernel 1.10x-1.51x faster than torch layer_norm.
  scope: Canonical LayerNorm forward on H200, bf16/fp16, hidden 4096..14336.
evidence_basis: H200 benchmark replay on 2026-07-21. The original harness completed
  on H200 with return code 0; its raw stdout/stderr is in the replay archive. This
  remains a local derived implementation, not an upstream-source performance claim.
---
> **Evidence boundary.** This is a locally derived canonical implementation, not a captured upstream source file. Any H200 latency/correctness text below comes from a self-reported run summary; retain it for hypothesis generation, but replay it with raw benchmark output before treating it as benchmark evidence.


## Overview

LayerNorm (Transformer pre-attention/pre-MLP norm). `y=(x-mean)*rsqrt(var+eps)*w+b`.
The fused forward computes `sum_x` and `sum_x2` in one pass over the row
(`var = E[x^2] - E[x]^2`), then centers, normalizes, scales, and adds bias.

```python
@triton.jit
def layernorm_fwd(x_ptr, w_ptr, b_ptr, o_ptr, eps, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N); mask = offs < N
    x = tl.load(x_ptr + row*N + offs, mask=mask, other=0.0).to(tl.float32)
    sum_x = tl.sum(x, axis=0); sum_x2 = tl.sum(x*x, axis=0)
    mean = sum_x / N; var = sum_x2 / N - mean*mean
    rrms = tl.rsqrt(var + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(o_ptr + row*N + offs, ((x-mean)*rrms*w + b).to(o_ptr.dtype.element_ty), mask=mask)
```

## Purpose

- **speedup**: one fused kernel (vs reduce + center + normalize + scale + bias).
- **robustness**: fp32 reductions + eps; note one-pass `E[x^2]-E[x]^2` can lose
  precision vs two-pass when variance is tiny — for adversarial inputs use the
  two-pass form (re-scan after centering).

## H200 measured evidence

Correctness PASS vs torch to dtype precision. Fused Triton kernel **1.10x-1.51x
faster** than `torch.nn.functional.layer_norm`. Full numbers:
[`data/crawl-runs/h200/op-layernorm-h200-results.md`](../../data/crawl-runs/h200/op-layernorm-h200-results.md).

## Related
- [rmsnorm-hopper](rmsnorm-hopper.md) — RMSNorm (no mean subtract / no bias).


## Robustness: fp32 reduction is required (validated)

A focused test pits fp32 variance reduction against bf16 reduction in LayerNorm.
**fp32** stays accurate (rel err ~0.25%) across hidden N up to 57344; **bf16**
reduction degrades from 0.48% (N=4096) to 3.64% (N=57344) — 8x worse, because
bf16 (~3 decimals) drops sub-ULP contributions to the growing sum_x/sum_x2, and
for adversarial large-magnitude inputs the bf16 sum can overflow to inf. **Always
reduce in fp32** for LayerNorm/RMSNorm (the kernel above does). Details:
[`data/crawl-runs/h200/op-layernorm-fp32-overflow-h200-results.md`](../../data/crawl-runs/h200/op-layernorm-fp32-overflow-h200-results.md).


## H200 benchmark replay (2026-07-21)

Original harness: `op_layernorm.py`. The original harness completed on H200 with return code 0; its raw stdout/stderr is in the replay archive. Evidence: [`replay-2026-07-21-all-experimental.md`](../../data/crawl-runs/h200/replay-2026-07-21-all-experimental.md). All speed ratios are shape- and reference-specific.
