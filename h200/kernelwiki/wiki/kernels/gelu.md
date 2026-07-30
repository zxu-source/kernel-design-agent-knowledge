---
id: kernel-gelu
title: GELU tanh-approx (Hopper / H200)
type: kernel
architectures:
- sm90
tags:
- gelu
- activation
confidence: experimental
reproducibility: benchmarked
kernel_types:
- fused-kernel
languages:
- triton
- python
related:
- kernel-silu-and-mul
sources:
- doc-nvidia-tuning-guide
performance_claims: []
blackwell_relevance: Same elementwise GELU on SM100; standard Transformer/FFN activation.
operator_purpose: speedup
what_it_does: 'GELU (tanh approx): 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3))) elementwise
  (tanh via exp).'
h200_validation:
  date: 2026-07-21 (phase2; self-reported summary)
  evidence_file: data/crawl-runs/h200/op-gelu-h200-results.md
  harness_dir: artifacts/kernels/gelu/variants
  gpu: H200
  arch: sm90
  toolchain: Triton 3.6.0, PyTorch 2.11.0+cu130
  correctness: PASS — bf16 bit-identical, fp16 ~6e-5 vs torch gelu(approximate=tanh).
  result: '~parity with torch (0.87x-1.12x): gelu is already a single fused op; Triton
    slight win at large sizes.'
  scope: 'bf16/fp16, LLM shapes 4096..14336, on H200. Note: tl.tanh absent in Triton
    3.6 -> tanh via (e^2x-1)/(e^2x+1).'
evidence_basis: H200 benchmark replay on 2026-07-21. The original harness completed
  on H200 with return code 0; its raw stdout/stderr is in the replay archive. This
  remains a local derived implementation, not an upstream-source performance claim.
---
> **Evidence boundary.** This is a locally derived canonical implementation, not a captured upstream source file. Any H200 latency/correctness text below comes from a self-reported run summary; retain it for hypothesis generation, but replay it with raw benchmark output before treating it as benchmark evidence.


## Overview

GELU (tanh approximation), the standard Transformer/FFN activation:
`0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))`. Note: Triton 3.6 has no
`tl.tanh`, so tanh is computed via `(e^(2x)-1)/(e^(2x)+1)`.

```python
@triton.jit
def gelu_tanh(x_ptr, o_ptr, total, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid*BLOCK + tl.arange(0, BLOCK); mask = offs < total
    x = tl.load(x_ptr+offs, mask=mask, other=0.0).to(tl.float32)
    c = 0.7978845608028654   # sqrt(2/pi)
    inner = c * (x + 0.044715 * x*x*x)
    e2 = tl.exp(2.0 * inner); tanh = (e2 - 1.0) / (e2 + 1.0)
    tl.store(o_ptr+offs, (0.5 * x * (1.0 + tanh)).to(o_ptr.dtype.element_ty), mask=mask)
```

## Purpose: SPEEDUP (marginal — characterization)
GELU is already a single fused op in torch, so this is a head-to-head rather
than a fusion win. Triton matches torch (0.87x at small / launch-bound,
1.05-1.12x at large). The value is the validated, self-contained impl.
[`data/crawl-runs/h200/op-gelu-h200-results.md`](../../data/crawl-runs/h200/op-gelu-h200-results.md).

## Related
- [silu-and-mul](silu-and-mul.md) — SiLU*mul (LLM MLP), clearer fusion win.


## H200 benchmark replay (2026-07-21)

Original harness: `op_gelu.py`. The original harness completed on H200 with return code 0; its raw stdout/stderr is in the replay archive. Evidence: [`replay-2026-07-21-all-experimental.md`](../../data/crawl-runs/h200/replay-2026-07-21-all-experimental.md). All speed ratios are shape- and reference-specific.
