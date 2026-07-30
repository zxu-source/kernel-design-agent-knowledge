---
id: kernel-silu-and-mul
title: SiLU-and-Mul (Hopper / H200)
type: kernel
architectures:
- sm90
tags:
- silu
- activation
- kernel-fusion
confidence: experimental
reproducibility: benchmarked
kernel_types:
- fused-kernel
languages:
- triton
- python
related:
- kernel-fused-add-rmsnorm
- kernel-gelu
sources:
- doc-nvidia-tuning-guide
performance_claims: []
blackwell_relevance: Same fused elementwise pattern on SM100; standard LLM MLP gate*up.
operator_purpose: speedup
what_it_does: 'SiLU-and-Mul: y = (gate/(1+e^-gate)) * up in one fused elementwise
  kernel.'
h200_validation:
  date: 2026-07-21 (phase2; self-reported summary)
  evidence_file: data/crawl-runs/h200/op-silu-and-mul-h200-results.md
  harness_dir: artifacts/kernels/silu-and-mul/variants
  gpu: H200
  arch: sm90
  toolchain: Triton 3.6.0, PyTorch 2.11.0+cu130
  correctness: PASS — matches torch silu(gate)*up to dtype precision (bf16 ~1.6e-2,
    fp16 ~1.9e-3).
  result: Fused kernel 1.30x-1.69x faster than torch (silu + mul, 2 launches + intermediate).
  scope: bf16/fp16, LLM MLP shapes 4096..14336, on H200.
evidence_basis: H200 benchmark replay on 2026-07-21. The original harness completed
  on H200 with return code 0; its raw stdout/stderr is in the replay archive. This
  remains a local derived implementation, not an upstream-source performance claim.
---
> **Evidence boundary.** This is a locally derived canonical implementation, not a captured upstream source file. Any H200 latency/correctness text below comes from a self-reported run summary; retain it for hypothesis generation, but replay it with raw benchmark output before treating it as benchmark evidence.


## Overview

LLM MLP gate/up activation: `y = silu(gate) * up = (gate / (1 + e^-gate)) * up`
(vLLM `silu_and_mul`, sglang, FlashInfer). Fused elementwise kernel reads `gate`
and `up` once and writes `y` once, vs torch's `silu(gate)` then `* up` (two
launches + an intermediate buffer).

```python
@triton.jit
def silu_and_mul(g_ptr, u_ptr, o_ptr, total, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid*BLOCK + tl.arange(0, BLOCK); mask = offs < total
    g = tl.load(g_ptr+offs, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(u_ptr+offs, mask=mask, other=0.0).to(tl.float32)
    silu = g / (1.0 + tl.exp(-g))
    tl.store(o_ptr+offs, (silu * u).to(o_ptr.dtype.element_ty), mask=mask)
```

## Purpose: SPEEDUP
One kernel vs two torch elementwise ops; eliminates the intermediate and one
launch. Measured **1.30x-1.69x** faster than torch. Full numbers:
[`data/crawl-runs/h200/op-silu-and-mul-h200-results.md`](../../data/crawl-runs/h200/op-silu-and-mul-h200-results.md).

## Related
- [gelu](gelu.md) — GELU activation (tanh/exact).
- [fused-add-rmsnorm](fused-add-rmsnorm.md) — residual + norm fusion.


## H200 benchmark replay (2026-07-21)

Original harness: `op_silu_and_mul.py`. The original harness completed on H200 with return code 0; its raw stdout/stderr is in the replay archive. Evidence: [`replay-2026-07-21-all-experimental.md`](../../data/crawl-runs/h200/replay-2026-07-21-all-experimental.md). All speed ratios are shape- and reference-specific.
