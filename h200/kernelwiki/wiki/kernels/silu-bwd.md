---
id: kernel-silu-bwd
title: SiLU Backward (Hopper / H200)
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
- kernel-silu-and-mul
- kernel-gelu-bwd
sources:
- doc-nvidia-tuning-guide
performance_claims: []
blackwell_relevance: Same SiLU backward on SM100; training gradient.
operator_purpose: speedup
what_it_does: 'SiLU backward: grad_x=grad_y*sigmoid(x)*(1+x*(1-sigmoid(x))); fused
  elementwise.'
h200_validation:
  date: 2026-07-21 (phase2; self-reported summary)
  evidence_file: data/crawl-runs/h200/op-silu-bwd-h200-results.md
  harness_dir: artifacts/kernels/silu-bwd/variants
  gpu: H200
  arch: sm90
  toolchain: Triton 3.6.0, PyTorch 2.11.0+cu130
  correctness: PASS — fp32 ~7e-7, bf16 ~4e-3 (sigmoid bf16 noise) vs torch autograd.
  result: 1.02x-1.29x faster than torch autograd (backward-only timed).
  scope: fp32/bf16, MLP shapes, on H200.
evidence_basis: H200 benchmark replay on 2026-07-21. The original harness completed
  on H200 with return code 0; its raw stdout/stderr is in the replay archive. This
  remains a local derived implementation, not an upstream-source performance claim.
---
> **Evidence boundary.** This is a locally derived canonical implementation, not a captured upstream source file. Any H200 latency/correctness text below comes from a self-reported run summary; retain it for hypothesis generation, but replay it with raw benchmark output before treating it as benchmark evidence.


## Overview

SiLU backward (training gradient): `grad_x = grad_y * silu'(x)`,
`silu'(x) = sigmoid(x) * (1 + x*(1-sigmoid(x)))`.

```python
@triton.jit
def silu_bwd(x_ptr, gy_ptr, gx_ptr, total, BLOCK):
    pid=tl.program_id(0); offs=pid*BLOCK+tl.arange(0,BLOCK); mask=offs<total
    x=tl.load(x_ptr+offs, mask=mask).to(tl.float32); gy=tl.load(gy_ptr+offs, mask=mask).to(tl.float32)
    sig=1.0/(1.0+tl.exp(-x)); dsilu=sig*(1.0 + x*(1.0-sig))
    tl.store(gx_ptr+offs, (gy*dsilu).to(gx_ptr.dtype.element_ty), mask=mask)
```

## Purpose: SPEEDUP
1.02x-1.29x faster than torch autograd (backward-only timed; bf16 gains more).
[`data/crawl-runs/h200/op-silu-bwd-h200-results.md`](../../data/crawl-runs/h200/op-silu-bwd-h200-results.md).

## H200 measured

| MxN | fp32 torch/Triton | bf16 torch/Triton |
|---|--:|--:|
| 4096x4096 | 1.20x | 1.29x |
| 8192x8192 | 1.06x | 1.11x |
| 8192x11008 | 1.03x | 1.07x |
| 8192x14336 | 1.04x | 1.10x |
| 16384x14336 | 1.02x | 1.03x |


## H200 benchmark replay (2026-07-21)

Original harness: `op_silu_bwd.py`. The original harness completed on H200 with return code 0; its raw stdout/stderr is in the replay archive. Evidence: [`replay-2026-07-21-all-experimental.md`](../../data/crawl-runs/h200/replay-2026-07-21-all-experimental.md). All speed ratios are shape- and reference-specific.
