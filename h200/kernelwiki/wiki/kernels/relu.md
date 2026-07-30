---
id: kernel-relu
title: ReLU Forward + Backward (Hopper / H200)
type: kernel
architectures:
- sm90
tags:
- activation
confidence: experimental
reproducibility: benchmarked
kernel_types:
- fused-kernel
languages:
- triton
- python
related:
- kernel-gelu
- kernel-silu-and-mul
sources:
- doc-nvidia-tuning-guide
performance_claims: []
blackwell_relevance: Same ReLU on SM100; the simplest activation (CNN/MLP).
operator_purpose: speedup
what_it_does: ReLU forward (y=max(x,0)) + backward (grad_x=grad_y*(x>0)); fused elementwise.
h200_validation:
  date: 2026-07-21 (phase2; self-reported summary)
  evidence_file: data/crawl-runs/h200/op-relu-h200-results.md
  harness_dir: artifacts/kernels/relu/variants
  gpu: H200
  arch: sm90
  toolchain: Triton 3.6.0, PyTorch 2.11.0+cu130
  correctness: PASS — bit-identical (err=0) for fwd and bwd.
  result: fwd 0.80x-0.94x (torch.relu already a single fast op); bwd 1.01x-1.35x (modest
    win over autograd).
  scope: fp32/bf16, MLP shapes, on H200.
evidence_basis: H200 benchmark replay on 2026-07-21. The original harness completed
  on H200 with return code 0; its raw stdout/stderr is in the replay archive. This
  remains a local derived implementation, not an upstream-source performance claim.
---
> **Evidence boundary.** This is a locally derived canonical implementation, not a captured upstream source file. Any H200 latency/correctness text below comes from a self-reported run summary; retain it for hypothesis generation, but replay it with raw benchmark output before treating it as benchmark evidence.


## Overview

ReLU: forward `y = max(x, 0)`, backward `grad_x = grad_y * (x > 0)`. The simplest
activation; one elementwise pass each.

```python
@triton.jit
def relu_fwd(x_ptr, o_ptr, total, BLOCK):
    ...; x=tl.load(...); tl.store(o_ptr+offs, tl.maximum(x, 0.0), mask=...)
@triton.jit
def relu_bwd(x_ptr, gy_ptr, gx_ptr, total, BLOCK):
    ...; x=tl.load(...); gy=tl.load(...)
    tl.store(gx_ptr+offs, gy * (x > 0.0).to(gy.dtype), mask=...)
```

## Purpose: SPEEDUP (characterization)
Correctness bit-identical. **Forward 0.80x-0.94x**: `torch.relu` is already a
single highly-tuned elementwise op, so the Triton kernel is launch-bound and
slightly slower. **Backward 1.01x-1.35x**: modest win over autograd overhead.
Use torch.relu for the forward; the Triton backward is a marginal win.
[`data/crawl-runs/h200/op-relu-h200-results.md`](../../data/crawl-runs/h200/op-relu-h200-results.md).

## H200 measured (fp32 / bf16)

| MxN | fwd torch/Triton (fp32) | bwd torch/Triton (bf16) |
|---|--:|--:|
| 4096x4096 | 0.84x | 1.35x |
| 8192x8192 | 0.92x | 1.12x |
| 8192x14336 | 0.93x | 1.07x |
| 16384x14336 | 0.94x | 1.02x |


## H200 benchmark replay (2026-07-21)

Original harness: `op_relu.py`. The original harness completed on H200 with return code 0; its raw stdout/stderr is in the replay archive. Evidence: [`replay-2026-07-21-all-experimental.md`](../../data/crawl-runs/h200/replay-2026-07-21-all-experimental.md). All speed ratios are shape- and reference-specific.
