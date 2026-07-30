---
id: kernel-l2-normalize
title: L2-Normalize (unit norm per row, Hopper / H200)
type: kernel
architectures:
- sm90
tags:
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
- kernel-layernorm-hopper
sources:
- doc-nvidia-tuning-guide
performance_claims: []
blackwell_relevance: Same L2-normalize on SM100; embedding/attention scaling.
operator_purpose: speedup
what_it_does: 'L2-normalize (unit norm per row): y=x*rsqrt(sum(x^2)+eps); fused.'
h200_validation:
  date: 2026-07-21 (phase2; self-reported summary)
  evidence_file: data/crawl-runs/h200/op-l2-normalize-h200-results.md
  harness_dir: artifacts/kernels/l2-normalize/variants
  gpu: H200
  arch: sm90
  toolchain: Triton 3.6.0, PyTorch 2.11.0+cu130
  correctness: PASS — fp32 ~1e-8, bf16 ~2e-4 vs torch.nn.functional.normalize.
  result: 1.95x-3.31x faster than torch (bf16 gains more).
  scope: fp32/bf16, hidden 4096..14336, on H200.
evidence_basis: H200 benchmark replay on 2026-07-21. The original harness completed
  on H200 with return code 0; its raw stdout/stderr is in the replay archive. This
  remains a local derived implementation, not an upstream-source performance claim.
---
> **Evidence boundary.** This is a locally derived canonical implementation, not a captured upstream source file. Any H200 latency/correctness text below comes from a self-reported run summary; retain it for hypothesis generation, but replay it with raw benchmark output before treating it as benchmark evidence.


## Overview

L2-normalize: scale each row to unit L2 norm. `y = x * rsqrt(sum(x^2) + eps)`.
One fused pass per row (sum-squared reduction + rsqrt + scale).

```python
@triton.jit
def l2norm_fwd(x_ptr, o_ptr, eps, N, BLOCK_N):
    row=tl.program_id(0); offs=tl.arange(0, BLOCK_N); mask=offs<N
    x=tl.load(x_ptr+row*N+offs, mask=mask, other=0.0).to(tl.float32)
    s=tl.sum(x*x, axis=0); inv=tl.rsqrt(s+eps)
    tl.store(o_ptr+row*N+offs, (x*inv).to(o_ptr.dtype.element_ty), mask=mask)
```

## Purpose: SPEEDUP (fusion)
1.95x-3.31x faster than torch (fused reduction+scale vs torch's multi-op normalize;
bf16 gains more — half the bytes, bandwidth-bound).
[`data/crawl-runs/h200/op-l2-normalize-h200-results.md`](../../data/crawl-runs/h200/op-l2-normalize-h200-results.md).

## H200 measured

| MxN | fp32 torch/Triton | bf16 torch/Triton |
|---|--:|--:|
| 4096x4096 | 1.95x | 2.64x |
| 8192x8192 | 2.06x | 3.24x |
| 8192x11008 | 1.99x | 3.13x |
| 8192x14336 | 2.02x | 3.22x |
| 16384x14336 | 2.03x | 3.31x |


## H200 benchmark replay (2026-07-21)

Original harness: `op_l2norm.py`. The original harness completed on H200 with return code 0; its raw stdout/stderr is in the replay archive. Evidence: [`replay-2026-07-21-all-experimental.md`](../../data/crawl-runs/h200/replay-2026-07-21-all-experimental.md). All speed ratios are shape- and reference-specific.
