---
id: kernel-grad-clip
title: Gradient Norm Clipping (Hopper / H200)
type: kernel
architectures:
- sm90
tags:
- reduction
confidence: experimental
reproducibility: benchmarked
kernel_types:
- fused-kernel
languages:
- triton
- python
related:
- kernel-block-reduce
- kernel-adam-step
sources:
- doc-nvidia-tuning-guide
performance_claims: []
blackwell_relevance: Same grad-clip on SM100; training stability (prevents gradient
  explosion).
operator_purpose: both
what_it_does: 'Gradient norm clipping: compute total norm (sqrt sum g^2), then scale
  g *= min(1, max_norm/(norm+eps)).'
h200_validation:
  date: 2026-07-21 (phase2; self-reported summary)
  evidence_file: data/crawl-runs/h200/op-grad-clip-h200-results.md
  harness_dir: artifacts/kernels/grad-clip/variants
  gpu: H200
  arch: sm90
  toolchain: Triton 3.6.0, PyTorch 2.11.0+cu130
  correctness: PASS — norm err ~3e-7, grad err ~5e-10 vs torch manual.
  result: record-negative — 0.31x-0.40x torch (2-kernel approach slower than torch's
    optimized norm+scale).
  scope: fp32, params up to 33M, max_norm=1.0, on H200.
evidence_basis: H200 benchmark replay on 2026-07-21. The original harness completed
  on H200 with return code 0; its raw stdout/stderr is in the replay archive. This
  remains a local derived implementation, not an upstream-source performance claim.
---
> **Evidence boundary.** This is a locally derived canonical implementation, not a captured upstream source file. Any H200 latency/correctness text below comes from a self-reported run summary; retain it for hypothesis generation, but replay it with raw benchmark output before treating it as benchmark evidence.


## Overview

Gradient norm clipping (training stability): compute the total gradient norm
(`sqrt(sum(g^2))`), then scale all gradients by `min(1, max_norm/(norm+eps))` if
the norm exceeds `max_norm`. Prevents gradient explosion. Two kernels: an
`atomic_add` sum-of-squares reduction + a conditional elementwise scale.

```python
@triton.jit
def sum_sq_kernel(g_ptr, out_ptr, N, BLOCK):
    ...; s=tl.sum(g*g, axis=0); tl.atomic_add(out_ptr, s)
@triton.jit
def scale_kernel(g_ptr, scale, N, BLOCK):
    ...; g=tl.load(...); tl.store(g_ptr+offs, g*scale, mask=...)
# host: norm=sqrt(sum_sq); scale=min(1, max_norm/(norm+eps)); if scale<1: scale_kernel[...]
```

## Purpose: BOTH
- **robustness**: prevents gradient explosion (caps the norm at max_norm).
- **speedup**: record-negative — 0.31x-0.40x torch. The 2-kernel approach
  (atomic_add reduction + separate scale) is slower than torch's optimized
  `norm()` + conditional scale (torch's CUB-style norm is faster). Use
  `torch.nn.utils.clip_grad_norm_` for production.
[`data/crawl-runs/h200/op-grad-clip-h200-results.md`](../../data/crawl-runs/h200/op-grad-clip-h200-results.md).

## H200 measured

| N (params) | torch/Triton |
|--:|--:|
| 1M | 0.40x |
| 4M | 0.31x |
| 16M | 0.34x |
| 33M | 0.35x |


## H200 benchmark replay (2026-07-21)

Original harness: `op_grad_clip.py`. The original harness completed on H200 with return code 0; its raw stdout/stderr is in the replay archive. Evidence: [`replay-2026-07-21-all-experimental.md`](../../data/crawl-runs/h200/replay-2026-07-21-all-experimental.md). All speed ratios are shape- and reference-specific.
