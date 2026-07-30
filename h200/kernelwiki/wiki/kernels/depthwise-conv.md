---
id: kernel-depthwise-conv
title: Depthwise Convolution (Hopper / H200)
type: kernel
architectures:
- sm90
tags:
- convolution
confidence: experimental
reproducibility: benchmarked
kernel_types:
- fused-kernel
languages:
- triton
- python
related:
- kernel-bf16-gemm-hopper
sources:
- doc-nvidia-tuning-guide
performance_claims: []
blackwell_relevance: Same depthwise conv on SM100; CNN building block (MobileNet/EfficientNet).
operator_purpose: speedup
what_it_does: 'Depthwise conv (direct, per-channel, 3x3 stride1 pad1): gather 9 taps
  per output; each channel independent.'
h200_validation:
  date: 2026-07-21 (phase2; self-reported summary)
  evidence_file: data/crawl-runs/h200/op-depthwise-conv-h200-results.md
  harness_dir: artifacts/kernels/depthwise-conv/variants
  gpu: H200
  arch: sm90
  toolchain: Triton 3.6.0, PyTorch 2.11.0+cu130
  correctness: PASS — bit-identical to torch conv2d(groups=C) (err=0).
  result: ~parity at small spatial (128x128, 64x64); 1.53x-1.76x faster at large spatial
    (256x256, 512x512).
  scope: fp32, 3x3 stride1 pad1, channels 64..256, on H200.
evidence_basis: H200 benchmark replay on 2026-07-21. The original harness completed
  on H200 with return code 0; its raw stdout/stderr is in the replay archive. This
  remains a local derived implementation, not an upstream-source performance claim.
---
> **Evidence boundary.** This is a locally derived canonical implementation, not a captured upstream source file. Any H200 latency/correctness text below comes from a self-reported run summary; retain it for hypothesis generation, but replay it with raw benchmark output before treating it as benchmark evidence.


## Overview

Depthwise conv: each output channel is independent (no cross-channel mixing).
`out[c,oh,ow] = sum_{kh,kw} in[c, oh+kh-pad, ow+kw-pad] * w[c,kh,kw]`. Direct
kernel (no im2col): one program per (channel, spatial tile), gathers the 9 taps.

```python
@triton.jit
def dwconv2d(in_ptr, w_ptr, o_ptr, C, H, W, K, PAD, BLOCK):
    c=tl.program_id(0); pid=tl.program_id(1)
    p = pid*BLOCK + tl.arange(0, BLOCK); oh = p//W; ow = p%W; mask = p<H*W
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for kh in tl.static_range(K):
        for kw in tl.static_range(K):
            ih = oh + kh - PAD; iw = ow + kw - PAD
            valid = (ih>=0)&(ih<H)&(iw>=0)&(iw<W)&mask
            wv = tl.load(w_ptr + c*K*K + kh*K + kw)
            iv = tl.load(in_ptr + c*H*W + ih*W + iw, mask=valid, other=0.0)
            acc += iv * wv
    tl.store(o_ptr + c*H*W + p, acc, mask=mask)
```

## Purpose: SPEEDUP (characterization)
Bit-identical to torch. ~parity at small spatial; **1.53x-1.76x faster** at large
spatial (256x256, 512x512) where the direct-gather amortizes over the tile better.
[`data/crawl-runs/h200/op-depthwise-conv-h200-results.md`](../../data/crawl-runs/h200/op-depthwise-conv-h200-results.md).

## H200 measured (3x3, stride1, pad1)

| CxHxW | torch/Triton |
|---|--:|
| 64x128x128 | 0.84x |
| 128x128x128 | 1.04x |
| 256x64x64 | 0.83x |
| 128x256x256 | 1.53x |
| 64x512x512 | 1.76x |


## H200 benchmark replay (2026-07-21)

Original harness: `op_depthwise_conv.py`. The original harness completed on H200 with return code 0; its raw stdout/stderr is in the replay archive. Evidence: [`replay-2026-07-21-all-experimental.md`](../../data/crawl-runs/h200/replay-2026-07-21-all-experimental.md). All speed ratios are shape- and reference-specific.
