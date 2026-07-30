---
id: kernel-max-pool
title: Max Pooling (Hopper / H200)
type: kernel
architectures:
- sm90
tags:
- pooling
confidence: experimental
reproducibility: benchmarked
kernel_types:
- fused-kernel
languages:
- triton
- python
related:
- kernel-depthwise-conv
sources:
- doc-nvidia-tuning-guide
performance_claims: []
blackwell_relevance: Same max pool on SM100; CNN building block.
operator_purpose: speedup
what_it_does: 'Max pooling (2x2 stride2): per-output max over window of 4 input taps.'
h200_validation:
  date: 2026-07-21 (phase2; self-reported summary)
  evidence_file: data/crawl-runs/h200/op-max-pool-h200-results.md
  harness_dir: artifacts/kernels/max-pool/variants
  gpu: H200
  arch: sm90
  toolchain: Triton 3.6.0, PyTorch 2.11.0+cu130
  correctness: PASS — bit-identical to torch.max_pool2d (err=0).
  result: 0.54x-0.68x small spatial (launch-bound); 1.24x-1.45x large spatial (256x256,
    512x512).
  scope: fp32, 2x2 stride2, channels 64..256, on H200.
evidence_basis: H200 benchmark replay on 2026-07-21. The original harness completed
  on H200 with return code 0; its raw stdout/stderr is in the replay archive. This
  remains a local derived implementation, not an upstream-source performance claim.
---
> **Evidence boundary.** This is a locally derived canonical implementation, not a captured upstream source file. Any H200 latency/correctness text below comes from a self-reported run summary; retain it for hypothesis generation, but replay it with raw benchmark output before treating it as benchmark evidence.


## Overview

Max pooling (2x2, stride 2): each output = max over the 2x2 window of 4 input
taps. One program per (channel, spatial tile).

```python
@triton.jit
def maxpool2d(in_ptr, o_ptr, C, H, W, OH, OW, BLOCK):
    c=tl.program_id(0); pid=tl.program_id(1)
    p=pid*BLOCK+tl.arange(0,BLOCK); mask=p<OH*OW; oh=p//OW; ow=p%OW
    base=c*H*W + (oh*2)*W + (ow*2)
    a=tl.load(in_ptr+base, mask=mask, other=-1e30); b=tl.load(in_ptr+base+1, mask=mask&(ow*2+1<W), other=-1e30)
    cc=tl.load(in_ptr+base+W, mask=mask&(oh*2+1<H), other=-1e30); d=tl.load(in_ptr+base+W+1, mask=mask&(oh*2+1<H)&(ow*2+1<W), other=-1e30)
    m=tl.maximum(tl.maximum(a,b), tl.maximum(cc,d))
    tl.store(o_ptr+c*OH*OW+p, m, mask=mask)
```

## Purpose: SPEEDUP (characterization)
Bit-identical to torch. 0.54x-0.68x at small spatial (launch-bound); 1.24x-1.45x
at large spatial (the 4-tap gather amortizes over the tile).
[`data/crawl-runs/h200/op-max-pool-h200-results.md`](../../data/crawl-runs/h200/op-max-pool-h200-results.md).

## H200 measured (2x2, stride2)

| CxHxW | torch/Triton |
|---|--:|
| 64x128x128 | 0.54x |
| 128x128x128 | 0.68x |
| 256x64x64 | 0.56x |
| 128x256x256 | 1.24x |
| 64x512x512 | 1.45x |


## H200 benchmark replay (2026-07-21)

Original harness: `op_maxpool.py`. The original harness completed on H200 with return code 0; its raw stdout/stderr is in the replay archive. Evidence: [`replay-2026-07-21-all-experimental.md`](../../data/crawl-runs/h200/replay-2026-07-21-all-experimental.md). All speed ratios are shape- and reference-specific.
