---
id: kernel-groupwise-quant
title: Groupwise (G=128) INT8 Quantization (Hopper / H200)
type: kernel
architectures:
- sm90
tags:
- quantization
- quant
- fine-grained-quantization
- reduction
confidence: experimental
reproducibility: benchmarked
kernel_types:
- quantization
- fused-kernel
languages:
- triton
- python
related:
- kernel-per-channel-quant
- kernel-fp8-block-scale-gemm
sources:
- doc-nvidia-tuning-guide
performance_claims: []
blackwell_relevance: Same groupwise quant on SM100; AWQ/GPTQ group_size=128 weight
  quant.
operator_purpose: both
what_it_does: 'Groupwise (G=128) symmetric INT8 quant: per-group amax+scale+round/clamp/cast
  (AWQ/GPTQ granularity).'
h200_validation:
  date: 2026-07-21 (phase2; self-reported summary)
  evidence_file: data/crawl-runs/h200/op-groupwise-quant-h200-results.md
  harness_dir: artifacts/kernels/groupwise-quant/variants
  gpu: H200
  arch: sm90
  toolchain: Triton 3.6.0, PyTorch 2.11.0+cu130
  correctness: PASS — match ~1.0 (maxdiff<=1, ~0.04% bf16 elems differ by 1), group
    scales exact.
  result: 2.70x-3.42x faster than torch (reshape/amax/div/round/clamp/cast).
  scope: fp32/bf16 weight matrices, group=128, hidden 4096..14336, on H200.
evidence_basis: H200 benchmark replay on 2026-07-21. The original harness completed
  on H200 with return code 0; its raw stdout/stderr is in the replay archive. This
  remains a local derived implementation, not an upstream-source performance claim.
---
> **Evidence boundary.** This is a locally derived canonical implementation, not a captured upstream source file. Any H200 latency/correctness text below comes from a self-reported run summary; retain it for hypothesis generation, but replay it with raw benchmark output before treating it as benchmark evidence.


## Overview

Groupwise (group_size=128) symmetric INT8 quantization — the AWQ/GPTQ granularity.
Each group of 128 consecutive elements gets its own scale: finer than per-channel
(better accuracy at large N) and coarser than per-element (compact scale storage).
One kernel per `(row, group)` computes the group amax, scale, and int8 cast.

```python
@triton.jit
def groupwise_quant(w_ptr, o_ptr, scale_ptr, M, Ngroups, G: tl.constexpr, N: tl.constexpr):
    pid=tl.program_id(0); row=pid//Ngroups; g=pid%Ngroups
    offs=tl.arange(0,G); base=row*N + g*G
    w=tl.load(w_ptr+base+offs, ...).to(tl.float32)
    amax=tl.maximum(tl.max(tl.abs(w),axis=0), 1e-12)
    q=tl.extra.libdevice.llrint(w*(127.0/amax)); q=clamp(q,-128,127)
    tl.store(o_ptr+base+offs, q.to(tl.int8), ...); tl.store(scale_ptr+row*Ngroups+g, amax/127.0)
```

## Purpose: BOTH
- **speedup**: fused per-group amax+quantize vs torch's reshape/amax/div/round/
  clamp/cast (2.70x-3.42x faster).
- **accuracy**: per-group (128) scales preserve local dynamic range better than a
  single per-channel scale for large hidden dims (AWQ/GPTQ motivation).

[`data/crawl-runs/h200/op-groupwise-quant-h200-results.md`](../../data/crawl-runs/h200/op-groupwise-quant-h200-results.md).

## H200 measured

| M x N | fp32 torch/Triton | bf16 torch/Triton |
|---|--:|--:|
| 4096x4096 | 2.70x | 3.23x |
| 8192x8192 | 2.79x | 3.38x |
| 8192x11008 | 2.80x | 3.39x |
| 8192x14336 | 2.81x | 3.40x |
| 16384x14336 | 2.82x | 3.42x |


## H200 benchmark replay (2026-07-21)

Original harness: `op_groupwise_quant.py`. The original harness completed on H200 with return code 0; its raw stdout/stderr is in the replay archive. Evidence: [`replay-2026-07-21-all-experimental.md`](../../data/crawl-runs/h200/replay-2026-07-21-all-experimental.md). All speed ratios are shape- and reference-specific.
