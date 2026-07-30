---
id: kernel-per-channel-quant
title: Per-Channel Symmetric INT8 Quantization (Hopper / H200)
type: kernel
architectures:
- sm90
tags:
- quantization
- quant
- reduction
- kernel-fusion
confidence: experimental
reproducibility: benchmarked
kernel_types:
- quantization
- fused-kernel
languages:
- triton
- python
related:
- kernel-per-tensor-quant
- kernel-fp8-block-scale-gemm
sources:
- doc-nvidia-tuning-guide
performance_claims: []
blackwell_relevance: Same per-channel quant on SM100; per-output-channel weight quant
  (TRT-LLM/vLLM W8A8).
operator_purpose: both
what_it_does: 'Per-channel (per-row) symmetric INT8 quant: fused amax+scale+round/clamp/cast
  per row.'
h200_validation:
  date: 2026-07-21 (phase2; self-reported summary)
  evidence_file: data/crawl-runs/h200/op-per-channel-quant-h200-results.md
  harness_dir: artifacts/kernels/per-channel-quant/variants
  gpu: H200
  arch: sm90
  toolchain: Triton 3.6.0, PyTorch 2.11.0+cu130
  correctness: PASS — match ~1.0, maxdiff<=1 (rounding tie ~0.03% of bf16 elems),
    per-channel scales exact.
  result: 6.25x-12.08x faster than torch (fused per-row amax+quantize vs torch's multi-op
    reduction chain).
  scope: fp32/bf16 weight matrices, hidden 4096..14336, on H200.
evidence_basis: H200 benchmark replay on 2026-07-21. The original harness completed
  on H200 with return code 0; its raw stdout/stderr is in the replay archive. This
  remains a local derived implementation, not an upstream-source performance claim.
---
> **Evidence boundary.** This is a locally derived canonical implementation, not a captured upstream source file. Any H200 latency/correctness text below comes from a self-reported run summary; retain it for hypothesis generation, but replay it with raw benchmark output before treating it as benchmark evidence.


## Overview

Per-output-channel (per-row) symmetric INT8 weight quantization: each row gets
its own `scale = amax_row/127`, preserving per-channel dynamic range (more
accurate than per-tensor). One fused kernel per row loads the row, computes its
amax, scales, rounds/clamps, and casts to int8.

```python
@triton.jit
def per_channel_quant(w_ptr, o_ptr, scale_ptr, M, N, BLOCK_N: tl.constexpr):
    row=tl.program_id(0)
    offs=tl.arange(0, BLOCK_N); mask=offs<N
    w=tl.load(w_ptr+row*N+offs, mask=mask, other=0.0).to(tl.float32)
    amax=tl.maximum(tl.max(tl.abs(w),axis=0), 1e-12)
    inv=127.0/amax
    q=tl.extra.libdevice.llrint(w*inv); q=tl.maximum(tl.minimum(q,127.0),-128.0)
    tl.store(o_ptr+row*N+offs, q.to(tl.int8), mask=mask)
    tl.store(scale_ptr+row, amax/127.0)
```

## Purpose: BOTH
- **speedup**: one fused per-row kernel vs torch's `abs().amax(dim=-1)` +
  `/scale` + `round` + `clamp` + `cast` (several passes + a row reduction +
  intermediates). **6.25x-12.08x** faster than torch.
- **robustness/accuracy**: per-channel scales keep each output channel's range
  (vs a single per-tensor scale that loses precision for small-magnitude channels).

[`data/crawl-runs/h200/op-per-channel-quant-h200-results.md`](../../data/crawl-runs/h200/op-per-channel-quant-h200-results.md).

## H200 measured

| M x N | dtype | torch/Triton |
|---|---|--:|
| 4096x4096 | fp32 | 6.25x |
| 8192x8192 | fp32 | 9.07x |
| 8192x11008 | fp32 | 8.01x |
| 8192x14336 | fp32 | 9.28x |
| 16384x14336 | fp32 | 9.90x |
| (bf16 tracks within 1-2x, often faster) | | |


## H200 benchmark replay (2026-07-21)

Original harness: `op_per_channel_quant.py`. The original harness completed on H200 with return code 0; its raw stdout/stderr is in the replay archive. Evidence: [`replay-2026-07-21-all-experimental.md`](../../data/crawl-runs/h200/replay-2026-07-21-all-experimental.md). All speed ratios are shape- and reference-specific.
