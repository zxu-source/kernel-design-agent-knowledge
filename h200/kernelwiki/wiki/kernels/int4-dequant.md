---
id: kernel-int4-dequant
title: INT4 -> BF16 Dequantization (W4A16, Hopper / H200)
type: kernel
architectures:
- sm90
tags:
- quantization
- dequantization
- quant
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
- kernel-fp8-quant
sources:
- doc-nvidia-tuning-guide
performance_claims: []
blackwell_relevance: Same W4A16 dequant on SM100; GPTQ/AWQ weight dequant (marlin).
operator_purpose: speedup
what_it_does: 'INT4->BF16 dequant (W4A16): unpack 2-int4-per-byte (two''s complement)
  * per-row scale.'
h200_validation:
  date: 2026-07-21 (phase2; self-reported summary)
  evidence_file: data/crawl-runs/h200/op-int4-dequant-h200-results.md
  harness_dir: artifacts/kernels/int4-dequant/variants
  gpu: H200
  arch: sm90
  toolchain: Triton 3.6.0, PyTorch 2.11.0+cu130
  correctness: PASS — bit-identical to torch reference (match=1.0, err=0.0).
  result: 19x-37x faster than torch (fused nibble-unpack + dequant vs torch stack/reshape/where).
  scope: Packed-uint8 INT4 weights, hidden 4096..14336, on H200. 4x weight-memory
    reduction (W4A16).
evidence_basis: H200 benchmark replay on 2026-07-21. The original harness completed
  on H200 with return code 0; its raw stdout/stderr is in the replay archive. This
  remains a local derived implementation, not an upstream-source performance claim.
---
> **Evidence boundary.** This is a locally derived canonical implementation, not a captured upstream source file. Any H200 latency/correctness text below comes from a self-reported run summary; retain it for hypothesis generation, but replay it with raw benchmark output before treating it as benchmark evidence.


## Overview

W4A16 weight dequantization: INT4 weights packed 2-per-`uint8` byte are unpacked
(nibble extract + two's-complement), multiplied by a per-row scale, and written
as bf16 for a bf16 GEMM. INT4 weights are 4x smaller than bf16, so the dequant
is done on-the-fly (or fused into the GEMM load, as in marlin/GPTQ).

```python
@triton.jit
def int4_dequant(pack_ptr, scale_ptr, o_ptr, M, N, BLOCK_N: tl.constexpr):
    row=tl.program_id(0)
    on=tl.arange(0, BLOCK_N); mask=on<N
    byte_idx=on//2; hi=on%2
    raw=tl.load(pack_ptr + row*(N//2) + byte_idx, mask=mask, other=0).to(tl.uint8)
    nib=(raw >> (hi*4)) & 0xF
    val=(nib.to(tl.int32) ^ 0x8) - 0x8          # two's-complement int4
    scale=tl.load(scale_ptr+row).to(tl.float32)
    tl.store(o_ptr+row*N+on, (val.to(tl.float32)*scale).to(tl.bfloat16), mask=mask)
```

## Purpose: SPEEDUP (memory)
INT4 weights are 4x smaller than bf16; dequantizing to bf16 for a W4A16 GEMM
cuts weight-memory bandwidth by ~4x. The fused unpack+dequant kernel is
**19x-37x faster** than torch's `stack/reshape/where` nibble extraction.
[`data/crawl-runs/h200/op-int4-dequant-h200-results.md`](../../data/crawl-runs/h200/op-int4-dequant-h200-results.md).

## H200 measured

| M x N | torch/Triton |
|---|--:|
| 4096x4096 | 19.26x |
| 8192x8192 | 30.83x |
| 8192x11008 | 31.46x |
| 8192x14336 | 34.03x |
| 16384x14336 | 36.84x |


## H200 benchmark replay (2026-07-21)

Original harness: `op_int4_dequant.py`. The original harness completed on H200 with return code 0; its raw stdout/stderr is in the replay archive. Evidence: [`replay-2026-07-21-all-experimental.md`](../../data/crawl-runs/h200/replay-2026-07-21-all-experimental.md). All speed ratios are shape- and reference-specific.
