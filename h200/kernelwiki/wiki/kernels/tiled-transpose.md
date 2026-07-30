---
id: kernel-tiled-transpose
title: Tiled Transpose (Hopper / H200)
type: kernel
architectures:
- sm90
tags:
- transpose
- vectorized-loads
- kernel-fusion
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
blackwell_relevance: Same tiled transpose on SM100; layout transform building block.
operator_purpose: speedup
what_it_does: 'Tiled transpose: coalesced [BM,BN] load + tl.trans + coalesced store
  (square + non-square).'
h200_validation:
  date: 2026-07-21 (phase2; self-reported summary)
  evidence_file: data/crawl-runs/h200/op-tiled-transpose-h200-results.md
  harness_dir: artifacts/kernels/tiled-transpose/variants
  gpu: H200
  arch: sm90
  toolchain: Triton 3.6.0, PyTorch 2.11.0+cu130
  correctness: PASS — bit-identical to torch (err=0.0), square + non-square.
  result: 2.79x-6.01x faster than torch a.t().contiguous() (fp16 gains more).
  scope: fp32/fp16, square + non-square up to 16384^2, on H200.
evidence_basis: H200 benchmark replay on 2026-07-21. The original harness completed
  on H200 with return code 0; its raw stdout/stderr is in the replay archive. This
  remains a local derived implementation, not an upstream-source performance claim.
---
> **Evidence boundary.** This is a locally derived canonical implementation, not a captured upstream source file. Any H200 latency/correctness text below comes from a self-reported run summary; retain it for hypothesis generation, but replay it with raw benchmark output before treating it as benchmark evidence.

## H200 replay evidence (2026-07-21)

The compact replay passed correctness (`err=0`) for all shapes. It measured
`torch_ms / triton_ms` of 0.85x, 4.16x, and 4.83x for 1024², 4096², and
8192x4096; the speedup is therefore shape-dependent. Raw stdout: [`replay-2026-07-21-three-operators-raw.md`](../../data/crawl-runs/h200/replay-2026-07-21-three-operators-raw.md).


## Overview

Coalesced tiled matrix transpose: load a [BM,BN] tile from the input, transpose
it, and store coalesced to the transposed position. Avoids the uncoalesced
strided writes of a naive transpose (the classic coalescing lesson).

```python
@triton.jit
def tiled_transpose(a_ptr, o_ptr, M, N, sam, son, BM: tl.constexpr, BN: tl.constexpr):
    pm = tl.program_id(0); pn = tl.program_id(1)
    rm = pm*BM + tl.arange(0, BM); cn = pn*BN + tl.arange(0, BN)
    a  = tl.load(a_ptr + rm[:,None]*sam + cn[None,:], mask=...)   # coalesced read
    at = tl.trans(a)                                              # [BN, BM]
    rn = pn*BN + tl.arange(0, BN); cm = pm*BM + tl.arange(0, BM)
    tl.store(o_ptr + rn[:,None]*son + cm[None,:], at, mask=...)   # coalesced write
```

## Purpose: SPEEDUP (coalescing)
2.79x-6.01x faster than torch `a.t().contiguous()`. Bit-identical. fp16 gains
more (half the bytes; transpose is bandwidth-bound).
[`data/crawl-runs/h200/op-tiled-transpose-h200-results.md`](../../data/crawl-runs/h200/op-tiled-transpose-h200-results.md).

## H200 measured

| M x N | fp32 torch/Triton | fp16 torch/Triton |
|---|--:|--:|
| 4096x4096 | 2.79x | 4.38x |
| 8192x8192 | 3.29x | 5.45x |
| 8192x4096 | 3.11x | 4.93x |
| 16384x16384 | 3.50x | 6.01x |
| 4096x14336 | 3.28x | 5.44x |


## H200 benchmark replay (2026-07-21)

Original harness: `op_transpose.py`. The original harness completed on H200 with return code 0; its raw stdout/stderr is in the replay archive. Evidence: [`replay-2026-07-21-all-experimental.md`](../../data/crawl-runs/h200/replay-2026-07-21-all-experimental.md). All speed ratios are shape- and reference-specific.
