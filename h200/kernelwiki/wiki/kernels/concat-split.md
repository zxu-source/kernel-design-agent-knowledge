---
id: kernel-concat-split
title: Fused Concat (Hopper / H200)
type: kernel
architectures:
- sm90
tags:
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
- kernel-fused-gemm-bias-act
sources:
- doc-nvidia-tuning-guide
performance_claims: []
blackwell_relevance: Same concat pattern on SM100; QKV-projection output concat /
  attention head concat.
operator_purpose: speedup
what_it_does: Fused concat of 3 [M,Dk] tensors along last dim into [M,3*Dk] in one
  kernel (one program/row).
h200_validation:
  date: 2026-07-21 (phase2; self-reported summary)
  evidence_file: data/crawl-runs/h200/op-concat-h200-results.md
  harness_dir: artifacts/kernels/concat-split/variants
  gpu: H200
  arch: sm90
  toolchain: Triton 3.6.0, PyTorch 2.11.0+cu130
  correctness: PASS — bit-identical to torch.cat (err=0.0).
  result: record-negative — ~parity with torch.cat (0.88x-1.02x; concat is memcpy-bound,
    torch already near-optimal); 0.22x regression at fp32 Dk=14336 from large-tile
    pressure.
  scope: fp32/bf16, Dk 4096..14336, on H200. Use torch.cat for concat.
evidence_basis: H200 benchmark replay on 2026-07-21. The original harness completed
  on H200 with return code 0; its raw stdout/stderr is in the replay archive. This
  remains a local derived implementation, not an upstream-source performance claim.
---
> **Evidence boundary.** This is a locally derived canonical implementation, not a captured upstream source file. Any H200 latency/correctness text below comes from a self-reported run summary; retain it for hypothesis generation, but replay it with raw benchmark output before treating it as benchmark evidence.


## Overview

Fused concatenation of 3 tensors `[M, Dk]` along the last dim into `[M, 3*Dk]`
(one program per row: copy a/b/c row slices into the output row). Common for
QKV-projection output `[Q;K;V]` concat or attention head concat.

```python
@triton.jit
def concat3(a_ptr,b_ptr,c_ptr,o_ptr, M, Dk, D, BLOCK: tl.constexpr):
    row=tl.program_id(0)
    j=tl.arange(0, BLOCK); mask=j<Dk
    ar=tl.load(a_ptr+row*Dk+j, mask=mask); br=tl.load(b_ptr+row*Dk+j, mask=mask); cr=tl.load(c_ptr+row*Dk+j, mask=mask)
    tl.store(o_ptr+row*D + j,       ar, mask=mask)   # out[m, 0:Dk)
    tl.store(o_ptr+row*D + (Dk+j),  br, mask=mask)   # out[m, Dk:2Dk)
    tl.store(o_ptr+row*D + (2*Dk+j),cr, mask=mask)   # out[m, 2Dk:3Dk)
```

## Purpose: SPEEDUP (record-negative)
Concat is **memcpy-bound** and `torch.cat` is already near-optimal, so the fused
Triton kernel is only **~parity** (0.88x-1.02x). At fp32 Dk=14336 the
one-program-per-row BLOCK=16384 tile hits register/shared pressure (0.22x).
**Use torch.cat for concat**; the value here is the validated, self-contained
impl. (Note: an initial flat-index version had a stride bug — output must use the
`3*Dk` row stride, not `Dk`; the per-row version is bit-correct.)
[`data/crawl-runs/h200/op-concat-h200-results.md`](../../data/crawl-runs/h200/op-concat-h200-results.md).

## H200 measured

| M x Dk | fp32 torch/Triton | bf16 torch/Triton |
|---|--:|--:|
| 4096x4096 | 0.95x | 0.88x |
| 8192x4096 | 0.98x | 0.95x |
| 8192x8192 | 0.99x | 0.98x |
| 8192x14336 | 0.22x | 1.01x |
| 16384x14336 | 0.22x | 1.02x |


## H200 benchmark replay (2026-07-21)

Original harness: `op_concat.py`. The original harness completed on H200 with return code 0; its raw stdout/stderr is in the replay archive. Evidence: [`replay-2026-07-21-all-experimental.md`](../../data/crawl-runs/h200/replay-2026-07-21-all-experimental.md). All speed ratios are shape- and reference-specific.
