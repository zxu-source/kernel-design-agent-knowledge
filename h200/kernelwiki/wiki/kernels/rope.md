---
id: kernel-rope
title: Rotary Position Embedding (RoPE, Hopper / H200)
type: kernel
architectures:
- sm90
tags:
- attention
- kernel-fusion
confidence: experimental
reproducibility: benchmarked
kernel_types:
- fused-kernel
languages:
- triton
- python
related:
- kernel-online-softmax
- kernel-triton-fa2-hopper
sources:
- doc-nvidia-tuning-guide
performance_claims: []
blackwell_relevance: Same RoPE on SM100; Llama/Qwen/Mistral position embedding.
operator_purpose: speedup
what_it_does: 'RoPE (rotary position embedding, Llama rotate-half): q*cos - partner*sin
  / q*cos + partner*sin.'
h200_validation:
  date: 2026-07-21 (phase2; self-reported summary)
  evidence_file: data/crawl-runs/h200/op-rope-h200-results.md
  harness_dir: artifacts/kernels/rope/variants
  gpu: H200
  arch: sm90
  toolchain: Triton 3.6.0, PyTorch 2.11.0+cu130
  correctness: PASS — err 4.9e-4 vs torch (fp16).
  result: 1.66x-5.61x faster than torch (fused rotate-half vs cat/slice); D=256 gains
    most.
  scope: fp16, LLM shapes (B,H,S,D), D 64..256, on H200.
evidence_basis: H200 benchmark replay on 2026-07-21. The original harness completed
  on H200 with return code 0; its raw stdout/stderr is in the replay archive. This
  remains a local derived implementation, not an upstream-source performance claim.
---
> **Evidence boundary.** This is a locally derived canonical implementation, not a captured upstream source file. Any H200 latency/correctness text below comes from a self-reported run summary; retain it for hypothesis generation, but replay it with raw benchmark output before treating it as benchmark evidence.


## Overview

Rotary Position Embedding (Llama-style rotate-half): for each position, split the
head dim into halves and rotate. `q_out[..., :D/2] = q[..., :D/2]*cos -
q[..., D/2:]*sin; q_out[..., D/2:] = q[..., D/2:]*cos + q[..., :D/2]*sin`.

```python
@triton.jit
def rope_fwd(q_ptr, cos_ptr, sin_ptr, o_ptr, NH, S, D, BLOCK_D):
    row=tl.program_id(0)                      # one (b,h,s) row
    d=tl.arange(0, BLOCK_D); mask=d<D; half=D//2
    q=tl.load(q_ptr+row*D+d, mask=mask).to(tl.float32)
    s=row%S
    c=tl.load(cos_ptr+s*half+(d%half), mask=mask).to(tl.float32)
    sn=tl.load(sin_ptr+s*half+(d%half), mask=mask).to(tl.float32)
    lo=d<half; partner=tl.where(lo, d+half, d-half)            # always in [0,D)
    qp=tl.load(q_ptr+row*D+partner, mask=mask).to(tl.float32)
    out=tl.where(lo, q*c - qp*sn, q*c + qp*sn)
    tl.store(o_ptr+row*D+d, out.to(o_ptr.dtype.element_ty), mask=mask)
```

## Purpose: SPEEDUP (fusion)
Fused rotate-half vs torch's cat/slice chain. 1.66x-5.61x faster (D=256 gains
most). **Robustness note**: the partner index `where(d<half, d+half, d-half)` is
always in [0,D); an earlier `tl.where` with two separate `d±half` loads went OOB
(both branches' loads execute) — compute the index first, load once.
[`data/crawl-runs/h200/op-rope-h200-results.md`](../../data/crawl-runs/h200/op-rope-h200-results.md).

## H200 measured

| BxHxSxD | torch/Triton |
|---|--:|
| 1x32x4096x128 | 2.99x |
| 2x32x2048x128 | 3.01x |
| 1x32x8192x128 | 3.07x |
| 1x8x4096x64 | 1.66x |
| 1x32x4096x256 | 5.61x |


## H200 benchmark replay (2026-07-21)

Original harness: `op_fused_ln_gelu_rmsnorm_rope.py`. The original harness completed on H200 with return code 0; its raw stdout/stderr is in the replay archive. Evidence: [`replay-2026-07-21-all-experimental.md`](../../data/crawl-runs/h200/replay-2026-07-21-all-experimental.md). All speed ratios are shape- and reference-specific.
