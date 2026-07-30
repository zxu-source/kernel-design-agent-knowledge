---
id: kernel-fused-add-rmsnorm
title: Fused Add + RMSNorm (Hopper / H200)
type: kernel
architectures:
- sm90
tags:
- rmsnorm
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
- kernel-silu-and-mul
sources:
- doc-nvidia-tuning-guide
performance_claims: []
blackwell_relevance: Same fused pattern on SM100; standard in vLLM/sglang/flashinfer
  residual streams.
operator_purpose: speedup
what_it_does: 'Fused (residual+x)+RMSNorm: residual_out=residual+x, out=rmsnorm(residual_out)
  in one kernel (2 reads, 2 writes).'
h200_validation:
  date: 2026-07-21 (phase2; self-reported summary)
  evidence_file: data/crawl-runs/h200/op-fused-add-rmsnorm-h200-results.md
  harness_dir: artifacts/kernels/fused-add-rmsnorm/variants
  gpu: H200
  arch: sm90
  toolchain: Triton 3.6.0, PyTorch 2.11.0+cu130
  correctness: PASS — out err bf16 ~3.1e-2 (dtype precision); residual_out bit-identical.
  result: Fused kernel 1.10x-1.45x faster than torch (add + rms_norm + copy).
  scope: bf16, hidden 4096..14336, on H200.
evidence_basis: H200 benchmark replay on 2026-07-21. The original harness completed
  on H200 with return code 0; its raw stdout/stderr is in the replay archive. This
  remains a local derived implementation, not an upstream-source performance claim.
---
> **Evidence boundary.** This is a locally derived canonical implementation, not a captured upstream source file. Any H200 latency/correctness text below comes from a self-reported run summary; retain it for hypothesis generation, but replay it with raw benchmark output before treating it as benchmark evidence.


## Overview

Standard LLM residual-stream kernel (vLLM `fused_add_rms_norm`,
flashinfer `FusedAddRMSNorm`). Adds the block output into the residual and
normalizes in one fused pass, writing both the summed residual (for the next
layer) and the normalized output.

```python
@triton.jit
def fused_add_rmsnorm(res_ptr, x_ptr, o_ptr, ro_ptr, w_ptr, eps, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N); mask = offs < N
    res = tl.load(res_ptr + row*N + offs, mask=mask, other=0.0).to(tl.float32)
    xx = tl.load(x_ptr + row*N + offs, mask=mask, other=0.0).to(tl.float32)
    s = res + xx
    var = tl.sum(s*s, axis=0) / N
    rrms = tl.rsqrt(var + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(ro_ptr + row*N + offs, s.to(ro_ptr.dtype.element_ty), mask=mask)            # residual_out
    tl.store(o_ptr  + row*N + offs, (s*rrms*w).to(o_ptr.dtype.element_ty), mask=mask)     # out
```

## Purpose: SPEEDUP
One kernel vs three torch ops: eliminates the add and the residual copy, halving
memory traffic (2 reads + 2 writes vs 3 + 3). Measured 1.10x-1.45x faster than
torch. Full numbers:
[`data/crawl-runs/h200/op-fused-add-rmsnorm-h200-results.md`](../../data/crawl-runs/h200/op-fused-add-rmsnorm-h200-results.md).

## Related
- [rmsnorm-hopper](rmsnorm-hopper.md) — standalone RMSNorm forward.


## H200 benchmark replay (2026-07-21)

Original harness: `op_fused_add_rmsnorm.py`. The original harness completed on H200 with return code 0; its raw stdout/stderr is in the replay archive. Evidence: [`replay-2026-07-21-all-experimental.md`](../../data/crawl-runs/h200/replay-2026-07-21-all-experimental.md). All speed ratios are shape- and reference-specific.
