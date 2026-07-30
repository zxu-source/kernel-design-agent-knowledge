---
id: hw-pdl-gdc
title: "Programmatic Dependent Launch / Grid Dependency Control"
type: hardware
architectures: [sm100, sm100a, sm90]
tags: [pdl, gdc]
confidence: verified
related: [technique-persistent-kernels, hw-clc]
sources: [doc-nvidia-tuning-guide, pr-cutlass-2161, doc-cutlass-changelog-sm100, pr-triton-6394]
aliases: [PDL, GDC, "programmatic dependent launch", "grid dependency control"]
blackwell_relevance: "PDL available on Hopper but enabled by default on Blackwell SM100."
h200_validation:
  date: '2026-07-20'
  source_id: pr-triton-6394
  evidence_file: data/crawl-runs/h200/triton6394-pdl-h200-results.md
  harness_dir: artifacts/kernels/pdl-dependent-launch/variants
  gpu: H200
  arch: sm90
  toolchain: "Triton 3.6.0, PyTorch 2.11.0+cu130"
  correctness: "PASS — PDL chain identical to non-PDL chain and to closed form; max delta = 0.0"
  result: "1.07x-1.18x wall-clock speedup for short back-to-back dependent kernels under CUDA-graph replay (best 1.178x); per-launch GPU time ~1.9us -> ~1.7us. ~1.0x under Python dispatch (CPU-bound)."
  scope: "Single elementwise chain kernel class on H200; benefit requires graph capture or otherwise CPU-overhead-free dispatch."
evidence_basis:
- evidence_type: official-doc
  source_id: doc-nvidia-tuning-guide
  description: >
    NVIDIA tuning guide documents PDL/GDC: the primary grid signals near
    completion via griddepcontrol.launch_dependents and the dependent grid
    waits via cudaGridDependencySynchronize / griddepcontrol.wait, overlapping
    ramp-down and ramp-up of consecutive grids. Opt-in on SM90, default on SM100.
- evidence_type: upstream-code
  source_id: pr-triton-6394
  description: >
    Triton PR #6394 implements the GDC intrinsics (tl.extra.cuda.gdc_wait /
    gdc_launch_dependents) and the launch_pdl compile option that sets
    CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION. CUTLASS PR #2161
    adds the equivalent launch_dependent_grids / wait_on_dependent_grids calls
    to SM100 GEMM kernels.
- evidence_type: h200-measured
  source_id: pr-triton-6394
  description: >
    H200 (SM90) measurement on 2026-07-20 with Triton 3.6.0 (see h200_validation
    above): short back-to-back dependent kernels saw 1.07x-1.18x wall-clock
    speedup under CUDA-graph replay (per-launch GPU time ~1.9us -> ~1.7us),
    correctness delta = 0.0. Benefit masked (~1.0x) under Python dispatch.
---

## Overview

PDL/GDC allows overlapping execution of dependent kernel launches. The primary kernel signals it is finishing; the secondary kernel begins before the primary fully completes.

## How It Works

```cuda
// Primary kernel signals near completion
cudaGridDependencySynchronize();  // or PTX equivalent

// Secondary kernel can start overlapping with primary's tail
// Enabled by default on SM100 (opt-in on SM90)
```

## Blackwell Default Behavior

On SM100, PDL is **enabled by default** — no opt-in needed. This means:
- Back-to-back kernel launches naturally overlap
- Memory fences ensure correctness for dependent data
- Reduces kernel launch gaps in compute-heavy pipelines

## When It Matters
- Chains of small kernels (e.g., MoE dispatch → compute → combine)
- Pipeline-parallel training with many sequential kernel launches
- Reduces overall wall-clock time without code changes on Blackwell

## Related
- [persistent-kernels](../techniques/persistent-kernels.md) — Alternative approach to reducing launch overhead
- [clc](clc.md) — Dynamic scheduling within persistent kernels

## H200 measured evidence (Triton, SM90)

PDL was validated on an H200 (132 SMs, cc 9.0) with Triton 3.6.0. The
Triton 3.6.0 enablement knob is the **`launch_pdl=True` compile/launch option**
(not `enable_pdl`, which is not a kwarg in this build). Passing `launch_pdl=True`
sets `metadata.launch_pdl`, which makes the backend driver set
`CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION` on the launch. The GDC
intrinsics `tl.extra.cuda.gdc_wait()` and `tl.extra.cuda.gdc_launch_dependents()`
are then emitted as `griddepcontrol.wait` / `griddepcontrol.launch_dependents`
in PTX (verified by PTX dump).

```python
@triton.jit
def chain_step(x_ptr, n, alpha, beta, BLOCK: tl.constexpr, PDL: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK); m = offs < n
    if PDL:
        tl.extra.cuda.gdc_wait()                  # dependent waits for prior grid
    x = tl.load(x_ptr + offs, mask=m)
    tl.store(x_ptr + offs, x * alpha + beta, mask=m)
    if PDL:
        tl.extra.cuda.gdc_launch_dependents()     # signal next grid may ramp up
# launch with: chain_step[grid](x, n, a, b, BLOCK=BLOCK, PDL=True, launch_pdl=True)
```

**Correctness**: a chain of N data-dependent kernels produced output identical
to the non-PDL chain and to the closed-form `beta*(alpha^N - 1)/(alpha - 1)`
(max abs delta = 0.0 across all tested configs).

**Speedup**: under CUDA-graph replay (which removes CPU dispatch overhead and
isolates the GPU-side overlap), PDL reduced per-launch GPU time from ~1.9 us to
~1.7 us for short back-to-back kernels — a **1.07x–1.18x** wall-clock speedup
(best 1.178x at BLOCK=2048, N=1024), growing with BLOCK size. Under naive
Python dispatch the GPU is already idle between launches, so PDL measured ~1.0x;
**graph capture (or otherwise CPU-overhead-free dispatch) is required to observe
the gain.** Full numbers and harnesses:
[`data/crawl-runs/h200/triton6394-pdl-h200-results.md`](../../data/crawl-runs/h200/triton6394-pdl-h200-results.md).
