---
id: kernel-layernorm-bwd
title: LayerNorm Backward (Hopper / H200)
type: kernel
architectures:
- sm90
tags:
- layernorm
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
- kernel-layernorm-hopper
- kernel-rmsnorm-bwd
sources:
- doc-nvidia-tuning-guide
performance_claims: []
blackwell_relevance: Same LayerNorm backward on SM100; training gradient.
operator_purpose: speedup
what_it_does: 'LayerNorm backward: grad_x=rrms*(gy*w - mean(gy*w) - (x-m)*mean(gy*w*(x-m))*rrms^2);
  grad_w/grad_b atomic-add.'
h200_validation:
  date: 2026-07-21 (phase2; self-reported summary)
  evidence_file: data/crawl-runs/h200/op-layernorm-bwd-h200-results.md
  harness_dir: artifacts/kernels/layernorm-bwd/variants
  gpu: H200
  arch: sm90
  toolchain: Triton 3.6.0, PyTorch 2.11.0+cu130
  correctness: 'PASS at moderate N (grad_x err ~3e-6, grad_w ~1e-3). Large N: higher
    error (~1e-3) from one-pass variance precision.'
  result: Marginal speedup (1.06x-1.15x) at moderate N; 0.21x-0.26x at large N (BLOCK
    spill + 3 atomics + one-pass variance precision).
  scope: fp32, hidden 4096..14336, on H200. Weaker than RMSNorm-bwd (more reductions
    + atomics).
evidence_basis: H200 benchmark replay on 2026-07-21. The original harness completed
  on H200 with return code 0; its raw stdout/stderr is in the replay archive. This
  remains a local derived implementation, not an upstream-source performance claim.
---
> **Evidence boundary.** This is a locally derived canonical implementation, not a captured upstream source file. Any H200 latency/correctness text below comes from a self-reported run summary; retain it for hypothesis generation, but replay it with raw benchmark output before treating it as benchmark evidence.


## Overview

LayerNorm training gradient (one fused pass per row + atomic-add grad_w, grad_b):
`grad_x = rrms*(gy*w - mean(gy*w) - (x-mean)*mean(gy*w*(x-mean))*rrms^2)`.

```python
@triton.jit
def layernorm_bwd(...):
    ...
    mean=tl.sum(x,axis=0)/N; xm=x-mean; var=tl.sum(xm*xm,axis=0)/N; rrms=tl.rsqrt(var+eps)
    gyw=gy*w; c1=tl.sum(gyw,axis=0)/N; c2=tl.sum(gyw*xm,axis=0)/N
    gx=rrms*(gyw - c1 - xm*c2*rrms*rrms)
    tl.store(gx_ptr+..., gx, mask=mask)
    tl.atomic_add(gw_ptr+offs, gy*xm*rrms, mask=mask)   # grad_w
    tl.atomic_add(gb_ptr+offs, gy, mask=mask)           # grad_b
```

## Purpose: SPEEDUP (marginal)
1.06x-1.15x at moderate N (layernorm-bwd has more reductions than RMSNorm-bwd
and 3 atomics for grad_w/grad_b, so the fusion win is thin). 0.21x-0.26x at large
N: BLOCK=16384 spills, 3 atomic-adds add overhead, and the one-pass variance
(`E[x²]-E[x]²`) loses precision at large N (grad error ~1e-3). For large N, tile
the reduction and use a two-pass variance.
[`data/crawl-runs/h200/op-layernorm-bwd-h200-results.md`](../../data/crawl-runs/h200/op-layernorm-bwd-h200-results.md).

## H200 measured

| M x N | grad_x err | torch/Triton |
|---|--:|--:|
| 4096x4096 | 2.9e-6 | 1.15x |
| 8192x8192 | 2.9e-6 | 1.06x |
| 8192x11008 | 5.7e-3 | 0.21x |
| 8192x14336 | 1.1e-3 | 0.26x |
| 16384x14336 | 1.1e-3 | 0.26x |


## H200 benchmark replay (2026-07-21)

Original harness: `op_layernorm_bwd.py`. The original harness completed on H200 with return code 0; its raw stdout/stderr is in the replay archive. Evidence: [`replay-2026-07-21-all-experimental.md`](../../data/crawl-runs/h200/replay-2026-07-21-all-experimental.md). All speed ratios are shape- and reference-specific.
