---
id: kernel-adam-step
title: Fused AdamW Optimizer Step (Hopper / H200)
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
- kernel-fused-add-rmsnorm
sources:
- doc-nvidia-tuning-guide
performance_claims: []
blackwell_relevance: Same fused AdamW on SM100; training optimizer step.
operator_purpose: speedup
what_it_does: 'Fused AdamW optimizer step: m,v,param update in one kernel (3R+3W vs
  ~8 separate torch ops).'
h200_validation:
  date: 2026-07-21 (phase2; self-reported summary)
  evidence_file: data/crawl-runs/h200/op-adam-step-h200-results.md
  harness_dir: artifacts/kernels/adam-step/variants
  gpu: H200
  arch: sm90
  toolchain: Triton 3.6.0, PyTorch 2.11.0+cu130
  correctness: PASS — fp32 ~3e-8, bf16 ~2e-4 vs torch separate-op AdamW.
  result: '2.56x-3.05x faster than torch (fused: 3R+3W in one kernel vs ~8 separate
    ops + intermediates).'
  scope: fp32/bf16, params up to 33M elements, on H200.
evidence_basis: H200 benchmark replay on 2026-07-21. The original harness completed
  on H200 with return code 0; its raw stdout/stderr is in the replay archive. This
  remains a local derived implementation, not an upstream-source performance claim.
---
> **Evidence boundary.** This is a locally derived canonical implementation, not a captured upstream source file. Any H200 latency/correctness text below comes from a self-reported run summary; retain it for hypothesis generation, but replay it with raw benchmark output before treating it as benchmark evidence.


## Overview

Fused AdamW optimizer step: update first/second moment estimates (`m`, `v`),
bias-correct, apply weight decay, and update parameters — all in one elementwise
kernel (3 reads: m, v, g + param; 3 writes: m, v, param). vs torch's ~8 separate
ops (each a launch + intermediate buffer).

```python
@triton.jit
def adam_w(p_ptr, g_ptr, m_ptr, v_ptr, lr, beta1, beta2, eps, wd, bias_c1, bias_c2, total, BLOCK):
    pid=tl.program_id(0); offs=pid*BLOCK+tl.arange(0,BLOCK); mask=offs<total
    p=tl.load(p_ptr+offs,mask=mask).to(tl.float32); g=tl.load(g_ptr+offs,mask=mask).to(tl.float32)
    m=tl.load(m_ptr+offs,mask=mask).to(tl.float32); v=tl.load(v_ptr+offs,mask=mask).to(tl.float32)
    m = beta1*m + (1.0-beta1)*g; v = beta2*v + (1.0-beta2)*g*g
    m_hat = m * bias_c1; v_hat = v * bias_c2
    p_new = p - lr*(m_hat / (tl.sqrt(v_hat) + eps) + wd*p)
    tl.store(m_ptr+offs, m.to(m_ptr.dtype.element_ty), mask=mask)
    tl.store(v_ptr+offs, v.to(v_ptr.dtype.element_ty), mask=mask)
    tl.store(p_ptr+offs, p_new.to(p_ptr.dtype.element_ty), mask=mask)
```

## Purpose: SPEEDUP (fusion)
2.56x-3.05x faster than torch: the fused kernel reads/writes each tensor once,
vs torch's ~8 separate elementwise ops (each a kernel launch + intermediate
buffer). The fusion is most valuable at large param counts (33M: 3.04x).
[`data/crawl-runs/h200/op-adam-step-h200-results.md`](../../data/crawl-runs/h200/op-adam-step-h200-results.md).

## H200 measured

| N (params) | fp32 torch/Triton | bf16 torch/Triton |
|--:|--:|--:|
| 1M | 2.56x | 2.58x |
| 4M | 2.88x | 2.85x |
| 16M | 3.05x | 3.05x |
| 33M | 3.04x | 3.04x |


## H200 benchmark replay (2026-07-21)

Original harness: `op_adam_step.py`. The original harness completed on H200 with return code 0; its raw stdout/stderr is in the replay archive. Evidence: [`replay-2026-07-21-all-experimental.md`](../../data/crawl-runs/h200/replay-2026-07-21-all-experimental.md). All speed ratios are shape- and reference-specific.
