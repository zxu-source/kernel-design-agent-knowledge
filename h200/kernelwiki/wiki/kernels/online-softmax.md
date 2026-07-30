---
id: kernel-online-softmax
title: Online Softmax Forward (Hopper / H200)
type: kernel
architectures:
- sm90
tags:
- softmax
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
- kernel-rmsnorm-hopper
sources:
- doc-nvidia-tuning-guide
performance_claims: []
blackwell_relevance: Same fused online softmax on SM100; tile for huge N to avoid
  register spill.
operator_purpose: both
what_it_does: 'Online softmax: max-subtract (stability), exp, sum, divide in one fused
  pass per row.'
h200_validation:
  date: 2026-07-21 (phase2; self-reported summary)
  evidence_file: data/crawl-runs/h200/op-softmax-h200-results.md
  harness_dir: artifacts/kernels/online-softmax/variants
  gpu: H200
  arch: sm90
  toolchain: Triton 3.6.0, PyTorch 2.11.0+cu130
  correctness: PASS — 1e-6..1e-5 vs torch.softmax.
  result: 1.2x-3.4x FASTER than torch at moderate N (4K..32K); DEGRADES to 0.23x at
    N=128K (BLOCK_N=131072 spills registers).
  scope: One-program-per-row online softmax; common hidden/vocab sizes. For huge N
    use a tiled (flash-style) softmax.
evidence_basis: H200 benchmark replay on 2026-07-21. The original harness completed
  on H200 with return code 0; its raw stdout/stderr is in the replay archive. This
  remains a local derived implementation, not an upstream-source performance claim.
---
> **Evidence boundary.** This is a locally derived canonical implementation, not a captured upstream source file. Any H200 latency/correctness text below comes from a self-reported run summary; retain it for hypothesis generation, but replay it with raw benchmark output before treating it as benchmark evidence.

## H200 replay evidence (2026-07-21)

The compact replay passed correctness (`err` 1.91e-6–7.63e-6). It measured
`torch_ms / triton_ms` of 0.62x, 1.98x, and 1.28x for 1024², 4096², and
4096x32000; this is not a universal speedup. Raw stdout: [`replay-2026-07-21-three-operators-raw.md`](../../data/crawl-runs/h200/replay-2026-07-21-three-operators-raw.md).


## Overview

Online softmax: subtract the row max (numerical stability — prevents `exp`
overflow), exponentiate, sum, and divide in one fused pass per row.

```python
@triton.jit
def softmax_fwd(x_ptr, o_ptr, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N); mask = offs < N
    x = tl.load(x_ptr + row*N + offs, mask=mask, other=-float('inf')).to(tl.float32)
    m = tl.max(x, axis=0)                      # stability
    e = tl.exp(x - m); e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    tl.store(o_ptr + row*N + offs, (e / s).to(o_ptr.dtype.element_ty), mask=mask)
```

## Purpose: BOTH
- **robustness**: max-subtraction prevents `exp(x)` overflow for large logits.
- **speedup**: one fused pass vs torch's multi-op softmax.

## H200 measured evidence

Correctness PASS (1e-6..1e-5 vs torch). **1.2x-3.4x faster** at moderate N
(4096..32000). **Caveat**: at huge N (128256, e.g. a 128K vocab) the naive
one-program-per-row uses `BLOCK_N = next_pow2(128256) = 131072`, which spills
registers and is **0.23x** (4.3x slower) than torch. For huge N, use a **tiled
/ multi-pass (flash-style) softmax** that streams the row in tiles with an
online running max+sum. Full numbers:
[`data/crawl-runs/h200/op-softmax-h200-results.md`](../../data/crawl-runs/h200/op-softmax-h200-results.md).

## Related
- [layernorm-hopper](layernorm-hopper.md) / [rmsnorm-hopper](rmsnorm-hopper.md) — other row-reduction fused norms.


## H200 benchmark replay (2026-07-21)

Original harness: `op_softmax.py`. The original harness completed on H200 with return code 0; its raw stdout/stderr is in the replay archive. Evidence: [`replay-2026-07-21-all-experimental.md`](../../data/crawl-runs/h200/replay-2026-07-21-all-experimental.md). All speed ratios are shape- and reference-specific.
