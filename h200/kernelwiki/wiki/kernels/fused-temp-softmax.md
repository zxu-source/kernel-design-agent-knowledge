---
id: kernel-fused-temp-softmax
title: Fused Temperature + Softmax (Hopper / H200)
type: kernel
architectures:
- sm90
tags:
- softmax
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
sources:
- doc-nvidia-tuning-guide
performance_claims: []
blackwell_relevance: Same fused pattern on SM100; standard in LLM sampling (sglang/vLLM).
operator_purpose: both
what_it_does: 'Fused temperature+softmax: y=softmax(x/temp) one pass, -inf-mask +
  overflow-safe (max-subtract).'
h200_validation:
  date: 2026-07-21 (phase2; self-reported summary)
  evidence_file: data/crawl-runs/h200/op-fused-temp-softmax-h200-results.md
  harness_dir: artifacts/kernels/fused-temp-softmax/variants
  gpu: H200
  arch: sm90
  toolchain: Triton 3.6.0, PyTorch 2.11.0+cu130
  correctness: PASS — incl. -inf-masked positions (err 1e-6..9.8e-4 vs torch).
  result: Fused 2.0x-5.5x faster than torch (x/temp then softmax).
  scope: bf16, sampling shapes (vocab up to 32K), temps 0.7/1.0/1.5, on H200.
evidence_basis: H200 benchmark replay on 2026-07-21. The original harness completed
  on H200 with return code 0; its raw stdout/stderr is in the replay archive. This
  remains a local derived implementation, not an upstream-source performance claim.
---
> **Evidence boundary.** This is a locally derived canonical implementation, not a captured upstream source file. Any H200 latency/correctness text below comes from a self-reported run summary; retain it for hypothesis generation, but replay it with raw benchmark output before treating it as benchmark evidence.


## Overview

LLM sampling kernel: `y = softmax(x / temperature)` in one fused pass. Combines
the temperature scale with the online (max-subtract) softmax, so `-inf`-masked
positions are handled correctly (they contribute 0) and large logits do not
overflow `exp`.

```python
@triton.jit
def fused_temp_softmax(x_ptr, o_ptr, inv_temp, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N); mask = offs < N
    x = tl.load(x_ptr + row*N + offs, mask=mask, other=-float('inf')).to(tl.float32)
    x = x * inv_temp
    m = tl.max(x, axis=0)
    e = tl.exp(x - m); e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    tl.store(o_ptr + row*N + offs, (e / s).to(o_ptr.dtype.element_ty), mask=mask)
```

## Purpose: BOTH
- **speedup**: scale + softmax fused (vs torch `x/temp` then `softmax` = a scale
  pass + a softmax pass + an intermediate). 2.0x-5.5x faster than torch.
- **robustness**: `-inf`-masked positions handled; max-subtraction prevents `exp`
  overflow. Correctness verified with ~10% of positions set to `-inf`.

[`data/crawl-runs/h200/op-fused-temp-softmax-h200-results.md`](../../data/crawl-runs/h200/op-fused-temp-softmax-h200-results.md).

## Related
- [online-softmax](online-softmax.md) — standalone online softmax.


## H200 benchmark replay (2026-07-21)

Original harness: `op_fused_temp_softmax.py`. The original harness completed on H200 with return code 0; its raw stdout/stderr is in the replay archive. Evidence: [`replay-2026-07-21-all-experimental.md`](../../data/crawl-runs/h200/replay-2026-07-21-all-experimental.md). All speed ratios are shape- and reference-specific.
