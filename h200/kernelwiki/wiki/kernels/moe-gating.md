---
id: kernel-moe-gating
title: MoE Top-K Gating (fused top-k + softmax, Hopper / H200)
type: kernel
architectures:
- sm90
tags:
- moe
- moe-gating
- softmax
- topk
- kernel-fusion
confidence: experimental
reproducibility: benchmarked
kernel_types:
- moe
- fused-kernel
languages:
- triton
- python
related:
- kernel-topk
- kernel-online-softmax
- kernel-grouped-gemm-hopper
sources:
- doc-nvidia-tuning-guide
performance_claims: []
blackwell_relevance: Same MoE gating on SM100; expert router (DeepSeek/Mixtral, k=2).
operator_purpose: speedup
what_it_does: 'MoE top-k gating: per-token top-k experts + softmax over winners (fused)
  -> routing weights+indices.'
h200_validation:
  date: 2026-07-21 (phase2; self-reported summary)
  evidence_file: data/crawl-runs/h200/op-moe-gate-h200-results.md
  harness_dir: artifacts/kernels/moe-gating/variants
  gpu: H200
  arch: sm90
  toolchain: Triton 3.6.0, PyTorch 2.11.0+cu130
  correctness: PASS — routing weights (werr 1e-7) and expert index set (set_match=1.0)
    match torch.
  result: 1.29x-3.69x faster than torch (topk + softmax), more so at higher expert
    count.
  scope: fp32, K=2, experts 64..256, on H200.
evidence_basis: H200 benchmark replay on 2026-07-21. The original harness completed
  on H200 with return code 0; its raw stdout/stderr is in the replay archive. This
  remains a local derived implementation, not an upstream-source performance claim.
---
> **Evidence boundary.** This is a locally derived canonical implementation, not a captured upstream source file. Any H200 latency/correctness text below comes from a self-reported run summary; retain it for hypothesis generation, but replay it with raw benchmark output before treating it as benchmark evidence.


## Overview

MoE expert router: for each token, take the top-k expert logits, softmax over the
k winners, and emit routing weights + expert indices. Fuses top-k selection and
softmax over the k winners into one kernel (vs torch `logits.topk(k)` + `softmax`).

```python
@triton.jit
def moe_gate(logits_ptr, w_ptr, idx_ptr, M, E, K, BLOCK_E):
    row=tl.program_id(0)
    x=tl.load(logits_ptr+row*E+offs, ...).to(tl.float32)
    for k in tl.static_range(K):                       # top-k via max+argmax+mask
        m=tl.max(x,0); sel=x==m
        idx=tl.argmax(tl.where(sel, offs, -1e30), 0)
        tl.store(idx_ptr+row*K+k, idx); tl.store(w_ptr+row*K+k, m); x=tl.where(sel,-1e30,x)
    kw=tl.load(w_ptr+row*K+tl.arange(0,K)); e=tl.exp(kw-tl.max(kw,0))   # softmax over winners
    tl.store(w_ptr+row*K+tl.arange(0,K), e/tl.sum(e,0))
```

## Purpose: SPEEDUP (fusion)
One fused kernel for top-k + softmax (vs torch's two ops). 1.29x-3.69x faster,
with bigger gains at higher expert counts (the top-k scan is the same cost but
torch's two-op overhead grows).
[`data/crawl-runs/h200/op-moe-gate-h200-results.md`](../../data/crawl-runs/h200/op-moe-gate-h200-results.md).

## H200 measured (K=2)

| M x E | torch/Triton |
|---|--:|
| 4096x64 | 1.29x |
| 8192x128 | 2.23x |
| 8192x256 | 3.69x |
| 8192x64 | 1.47x |
| 16384x128 | 2.81x |


## H200 benchmark replay (2026-07-21)

Original harness: `op_moe_gate.py`. The original harness completed on H200 with return code 0; its raw stdout/stderr is in the replay archive. Evidence: [`replay-2026-07-21-all-experimental.md`](../../data/crawl-runs/h200/replay-2026-07-21-all-experimental.md). All speed ratios are shape- and reference-specific.
