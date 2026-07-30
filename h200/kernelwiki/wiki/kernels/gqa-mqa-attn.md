---
id: kernel-gqa-mqa-attn
title: Grouped / Multi-Query Attention (GQA/MQA, Hopper / H200)
type: kernel
architectures:
- sm90
tags:
- attention
- flash-attention
confidence: experimental
reproducibility: benchmarked
kernel_types:
- attention
- fused-kernel
languages:
- triton
- python
related:
- kernel-triton-fa2-hopper
- kernel-flashmla
sources:
- doc-nvidia-tuning-guide
performance_claims: []
blackwell_relevance: Same GQA/MQA pattern on SM100; LLM-typical (Llama/Qwen use Hq:Hkv=4:1..8:1).
operator_purpose: speedup
what_it_does: 'Grouped/Multi-Query Attention forward: Hq query heads share Hkv KV
  groups (head_kv=head_q//n_rep), no KV replication.'
h200_validation:
  date: 2026-07-21 (phase2; self-reported summary)
  evidence_file: data/crawl-runs/h200/op-gqa-mqa-attn-h200-results.md
  harness_dir: artifacts/kernels/gqa-mqa-attn/variants
  gpu: H200
  arch: sm90
  toolchain: Triton 3.6.0, PyTorch 2.11.0+cu130
  correctness: PASS — err 3.8e-6..7.6e-6 vs torch SDPA with repeat_interleave'd K,V.
  result: '0.27x-0.47x torch SDPA (backend-wins: naive Triton GQA loses to torch''s
    optimized attention; GQA head-group mapping is correct, avoids KV replication).'
  scope: fp16, LLM-typical GQA ratios (Hq:Hkv 32:8, 16:8) and MQA (8:1), on H200.
evidence_basis: H200 benchmark replay on 2026-07-21. The original harness completed
  on H200 with return code 0; its raw stdout/stderr is in the replay archive. This
  remains a local derived implementation, not an upstream-source performance claim.
---
> **Evidence boundary.** This is a locally derived canonical implementation, not a captured upstream source file. Any H200 latency/correctness text below comes from a self-reported run summary; retain it for hypothesis generation, but replay it with raw benchmark output before treating it as benchmark evidence.


## Overview

Grouped-Query Attention (GQA) / Multi-Query Attention (MQA): `Hq` query heads
share `Hkv` (Hkv < Hq) KV-head groups to cut KV-cache memory/bandwidth. Each
query head maps to its KV group `head_kv = head_q // n_rep` where
`n_rep = Hq/Hkv`, so the kernel reads each KV group `n_rep` times with no KV
replication in memory.

```python
# per program: Q head hq -> KV group hkv = hq // NREP
Kb = tl.make_block_ptr(base=K + b*sKB + hkv*sKH, shape=(HEAD_DIM,N), ...)
Vb = tl.make_block_ptr(base=V + b*sKB + hkv*sKH, shape=(N,HEAD_DIM), ...)
k = tl.load(Kb, boundary_check=(1,)); v = tl.load(Vb, boundary_check=(0,))
qk = tl.dot(q, k) * sm_scale                  # one KV group serves n_rep Q heads
acc += tl.dot(softmax(qk), v)
```

## Purpose: SPEEDUP (in principle)
GQA avoids expanding K/V (n_rep-fold), cutting KV-cache memory and attention
read traffic. However, the naive Triton GQA kernel measured **0.27x-0.47x**
torch SDPA (torch's optimized FA3/cuDNN backend wins) — the same backend-wins
pattern as the non-causal FA-2. The GQA head-group mapping is validated correct;
throughput is NOT representative of a production GQA kernel (FlashInfer /
flash-attention GQA paths). Full numbers:
[`data/crawl-runs/h200/op-gqa-mqa-attn-h200-results.md`](../../data/crawl-runs/h200/op-gqa-mqa-attn-h200-results.md).

## H200 measured

| config | sdpa/Triton |
|---|--:|
| 1x32 x8 x8192x128 (GQA) | 0.28x |
| 2x32 x8 x4096x128 (GQA) | 0.28x |
| 1x32 x8 x16384x128 (GQA) | 0.28x |
| 1x16 x8 x8192x64 (GQA) | 0.47x |
| 1x8 x1 x4096x128 (MQA) | 0.29x |


## H200 benchmark replay (2026-07-21)

Original harness: `op_gqa_attn.py`. The original harness completed on H200 with return code 0; its raw stdout/stderr is in the replay archive. Evidence: [`replay-2026-07-21-all-experimental.md`](../../data/crawl-runs/h200/replay-2026-07-21-all-experimental.md). All speed ratios are shape- and reference-specific.
