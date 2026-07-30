# GQA/MQA Attention (Triton) on H200
Date: 2026-07-21 (phase2). GQA/MQA FA-2 forward (head_kv=head_q//n_rep, no KV
replication) vs torch SDPA with repeat_interleave'd K,V. fp16. H200, Triton 3.6.

## Purpose: SPEEDUP (no KV replication — in principle; naive kernel loses to backend).

## Correctness — PASS (err 3.8e-6..7.6e-6).

## Latency — 0.27x-0.47x torch SDPA (backend-wins)

| BxHq xHkv x M x D | sdpa/Triton |
|---|--:|
| 1x32 x8 x8192x128 | 0.28x |
| 2x32 x8 x4096x128 | 0.28x |
| 1x32 x8 x16384x128 | 0.28x |
| 1x16 x8 x8192x64 | 0.47x |
| 1x8 x1(MQA) x4096x128 | 0.29x |

GQA head-group mapping correct; naive Triton kernel ~3.5x slower than torch
optimized SDPA backend. Production GQA needs FlashInfer/flash-attention.

## File
gqa_mqa_attn_h200.py — sha256 10f0b85cadc3d5d3345bc686eeb3c7696ad5e107db21ce1154e8db15034f55b4
