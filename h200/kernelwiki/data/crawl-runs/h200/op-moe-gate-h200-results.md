# MoE Top-K Gating (Triton) on H200
Date: 2026-07-21 (phase2). Fused per-token top-k experts + softmax over winners
(K=2) vs torch (logits.topk(k)+softmax). fp32. H200, Triton 3.6.0.

## Purpose: SPEEDUP (fused gating).

## Correctness — PASS (weights werr 1e-7; expert index set_match=1.0).

## Latency (K=2)

| M x E | torch/Triton |
|---|--:|
| 4096x64 | 1.29x |
| 8192x128 | 2.23x |
| 8192x256 | 3.69x |
| 8192x64 | 1.47x |
| 16384x128 | 2.81x |

## File
moe_gate_h200.py — sha256 2666f7f439694287e5c7625516a71986f2e9a638358ccde8fdbcd62deba4180e
