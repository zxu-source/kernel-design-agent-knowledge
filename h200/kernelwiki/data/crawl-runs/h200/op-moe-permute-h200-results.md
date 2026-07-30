# MoE Permute/Unpermute (Triton) on H200
Date: 2026-07-21 (phase2). Gather-permute + scatter-unpermute vs torch fancy
indexing. H200, Triton 3.6.0. CUDA events min of 50.

## Purpose: characterization (dispatch building block; record-negative vs torch indexing).

## Correctness — PASS (bit-identical gather + scatter).

## Latency (gather/permute) — ~parity (0.83x-1.06x)

| M x D | fp32 torch/Triton | bf16 torch/Triton |
|---|--:|--:|
| 4096x4096 | 0.90x | 0.83x |
| 8192x8192 | 1.01x | 0.97x |
| 8192x11008 | 1.04x | 1.03x |
| 8192x14336 | 1.03x | 1.02x |
| 16384x14336 | 1.06x | 1.05x |

Gather/scatter is memcpy-bound; torch indexing near-optimal -> ~parity.

## File
moe_permute_h200.py — sha256 b904552e967f8b6cf717c560d0cd58c43de49740465613d81bf077c46425e33f
