# Fused Gated MLP (Triton) on H200
Date: 2026-07-21 (phase2). 1 GEMM@2N + silu_and_mul vs torch sequential
(silu(x@gate)*(x@up), 2 cuBLAS GEMMs). bf16. H200, Triton 3.6.0. CUDA events min 20.

## Purpose: SPEEDUP (record-negative vs torch 2-GEMM sequential).

## Correctness — OK (bf16 acc noise err 1-2).

## Latency — 0.80x-0.90x torch sequential

| MxKxN | seq/fused |
|---|--:|
| 2048x4096x4096 | 0.80x |
| 4096x4096x4096 | 0.81x |
| 8192x4096x11008 | 0.87x |
| 8192x8192x14336 | 0.90x |
| 8192x4096x14336 | 0.81x |

cuBLAS per-GEMM efficiency outweighs the x-read-once fusion benefit.

## File
fused_gated_mlp_h200.py — sha256 ba746359b385f43f2bf78c0aaf68b8aad4fdee2942ce7038b400726ad877dfa1
