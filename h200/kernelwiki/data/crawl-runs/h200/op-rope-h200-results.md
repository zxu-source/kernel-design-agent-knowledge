# RoPE (Triton) on H200
Date: 2026-07-21 (phase2). Rotate-half RoPE vs torch (cat/slice). fp16.
H200, Triton 3.6.0. CUDA events min of 50.

## Purpose: SPEEDUP (fused RoPE).

## Correctness — PASS (err 4.9e-4 vs torch).

## Latency — 1.66x-5.61x faster than torch

| BxHxSxD | torch/Triton |
|---|--:|
| 1x32x4096x128 | 2.99x |
| 2x32x2048x128 | 3.01x |
| 1x32x8192x128 | 3.07x |
| 1x8x4096x64 | 1.66x |
| 1x32x4096x256 | 5.61x |

## File
rope_h200.py — sha256 68e8cf9f039296fe2cdb4e827c29af261f65eb2a3f78d490725d69ad6da7175a
