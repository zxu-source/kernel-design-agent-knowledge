# Batched Matmul (Triton) on H200
Date: 2026-07-21 (phase2). 3D grid (B,M,N) vs torch.bmm (cuBLAS strided batched).
bf16. H200, Triton 3.6.0. CUDA events min of 20.

## Purpose: characterization (record-negative vs cuBLAS bmm).

## Correctness — PASS (err=0.0).

## Latency — 0.49x-0.69x cuBLAS

| BxMxNxK | Triton TF | cuBLAS TF | cuBLAS/Triton |
|---|--:|--:|--:|
| 16x512³ | 144 | 234 | 0.61x |
| 32x1024³ | 310 | 635 | 0.49x |
| 16x2048³ | 393 | 767 | 0.51x |
| 8x2048x2048x4096 | 538 | 783 | 0.69x |
| 32x1024x1024x2048 | 354 | 714 | 0.50x |

Naive Triton bmm ~0.5-0.7x cuBLAS (strided-batched cuBLAS tuned). Use torch.bmm.

## File
bmm_h200.py — sha256 13c433fab3bf7abd8a8b49f718d6e73c52812e6ebcabf8ab51b42b8111193f43
