# Depthwise Conv (Triton) on H200
Date: 2026-07-21 (phase2). Direct per-channel depthwise conv (3x3, stride1, pad1,
9-tap gather) vs torch conv2d(groups=C). fp32. H200, Triton 3.6.0. CUDA events min 50.

## Purpose: characterization (CNN building block).

## Correctness — PASS (bit-identical, err=0).

## Latency — ~parity small / 1.53-1.76x faster large

| CxHxW | torch/Triton |
|---|--:|
| 64x128x128 | 0.84x |
| 128x128x128 | 1.04x |
| 256x64x64 | 0.83x |
| 128x256x256 | 1.53x |
| 64x512x512 | 1.76x |

## File
depthwise_conv_h200.py — sha256 f1ed5b10a1eb8f07c2154220d952a7837e0f6ce47f836c8dee424feaed66a6d0
