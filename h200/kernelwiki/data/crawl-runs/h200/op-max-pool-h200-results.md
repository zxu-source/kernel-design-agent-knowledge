# Max Pooling (Triton) on H200
Date: 2026-07-21 (phase2). 2x2 stride2 max-pool vs torch.max_pool2d. fp32.
H200, Triton 3.6.0. CUDA events min of 50.

## Purpose: characterization (CNN building block).

## Correctness — PASS (bit-identical, err=0).

## Latency — 0.54x-0.68x small / 1.24x-1.45x large

| CxHxW | torch/Triton |
|---|--:|
| 64x128x128 | 0.54x |
| 128x128x128 | 0.68x |
| 256x64x64 | 0.56x |
| 128x256x256 | 1.24x |
| 64x512x512 | 1.45x |

## File
maxpool_h200.py — sha256 cfee5ab35b241fb6423d9f1c1eb80575b0dc6a79fd334ce0fc35e43545559820
