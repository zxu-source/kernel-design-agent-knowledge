# Fused Concat (Triton) on H200
Date: 2026-07-21 (phase2). One-program-per-row concat of 3 [M,Dk] -> [M,3Dk] vs
torch.cat. H200, Triton 3.6.0. CUDA events min of 50.

## Purpose: SPEEDUP (record-negative — concat is memcpy-bound, torch.cat already near-optimal).

## Correctness — PASS (bit-identical, err=0.0).

## Latency — ~parity (0.88x-1.02x); 0.22x regression at fp32 Dk=14336 (large-tile pressure)

| M x Dk | fp32 torch/Triton | bf16 torch/Triton |
|---|--:|--:|
| 4096x4096 | 0.95x | 0.88x |
| 8192x4096 | 0.98x | 0.95x |
| 8192x8192 | 0.99x | 0.98x |
| 8192x14336 | 0.22x | 1.01x |
| 16384x14336 | 0.22x | 1.02x |

Use torch.cat for concat.

## File
concat_h200.py — sha256 994f7cece4baf37d856e6c89fb8a6038c7f3f33e26d6783a7860e6c721cb8354
