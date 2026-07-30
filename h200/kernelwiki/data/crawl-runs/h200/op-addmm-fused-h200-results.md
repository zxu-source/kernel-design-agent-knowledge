# Fused AddMM (Triton) on H200
Date: 2026-07-21 (phase2). GEMM + alpha/beta residual epilogue vs torch.addmm.
bf16. H200, Triton 3.6.0. CUDA events min of 20.

## Purpose: SPEEDUP (characterization — torch.addmm already fused).

## Correctness — PASS (bf16 accumulation noise).

## Latency — 0.86x-0.95x cuBLAS

| MxNxK | Triton TF | cuBLAS TF | cuBLAS/Triton |
|---|--:|--:|--:|
| 2048³ | 385 | 449 | 0.86x |
| 4096³ | 631 | 689 | 0.92x |
| 8192x8192x4096 | 662 | 702 | 0.94x |
| 8192³ | 613 | 663 | 0.92x |
| 8192x11008x4096 | 598 | 632 | 0.95x |

torch.addmm is already a fused cuBLAS op; Triton ~0.9x (tracks bf16 GEMM baseline).

## File
addmm_fused_h200.py — sha256 ccfe165b8e2810b603f118a99183958d3639f387b9c064547080fa89ee216e29
