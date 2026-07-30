# BF16 GEMM (Triton vs cuBLAS) on H200
Date: 2026-07-21 (phase2). Triton bf16 GEMM (BM=128 BN=256 BK=64 nw=8 ns=3,
tl.dot bf16->fp32) vs cuBLAS torch.matmul (TF32 off). H200, Triton 3.6.0.
CUDA events min of 20 trials. bf16 peak ~989 TFLOPS.

## Purpose: SPEEDUP baseline / characterization.

## Correctness — PASS
vs fp32 reference: bf16 accumulation noise 0.5-1.0 (expected at large K; bf16 ~3-decimal, grows with sqrt(K)).

## Throughput

| M=N=K | Triton TF | Triton util | cuBLAS TF | cuBLAS util | cuBLAS/Triton |
|--:|--:|--:|--:|--:|--:|
| 1024 | 74  | 7%  | 144 | 15% | 0.51x |
| 2048 | 427 | 43% | 522 | 53% | 0.82x |
| 4096 | 674 | 68% | 758 | 77% | 0.89x |
| 8192x8192x4096 | 696 | 70% | 781 | 79% | 0.89x |
| 8192 | 715 | 72% | 725 | 73% | 0.99x |

Triton reaches ~72% of bf16 peak at large shapes, within ~10% of cuBLAS.
Launch-bound at 1024^3 (7%).

## File
`bf16_gemm_h200.py` — sha256 `3e359cdd7f959924c7a8af820f36b8102e0f6ae852094866202729c2fb620b54`
