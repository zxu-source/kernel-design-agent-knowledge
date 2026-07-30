# TF32 GEMM (Triton vs cuBLAS) on H200
Date: 2026-07-21 (phase2). Triton TF32 matmul (fp32 inputs, input_precision='tf32',
BM=128 BN=128 BK=32 nw=4 ns=3) vs cuBLAS torch.matmul (allow_tf32=True).
H200, Triton 3.6.0. CUDA events min of 20 trials. TF32 peak ~989 TF.

## Purpose: SPEEDUP baseline (record-negative)

## Correctness — PASS (TF32 rounding 0.04-0.12)

## Throughput

| M=N=K | Triton TF | Triton util | cuBLAS TF | cuBLAS util | cuBLAS/Triton |
|--:|--:|--:|--:|--:|--:|
| 1024 | 28 | 3% | 87  | 9%  | 0.32x |
| 2048 | 75 | 8% | 291 | 29% | 0.26x |
| 4096 | 81 | 8% | 392 | 40% | 0.21x |
| 8192x8192x4096 | 82 | 8% | 349 | 35% | 0.23x |
| 8192 | 82 | 8% | 416 | 42% | 0.20x |

Naive Triton TF32 stuck at ~8% peak (0.2-0.3x cuBLAS); input_precision='tf32'
explicit did not help. Use cuBLAS / tuned Triton TF32 for real throughput.
(Contrast bf16 GEMM: Triton 72% peak.)

## File
`tf32_gemm_h200.py` — sha256 `f55a4994703712d0f4e64775366c5fbf64e28022c80b645cc4881a494842fb76`
