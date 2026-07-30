# Split-K GEMM (Triton) on H200
Date: 2026-07-21 (phase2). Split-K bf16 GEMM (3D grid M,N,SPLIT; atomic_add fp32
partials) vs single GEMM. H200, Triton 3.6.0. CUDA events min of 30 trials.

## Purpose: SPEEDUP (more parallelism for tall-K / small-MN).

## Correctness — PASS (err=0.0 vs fp32 reference, both split-K and single).

## Latency

| MxNxK (SPLIT) | single/split-K |
|---|--:|
| 128x128x8192 (8) | 1.29x |
| 256x256x8192 (8) | 1.26x |
| 128x128x16384 (16) | 2.20x |
| 256x128x8192 (8) | 1.28x |
| 512x512x4096 (4) | 0.77x |

Split-K wins for tall-K/small-MN (best 2.20x at 128x128 K=16384). Loses at
512x512 (0.77x) where the GEMM already fills the GPU and atomic-add overhead
dominates. Use split-K when M_tiles*N_tiles << num_SMs (132).

## File
`splitk_gemm_h200.py` — sha256 `6ab223a6a987eb55d543c3faba04db470bf8a2a933047edc3accf7b01342eef6`
