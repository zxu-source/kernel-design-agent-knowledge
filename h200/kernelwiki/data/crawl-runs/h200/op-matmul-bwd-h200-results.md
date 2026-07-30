# Matmul Backward (Triton) on H200
Date: 2026-07-21 (phase2). 2 strided GEMMs (grad_a, grad_b) vs torch autograd
(backward-only timed). bf16. H200, Triton 3.6.0. CUDA events min of 20.

## Purpose: characterization.

## Correctness — OK (bf16 acc noise 0.25-0.50; reduction-order difference).

## Latency — ~0.83x-1.05x cuBLAS (parity)

| MxKxN | torch/Triton |
|---|--:|
| 2048³ | 1.05x |
| 4096³ | 0.83x |
| 8192x8192x4096 | 0.94x |
| 8192x4096x8192 | 0.94x |
| 4096x4096x8192 | 0.93x |

Note: torch must be timed backward-only (forward graph built once); forward+backward
timing inflates the ratio.

## File
matmul_bwd_h200.py — sha256 672cd45a4c2691a6e740770aac7020a80e3c4be31252ed11d1b7fba8144cd9fe
