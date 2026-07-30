# Mixed-Precision GEMM Accumulator Robustness on H200
Date: 2026-07-21 (phase2). bf16 GEMM fp32-acc vs bf16-acc. H200, Triton 3.6.0.

## Purpose: ROBUSTNESS — fp32 accumulator mandatory for bf16/fp16 GEMM.

## Result

| MxNxK | fp32-acc rel err | bf16-acc rel err | bf16 worse by | bf16 inf |
|---|--:|--:|--:|--:|
| 2048x2048x2048 | 3.5e-6 | 2.58% | 7317x | no |
| 4096x4096x4096 | 5.5e-6 | 3.00% | 5478x | no |
| 4096x4096x8192 | 1.1e-5 | 4.43% | 3988x | no |
| 2048x2048x16384 | 2.2e-5 | 6.87% | 3109x | no |
| 2048x2048x32768 | 4.4e-5 | 12.86% | 2928x | no |

bf16 accumulator drops sub-ULP partial sums each K-step; error grows with K
(2.6% -> 12.9%). fp32 accumulator stays ~1e-5. For large-magnitude inputs the
bf16 accumulator overflows to inf. **Always accumulate bf16/fp16 GEMM in fp32.**

## File
mixed_prec_gemm_acc_h200.py — sha256 e6631027d5fc908aa74bfb85b763d1d94d0e9286268230f8967acebe8b5a089b
