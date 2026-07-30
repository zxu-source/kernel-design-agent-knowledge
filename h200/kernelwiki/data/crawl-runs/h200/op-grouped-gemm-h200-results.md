# Grouped GEMM (Triton) on H200
Date: 2026-07-21 (phase2). GROUP expert bf16 GEMMs in one launch (2D tile grid +
group lookup) vs torch loop. H200, Triton 3.6.0. CUDA events min of 20 trials.

## Purpose: SPEEDUP (one launch vs GROUP cuBLAS calls).

## Correctness — PASS (bit-identical, err=0.0 across all shapes)

## Latency — modest 0.85x-1.39x vs torch loop

| GROUP x Mg x N x K | loop/grouped |
|---|--:|
| 8 x 512 x 4096 x 4096 | 0.93x |
| 8 x 1024 x 4096 x 4096 | 0.85x |
| 16 x 256 x 4096 x 4096 | 1.09x |
| 32 x 128 x 4096 x 4096 | 1.39x |
| 8 x 2048 x 8192 x 4096 | 1.02x |

Biggest win at many small expert groups (32x128 -> 1.39x launch amortization);
~parity when per-expert GEMM dominates. Production grouped-GEMM wins need token
sorting + no-padding. Note: an earlier 1D-grid version had a correctness bug
(no N-tiling); this 2D-grid version is bit-correct.

## File
`grouped_gemm_h200.py` — sha256 `819edb7ddc5e3d54e23ccb7b8dba569ac0e2948fc31a7c611da394d49364973a`
