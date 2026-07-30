# Online Softmax (Triton) on H200
Date: 2026-07-21 (phase2). Online softmax (max-subtract, exp, sum, divide) one
program/row. vs torch.softmax. H200, Triton 3.6.0. CUDA events min of 50 trials.

## Purpose: BOTH — robustness (max-subtract prevents exp overflow) + speedup (fused).

## What it does
Per-row: `y = exp(x-max) / sum(exp(x-max))`, fused one pass.

## Correctness — PASS (1e-6..1e-5 vs torch.softmax)

## Latency

| M | N | bf16 torch/Triton |
|--:|--:|--:|
| 4096 | 4096   | 2.18x |
| 8192 | 8192   | 3.36x |
| 8192 | 11008  | 1.23x |
| 4096 | 32000  | 1.30x |
| 8192 | 32000  | 1.36x |
| 4096 | 128256 | **0.23x (regression)** |

Faster at moderate N (4K..32K). At N=128256 the naive BLOCK_N=131072 spills
registers -> 4.3x slower than torch. Use a tiled/flash-style softmax for huge N.

## File
`online_softmax_h200.py` — sha256 `4076eebe8419616b3298959f56ae69684f68330d86e34a5dab86ad5cabcdc1bf`
