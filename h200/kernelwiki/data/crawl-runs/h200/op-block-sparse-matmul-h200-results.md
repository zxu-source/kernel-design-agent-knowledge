# Block-Sparse Matmul (Triton) on H200
Date: 2026-07-21 (phase2). 1D grid over nonzero mask tiles (indirect mb/nb) vs
dense all-tiles. bf16, 128x128 tiles. H200, Triton 3.6.0. CUDA events min of 20.

## Purpose: SPEEDUP (record-negative for naive random-tile version).

## Correctness — PASS (nonzero tiles match dense-masked exactly, err=0.0).

## Latency — naive random-tile block-sparse is ~0.01x dense (100x slower!)

| MxNxK (density, tiles nz/total) | dense/sparse |
|---|--:|
| 4096x4096x4096 (25%, 266/1024) | 0.012x |
| 4096x4096x4096 (10%, 96/1024) | 0.012x |
| 2048x2048x2048 (25%, 71/256) | 0.009x |
| 8192x8192x4096 (25%, 1060/4096) | 0.012x |

The naive version's random tile access order destroys L2 locality (consecutive
programs hit unrelated A rows / B columns) -> memory-latency-bound at tiny
effective bandwidth. Block-sparse only beats dense with locality-aware tile
sorting (cuSPARSELt / FlashInfer block-sparse / Triton BlockSparse).

## File
block_sparse_mm_h200.py — sha256 (see PROVENANCE)
