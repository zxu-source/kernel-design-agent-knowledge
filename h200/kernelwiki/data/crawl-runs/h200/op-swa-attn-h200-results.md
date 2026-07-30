# Sliding-Window Attention (Triton) on H200
Date: 2026-07-21 (phase2). FA-2 + causal+window mask (query i -> keys [i-W+1,i]).
vs full-materialized torch reference. fp16. H200, Triton 3.6.0. CUDA events min 20.

## Purpose: BOTH — speedup (no O(M^2) materialization) + accuracy (windowed context).

## Correctness — PASS (err 2.4e-4 vs full-materialized reference).

## Latency — 6.9x-14.3x faster than the naive O(M^2) reference

| BxHxMxD (W) | naive-ref/Triton |
|---|--:|
| 1x8x8192x64 (512) | 14.27x |
| 1x8x8192x128 (1024) | 7.70x |
| 4x8x4096x64 (512) | 13.04x |
| 1x4x16384x64 (1024) | 13.30x |
| 2x16x2048x128 (512) | 6.88x |

Robustness: finite-neg (-1e30) masking avoids -inf-(-inf)=NaN on fully-masked
tiles (rows whose window doesn't reach tile 0). Deferred: naive version loops
full N (window=accuracy not speed); window-capping needs dynamic loop bounds.

## File
slidwindow_attn_h200.py — sha256 bf8ae79f13ae8214a13bdd944b314a4b262fdf3db5a32d586aebc670db04020f
