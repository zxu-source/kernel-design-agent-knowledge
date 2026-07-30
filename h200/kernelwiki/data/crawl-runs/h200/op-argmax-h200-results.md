# Per-Row Argmax (Triton) on H200
Date: 2026-07-21 (phase2). tl.argmax per row vs torch.argmax. H200, Triton 3.6.0.
CUDA events min of 50 trials.

## Purpose: SPEEDUP (sampling top-1 building block).

## Correctness — PASS (100% match vs torch.argmax, agree=1.0).

## Latency — 1.10x-1.76x faster than torch.argmax

| M | N | torch/Triton |
|--:|--:|--:|
| 4096 | 4096 | 1.10x |
| 8192 | 8192 | 1.76x |
| 4096 | 32000 | 1.22x |
| 8192 | 32000 | 1.23x |
| 8192 | 128256 | 1.16x |

## File
`argmax_row_h200.py` — sha256 `faedf21c42d38f1072601f8adb287c38f18e00be690f7d96e362de9f02f41572`
