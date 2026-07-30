# Embedding Lookup (Triton) on H200
Date: 2026-07-21 (phase2). Gather by token id vs torch indexing. H200, Triton 3.6.
CUDA events min of 50.

## Purpose: characterization (record-negative — memcpy-bound).

## Correctness — PASS (bit-identical, err=0).

## Latency — ~parity (0.82x-1.02x)

| MxVxD | fp32 torch/Triton | bf16 torch/Triton |
|---|--:|--:|
| 8192x128256x4096 | 0.96x | 0.90x |
| 4096x128256x4096 | 0.88x | 0.82x |
| 8192x32000x4096 | 0.95x | 0.89x |
| 8192x128256x8192 | 1.00x | 0.95x |
| 16384x128256x4096 | 1.02x | 0.97x |

## File
embedding_lookup_h200.py — sha256 71a2935095e06cb03774f1a58203cf39782f84b81e56fe9085a18ae9b1f05f0d
