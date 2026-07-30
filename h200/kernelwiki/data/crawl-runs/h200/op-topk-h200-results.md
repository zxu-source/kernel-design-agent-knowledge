# Per-Row Top-K (Triton) on H200
Date: 2026-07-21 (phase2). k-pass max+argmax+mask per row (K=5) vs torch.topk.
fp32. H200, Triton 3.6.0. CUDA events min of 50.

## Purpose: characterization (top-k sampling building block).

## Correctness — PASS (val_err=0, index set_match=1.0 vs torch.topk).

## Latency (K=5, fp32)

| M | N | torch/Triton |
|--:|--:|--:|
| 4096 | 4096 | 4.14x |
| 8192 | 8192 | 3.54x |
| 4096 | 32000 | 0.58x |
| 8192 | 32000 | 0.57x |

Moderate N: 3.5-4.1x faster. N=32000: O(kN) k-pass slower than torch partial sort.

## File
topk_h200.py — sha256 ba46613537a5ed6f5e822d689b26bbd6ebecb33cbcda9858a35db4797fc3cdd2
