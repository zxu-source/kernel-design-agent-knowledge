# KL Divergence (Triton) on H200
Date: 2026-07-21 (phase2). Fused log_softmax + target gather vs torch.kl_div. fp32.
H200, Triton 3.6.0. CUDA events min 50.
## Purpose: SPEEDUP (fused).
## Correctness — PASS (err ~3e-8).
## Latency — 2.96x-3.07x moderate N; 0.55x at N=32K (BLOCK spill).
| MxN | torch/Triton |
|---|--:|
| 4096x4096 | 3.07x |
| 8192x8192 | 2.96x |
| 4096x32000 | 0.55x |
## File: kl_div_cumprod_h200.py — sha256 662e09b43ce69a3b9696df60d4d52d636e32509d8f2ace1012ce4b2d225329ab
