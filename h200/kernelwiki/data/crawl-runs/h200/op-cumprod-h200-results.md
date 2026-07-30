# Cumprod (Triton) on H200
Date: 2026-07-21 (phase2). exp(cumsum(log(x))) vs torch.cumprod. fp32.
H200, Triton 3.6.0. CUDA events min 50.
## Purpose: characterization.
## Correctness — PASS (rel ~2e-6).
## Latency — 1.50x-3.28x.
| MxN | torch/Triton |
|---|--:|
| 4096x4096 | 3.28x |
| 8192x8192 | 1.50x |
## File: kl_div_cumprod_h200.py — sha256 662e09b43ce69a3b9696df60d4d52d636e32509d8f2ace1012ce4b2d225329ab
