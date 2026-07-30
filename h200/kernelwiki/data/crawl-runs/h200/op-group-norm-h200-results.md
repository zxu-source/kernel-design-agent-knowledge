# GroupNorm (Triton) on H200
Date: 2026-07-21 (phase2). Fused GroupNorm [N,C] G groups vs torch.group_norm. fp32.
## Purpose: SPEEDUP (fused).
## Correctness — PASS (err ~1e-6).
## Latency — 1.48x-5.29x faster than torch.
| NxCxG | torch/Triton |
|---|--:|
| 4096x4096x32 | 1.48x |
| 8192x8192x32 | 2.41x |
| 8192x4096x16 | 2.34x |
| 4096x4096x8 | 5.29x |
## File: groupnorm_sigmoid_h200.py — sha256 092949ff4ebe78a3ee8d6ff0da87b3fc56f56b1278e4b936ed3a16ac4d144709
