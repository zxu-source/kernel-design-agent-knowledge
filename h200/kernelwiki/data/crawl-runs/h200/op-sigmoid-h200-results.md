# Sigmoid Forward (Triton) on H200
Date: 2026-07-21 (phase2). 1/(1+exp(-x)) vs torch.sigmoid. H200, Triton 3.6.
## Purpose: characterization (record-negative).
## Correctness — PASS (err 1.2e-7 fp32 / 0.0 bf16).
## Latency — 0.85x-0.99x torch (~parity).
| MxN | fp32 torch/Triton |
|---|--:|
| 4096x4096 | 0.85x |
| 8192x8192 | 0.93x |
| 8192x14336 | 0.95x |
## File: groupnorm_sigmoid_h200.py — sha256 092949ff4ebe78a3ee8d6ff0da87b3fc56f56b1278e4b936ed3a16ac4d144709
