# ReLU Forward+Backward (Triton) on H200
Date: 2026-07-21 (phase2). ReLU fwd (max(x,0)) + bwd (grad_y*(x>0)) vs torch.
H200, Triton 3.6.0. CUDA events min of 50.

## Purpose: characterization.

## Correctness — PASS (bit-identical, err=0, fwd+bwd).

## Latency

| MxN | fwd fp32 torch/Triton | bwd bf16 torch/Triton |
|---|--:|--:|
| 4096x4096 | 0.84x | 1.35x |
| 8192x8192 | 0.92x | 1.12x |
| 8192x14336 | 0.93x | 1.07x |
| 16384x14336 | 0.94x | 1.02x |

fwd ~0.9x (torch.relu single fast op); bwd 1.0-1.35x (modest).

## File
relu_fwd_bwd_h200.py — sha256 5f6158da6cca2c06f0bf620f161b78b5941b5a1ad6e953ff0c04e83978398586
