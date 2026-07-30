# Fused Pre-Norm Residual + Add+SiLU (Triton) on H200
Date: 2026-07-21 (phase2). Fused 2-op combos vs torch. fp32/bf16. H200, Triton 3.6.
## Pre-norm residual: 2.23x-2.86x (err ~1e-6).
## Add+SiLU: 1.36x-1.66x (err ~2e-7 fp32 / ~1e-2 bf16).
## File: fused_prenorm_addsilu_h200.py — sha256 066820337dc37637b026381c48a4383ff5bda438f77c24fd94f8bf98ac974900
