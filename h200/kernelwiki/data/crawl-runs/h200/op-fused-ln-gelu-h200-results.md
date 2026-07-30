# Fused LN+GELU + Fused RMSNorm+RoPE (Triton) on H200
Date: 2026-07-21 (phase2). Fused 2-op combos vs torch separate. fp32. H200, Triton 3.6.
## Fused LN+GELU: 1.81x-2.29x (err ~1e-6).
## Fused RMSNorm+RoPE: 1.77x-2.74x (err ~1e-6).
## File: fused_ln_gelu_rmsnorm_rope_h200.py — sha256 b477a309297cc434c53db110f4004f3d3fc15486ad46a8510fdf77e59341f3cd
