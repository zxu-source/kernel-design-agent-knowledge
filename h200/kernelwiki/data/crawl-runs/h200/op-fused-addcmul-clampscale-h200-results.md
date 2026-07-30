# Fused Addcmul + Clamp+Scale (Triton) on H200
Date: 2026-07-21 (phase2). vs torch. fp32/bf16. H200, Triton 3.6.
## Addcmul: 0.86x-0.98x torch (~parity; bit-identical).
## Clamp+Scale: 1.61x-1.86x torch (bit-identical; 2-op fusion).
## File: fused_addcmul_clampscale_h200.py — sha256 9bc2c9f4961b91750f800ea3f81b6ad69a3c1b04727a123a0f5fc332da9d8c91
