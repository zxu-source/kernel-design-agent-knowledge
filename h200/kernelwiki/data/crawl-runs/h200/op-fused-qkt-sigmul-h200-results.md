# Fused QK^T+Scale+Mask + Sigmoid*Mul (Triton) on H200
Date: 2026-07-21 (phase2). vs torch. H200, Triton 3.6.
## QK^T+scale+mask: 2.15x-5.49x (err_finite ~7e-8; -inf mask excluded).
## Sigmoid*mul: 1.37x-1.67x (err ~2e-7 fp32 / ~4e-3 bf16).
## File: fused_qkt_sigmul_h200.py — sha256 c04b65ff76afee504798bbb1d7f1218d7f21a646f4cb4b576f4a50fbf2f99a7a
