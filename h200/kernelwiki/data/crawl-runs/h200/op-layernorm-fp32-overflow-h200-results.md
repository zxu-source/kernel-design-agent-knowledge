# LayerNorm fp32-reduction Robustness on H200
Date: 2026-07-21 (phase2). fp32 vs bf16 variance reduction in LayerNorm. bf16 input.
H200, Triton 3.6.0.

## Purpose: ROBUSTNESS — fp32 reduction keeps LayerNorm accurate as N grows; bf16 degrades / can overflow.

## Result

| M | N | fp32 rel err | bf16 rel err | bf16 degradation |
|--:|--:|--:|--:|--:|
| 4096 | 4096 | 0.24% | 0.48% | 2x |
| 8192 | 8192 | 0.25% | 0.49% | 2x |
| 8192 | 14336 | 0.25% | 0.99% | 4x |
| 8192 | 28672 | 0.24% | 1.45% | 6x |
| 8192 | 57344 | 0.46% | 3.64% | 8x |

bf16 accumulation of sum_x/sum_x2 loses precision as N grows (sum grows, sub-ULP
contributions dropped). fp32 reduction stays accurate. For adversarial large-
magnitude inputs the bf16 sum can overflow to inf; fp32 is the robustness fix.

## File
ln_fp32_overflow_h200.py — sha256 88538b8ee039d3f2f37674148fa61742783a59ac8e25ecce98c2fcc0fb4c414e
