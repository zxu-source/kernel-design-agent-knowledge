# Codex Review — Iteration 0

## Verdict

**BLOCKED: do not benchmark or count an iteration yet.** The submitted kernels are not safe for the required irregular row counts.

## Evidence review

- `docs/kernelwiki_evidence.md` records real KernelWiki queries and accurately labels `technique-software-exp` as a supporting Softmax example, not a dedicated SM90 shuffle-reduction page.
- The hierarchical reduction mapping is traceable to `technique-vectorized-loads` and is consistent with the candidate's intended implementation.
- Before final acceptance, the evidence file must include the exact inspected source-page and `PROVENANCE.yaml` paths for every cited PR; a title/summary alone is not sufficient provenance.

## Mandatory code correction

Both kernels originally executed `if (row >= rows) return;` before block-wide `__syncthreads()`. In a tail block such as rows=127, threads assigned invalid rows exit while valid rows wait at barriers. This is undefined CUDA behavior and can deadlock.

Claude began changing the baseline to `bool valid_row = row < rows`, but the current source still dereferences `in_row[j]` and writes `out_row[j]` without guarding invalid rows, and the candidate still has the early return. The required correction is:

1. Every thread in a block reaches every `__syncthreads()`.
2. Invalid-row threads contribute `-INFINITY` for max and `0.0f` for sum.
3. Invalid-row threads perform no input dereference and no output store.
4. Apply the same mechanism to baseline and candidate, with no other algorithmic change.

## Fairness review

- `softmax.cu` uses the same 256-thread / 4-row mapping, FP32 reductions, and `expf` in both kernels. That is the correct starting controlled comparison.
- `docs/plan.md` originally proposed a 48-variant block-size / rows-per-block tuning grid. This violates the single-knowledge-point and same-block-size constraints. That grid has been removed; subsequent iterations must retain the fixed mapping unless an explicitly separate, paired control experiment changes both kernels.
- No H200 build, full correctness matrix, Torch timing, or benchmark.csv results exist yet. Therefore no performance conclusion, including the three-no-improvement stopping rule, can be applied.

## Required next gate

After Claude applies the tail-block correction, compile on H200, run all required shape/input correctness cases, and provide the raw command outputs. Codex will then review the test harness and approve or reject transition to the fair three-group Torch/custom benchmark.
