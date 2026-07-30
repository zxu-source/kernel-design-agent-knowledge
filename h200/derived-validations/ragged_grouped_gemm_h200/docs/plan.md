# Executable plan

## 1. Scaffold the isolated CUDA project

- Files: `CMakeLists.txt`, `include/ragged_grouped_gemm.hpp`, `src/`, `tests/`,
  `scripts/`.
- Function: define the packed-offset API and candidates without modifying an
  earlier GEMM task.
- Verify: `find ragged_grouped_gemm_h200 -maxdepth 2 -type f` locally; compile
  remotely with `nvcc -arch=sm_90`.
- Success: one self-contained binary is produced on H200.
- Fallback: use the direct `nvcc` CUDA runtime path because the present remote
  container lacks the documented conda environment and CMake.

## 2. Implement `candidate_00_baseline`

- Files: `src/ragged_grouped_gemm.cu`, `tests/ragged_grouped_gemm_test.cu`.
- Function: validate host offsets, skip `M_e=0`, then launch a guarded
  BF16-WMMA/FP32-accumulation kernel for every nonempty expert.
- Verify: remote test binary covers all required expert counts/distributions
  and small CPU-FP32 reference cases.
- Success: all checks report finite outputs, nonoverlap, and tolerance pass.
- Fallback: replace WMMA with a scalar BF16-to-FP32 CUDA baseline if a given
  toolchain rejects BF16 WMMA.

## 3. Implement `candidate_01_kernelwiki`

- Files: same source plus `include/ragged_grouped_gemm.hpp`.
- Function: host-build a `TileTask{expert,tile_m,tile_n}` list, upload it once,
  reset a device ticket, and launch a bounded persistent grid; each CTA claims
  a next tile with `atomicAdd`.
- Verify: compare candidate output to CPU FP32 on small cases and to baseline
  on all benchmark-size cases; assert zero-M creates neither a task nor an
  invalid pointer dereference.
- Success: one candidate kernel launch handles all work, and offsets stay in
  their packed output ranges.
- Fallback: retain the static flattened one-launch grid (no atomic loop) and
  record a persistent-scheduler failure/regression in `candidates.jsonl`.

## 4. H200 measurement and references

- Files: `tests/ragged_grouped_gemm_test.cu`, `benchmark.csv`, `outputs/`.
- Function: CUDA-event measurement with 20 warmups, 100 iterations, 3 trials;
  compare custom baseline, persistent candidate, and a clearly labeled
  per-expert reference path. Metadata construction is separately timed.
- Verify: set `CUDA_VISIBLE_DEVICES=0`; record GPU and build command; calculate
  useful FLOPs from actual M values only.
- Success: rows cover uniform, skewed, empty, small, mixed, single-expert, and
  at least three K/N pairs including 4096x4096.
- Fallback: if the environment lacks cuBLAS/PyTorch headers, use CPU FP32 only
  for small correctness cases and label the custom per-expert CUDA baseline as
  a comparison candidate rather than as an independent library reference.

## 5. Profile, independent review, and reporting

- Files: `profile/`, `docs/codex_review.md`, `docs/final_report.md`.
- Function: query NCU availability, profile baseline and best candidate when
  possible, then independently audit source/measurements against this plan.
- Verify: `ncu --version`, `ncu --query-metrics`, and a recorded error if they
  are absent or permission-limited.
- Success: review verdict has no unresolved correctness critical issue; failed
  candidates remain in the JSONL record.
- Fallback: use CUDA-event launch-count/resource discussion and quote the exact
  NCU availability failure without inventing counters.
