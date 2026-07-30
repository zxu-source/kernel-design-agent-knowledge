# KDA Plan Draft: H200 FP16 Row-wise Softmax — Warp-Shuffle Reduction Experiment

**Date**: 2026-07-17
**Target Hardware**: NVIDIA H200 (SM90, Hopper)
**Data Type**: FP16 (half, `__half`) with FP32 accumulator intermediates
**Experiment Type**: Micro-benchmark comparison: shared-memory tree reduction vs. hierarchical warp-shuffle reduction

---

## 1. Problem Statement

Row-wise softmax over a `[M, N]` FP16 matrix requires per-row max-finding, sum-of-exp, and normalization. The reduction phase (finding the row max and sum) is bandwidth-bound when implemented with naive shared-memory atomics or full shared-memory tree reductions. Warp-shuffle intrinsics (`__shfl_xor_sync`) perform the reduction entirely in registers within a warp, eliminating shared-memory round-trips for the intra-warp phase. This experiment quantifies the speedup on H200.

Since H200 (SM90) has a balanced SFU-to-MMA ratio, we use hardware `__expf()` and `__hfma()`, not software-emulated exp (which is only recommended on Blackwell SM100 per KernelWiki `technique-software-exp`).

---

## 2. Kernel Designs (Conceptual — No Code Yet)

### 2.1 Baseline: Shared-Memory Tree Reduction (pure smem)

- **Thread mapping**: 1D block of `BLOCK_SIZE` threads. Each block processes `ROWS_PER_BLOCK` rows. Threads within a block are partitioned into teams handling one row each.
- **Phase 1 — Per-thread partials**: Each thread strides across its assigned portion of the row, accumulating a local FP32 max (or sum-of-exp). Stride = `ROWS_PER_BLOCK * threads_per_row`.
- **Phase 2 — Shared-memory tree reduction**: Threads write partials to `__shared__` scratch buffers (one per row). Then perform an in-place tree reduction over `threads_per_row` elements. Only lane 0 of each row-team holds the final reduced value.
- **Phase 3 — Rescale and normalize**: Lane 0 broadcasts the final max/sum to team members. All threads compute `exp(x - max) / sum` and write FP16 output.
- **Shared memory footprint**: `ROWS_PER_BLOCK * threads_per_row * sizeof(float)` for max buffer and sum buffer.
- **Fixed config**: `BLOCK_SIZE=256`, `ROWS_PER_BLOCK=4`. Threads per row = 64.

### 2.2 Candidate: Hierarchical Warp-Shuffle + Shared-Memory Reduction

- **Thread mapping**: Identical 1D block shape as baseline for fair comparison.
- **Phase 1 — Per-thread partials**: Same strided accumulation as baseline.
- **Phase 2a — Intra-warp butterfly reduction** (the optimization): Within each warp, use `__shfl_xor_sync(0xFFFFFFFF, val, offset)` in a `for (offset = 16; offset > 0; offset >>= 1)` loop. This produces a single partial per warp in lane 0, entirely in registers — no shared memory.
- **Phase 2b — Inter-warp shared-memory reduction**: Each warp's lane 0 writes its partial to `__shared__ float partials[num_warps]`. A small shared-memory reduction across warps produces the final row max/sum. With typical row widths (≤1024 elements), this is 1-2 warps, so the inter-warp phase is trivial (1-2 element smem reduce).
- **Phase 3 — Rescale and normalize**: Same as baseline. Lane 0 holds the final value; broadcast to all row-team threads; compute `exp(x - max) / sum`.
- **Shared memory footprint**: `ROWS_PER_BLOCK * num_warps_per_row * sizeof(float)` — significantly smaller than baseline since the per-thread partials are reduced in registers.
- **Key evidence**: Pattern from KernelWiki `technique-software-exp` (warp-shuffle softmax in FA4) and `technique-vectorized-loads` (hierarchical warp+smem pattern). Production validation: sglang PR #8130 saw 5-7% speedup from warp reduce in per-token quant (a structurally similar row-wise reduction).

---

## 3. PyTorch Reference Baseline

```python
y_ref = torch.softmax(x_half.float(), dim=-1).half()
```

- Input: `torch.float16` tensor on CUDA.
- Reference output: upcast to FP32, run `torch.softmax`, downcast to FP16.
- Used for both correctness validation (max absolute error ≤ 5e-3 for FP16) and performance comparison.

---

## 4. Task Contract

### 4.1 Constraints
- No CUTLASS, Triton, FlashInfer, Tensor Cores, TMA, WGMMA
- No persistent kernels, warp specialization, pipelines, cooperative groups
- No approximated exp differences between kernels
- Identical block size (256) for comparison runs, input data, FP32 accumulation
- Torch is the performance baseline in every tested shape
- Torch must be measured but NOT used as kernel implementation

### 4.2 Correctness Requirements
- Max absolute error ≤ 5e-3 vs PyTorch reference
- Max relative error ≤ 1e-2 for non-zero reference values
- Zero NaN or Inf in output
- Row-sum deviation ≤ 1e-4 (each row should sum to 1.0)

### 4.3 Success Criteria
1. Both kernels pass correctness on all 32 configs (8 shapes × 4 input classes)
2. Both kernels measured across all 8 shapes with identical protocol
3. Benchmark: ≥20 warmups, ≥100 event-timed iterations, 3 independent groups
4. Report: medians, min/max, effective bandwidth, custom-to-custom speedup, vs-torch
5. Torch timing method is documented and fair (same warmup/repeat/event structure)

### 4.4 Stop Condition
- Stop when candidate median time ≤ 0.95 × torch median (candidate exceeds torch with 5% margin), OR
- After 3 consecutive tuning iterations without improvement (within ±2% of best-so-far)

---

## 5. Benchmark Configuration

### 5.1 Eight Required Shapes

| # | Name        | M (rows)  | N (cols)   | Rationale                                          |
|---|-------------|-----------|------------|-----------------------------------------------------|
| 1 | small-sq    | 256       | 256        | Small square, cache-resident                       |
| 2 | small-wide  | 64        | 1024       | Few rows, wide columns — stresses reduction width  |
| 3 | small-tall  | 1024      | 64         | Many rows, narrow — stresses launch overhead        |
| 4 | med-sq      | 1024      | 1024       | Medium square, L2 pressure                         |
| 5 | med-wide    | 256       | 4096       | Medium rows, wide — typical LLM head-dim           |
| 6 | med-tall    | 4096      | 256        | Many rows, medium width                            |
| 7 | large-sq    | 4096      | 4096       | Large square, HBM-bandwidth bound                  |
| 8 | large-wide  | 1024      | 8192       | Large K dimension — tests reduction at scale       |

### 5.2 Four Required Input Classes

| # | Class           | Description                                       | Purpose                                      |
|---|-----------------|---------------------------------------------------|----------------------------------------------|
| A | Uniform random  | `U(-1, 1)` in FP16                                | Baseline, typical initialization             |
| B | Large magnitude | `N(0, 10)` in FP32 → FP16                        | Stresses numerical stability (exp overflow)  |
| C | Biased          | `U(0, 1) + col_bias` where bias ∈ [0, 5]          | Skewed distributions test max-finding        |
| D | Zero rows       | 10% of rows are all zeros                          | Edge case: max = 0, uniform softmax output   |

### 5.3 Timing Protocol

- **CUDA events**: `cudaEventCreate`, `cudaEventRecord`, `cudaEventElapsedTime`.
- **Warmup**: 20 iterations (discarded) — primes the GPU clock, fills caches.
- **Measurement**: 100 timed iterations.
- **Groups**: 3 independent measurement groups (each: 20 warmup + 100 timed). This yields 3 independent sample sets for statistics (min/median across groups, stddev within-group).
- **Synchronization**: `cudaDeviceSynchronize()` before each timing block to isolate kernel execution.

---

## 6. Tuning Axes (for Iterative Refinement)

| Parameter         | Search Space              | Default                  |
|-------------------|---------------------------|--------------------------|
| `BLOCK_SIZE`      | {128, 256, 512, 1024}    | 256                      |
| `ROWS_PER_BLOCK`  | {1, 2, 4, 8}             | 4                        |
| `UNROLL_FACTOR`   | {1, 2, 4}                | 1                        |

Only tune the candidate kernel. The baseline uses a fixed, reasonable configuration (`BLOCK_SIZE=256`, `ROWS_PER_BLOCK=4`) — it's the reference to beat, not a second tuning target.

---

## 7. File Structure (Planned)

```
rowwise_softmax_warp_reduce_h200/
├── docs/
│   ├── draft.md                  # This file
│   ├── plan.md                   # Executable checklist
│   └── kernelwiki_evidence.md    # Evidence from KernelWiki queries
├── src/
│   ├── kernels/
│   │   ├── softmax_baseline.cu   # Shared-memory tree reduction (baseline)
│   │   └── softmax_candidate.cu  # Hierarchical warp-shuffle reduction (candidate)
│   ├── bench.py                  # Python driver: generate inputs, launch kernels, time, validate
│   ├── timer.cuh                 # CUDA event timing utilities (shared header)
│   └── validate.cuh              # FP16 correctness check utilities
├── profile/
│   └── (NCU reports after profiling passes)
├── outputs/
│   └── results.json              # Structured benchmark results
└── runs/
    └── (dated run logs)
```

---

## 8. References

- KernelWiki `technique-software-exp`: Warp-shuffle `__shfl_xor_sync` max/sum reduction in FA4 softmax (SM100, pattern applies to SM90).
- KernelWiki `technique-vectorized-loads`: Hierarchical warp+smem reduction pattern (two-phase).
- sglang PR #8130: 5-7% speedup from warp reduce in per-token FP8 quant.
- `blog-amandeep-nvfp4`: Caveat — shuffle overhead may dominate for very small K dimensions.
- CUDA Programming Guide §B.14 (Warp Shuffle Functions) and §3.2.5.3 (Shared Memory).

---

## 9. Risks and First Steps

- **Risk**: The selected KernelWiki page contains a softmax code example with shuffle reductions as an incidental example, not as a dedicated Hopper reduction page. Mitigation: the shuffle reduction pattern is architecture-agnostic and validated by production PRs.
- **Risk**: Warp-shuffle may not beat shared-memory at small K (per `blog-amandeep-nvfp4`). Mitigation: 8-shape grid covers small-K (N=64) through large-K (N=8192).
- **Risk**: Remote conda wrapper may fail. Use `scripts/h200-raw.sh` from parent workspace. Record the exact environment command.
- **First steps**: Compile and validate the shared-memory baseline. Then add hierarchical shuffle max/sum reductions and repeat the full validation and benchmark protocol.
