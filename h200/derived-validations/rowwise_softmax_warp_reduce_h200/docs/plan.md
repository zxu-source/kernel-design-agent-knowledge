# Executable Plan: H200 FP16 Row-wise Softmax — Warp-Shuffle Reduction

**Plan for**: `docs/draft.md`
**Created**: 2026-07-17

Each step is atomic and verifiable. A step is not "done" until its verification passes.

---

## Phase 0: Scaffolding

### Step 0.1 — Create directory structure
```bash
mkdir -p kernels scripts outputs/golden profile runs
```
**Verify**: `ls kernels/ scripts/ outputs/ profile/ runs/` shows five directories with no errors.

### Step 0.2 — Create Python validation harness skeleton
Write `scripts/validate.py` with:
- `--mode save-golden`: generate torch golden outputs for all 8 shapes × 4 input types, save to `outputs/golden/`
- `--mode check-torch`: verify torch self-consistency on all shapes/inputs
- `--mode compare`: load a kernel's output and compare against golden using `torch.allclose(atol=0.01, rtol=0.001)`
- `--mode full-matrix`: run `compare` for all (kernel, shape, input-type) tuples
**Verify**: `python scripts/validate.py --mode check-torch` passes all 32 self-consistency checks.

### Step 0.3 — Create Python benchmark harness skeleton
Write `scripts/benchmark.py` with:
- CUDA Event-based timing (Section 8 of draft)
- `--warmup` and `--iters` flags
- `--output` flag for JSON results file
- Per-shape reporting: mean, median, p5, p95, std, bandwidth
**Verify**: `python scripts/benchmark.py --help` prints usage without error.

### Step 0.4 — Create single-kernel runner script
Write `scripts/run_single.py`:
- `--kernel <name> --shape "M,N" --input-type <type>`
- Loads the compiled `.so`, allocates tensors, launches kernel, returns output tensor
**Verify**: Script imports without error (will fail on kernel load until kernels exist — expected).

### Step 0.5 — Create CUDA compilation script
Write `scripts/compile.py` (or `Makefile`):
- Compiles `.cu` files in `kernels/` to shared libraries (`.so`) loadable via `ctypes` or `torch.utils.cpp_extension`
- Uses `nvcc` with flags: `-arch=sm_90 -O3 -use_fast_math --ptxas-options=-v`
**Verify**: `python scripts/compile.py --help` prints usage without error. The actual compile step is gated on kernel source existing.

---

## Phase 1: Torch Baseline

### Step 1.1 — Save torch golden outputs
```bash
python scripts/validate.py --mode save-golden \
    --shapes "64,64 256,128 1024,256 512,512 4096,512 1024,1024 256,4096 4096,4096" \
    --input-types "uniform large_magnitude outlier all_zero" \
    --output-dir outputs/golden/
```
**Verify**: `outputs/golden/` contains 32 `.pt` files, one per (shape, input-type) pair.

### Step 1.2 — Benchmark torch softmax
```bash
python scripts/bench_torch.py \
    --shapes "64,64 256,128 1024,256 512,512 4096,512 1024,1024 256,4096 4096,4096" \
    --warmup 10 --iters 100 \
    --output outputs/torch_baseline_times.json
```
**Verify**: `outputs/torch_baseline_times.json` contains 8 entries with `mean_us`, `median_us`, etc.

### Step 1.3 — Gate check
Confirm torch golden outputs pass `torch.allclose` against a second torch computation (different code path if possible).
```bash
python scripts/validate.py --mode check-torch
```
**Verify**: Exit code 0, all checks pass.

---

## Phase 2: Baseline Kernel

### Step 2.1 — Implement `rowwise_softmax_smem`
Write `kernels/rowwise_softmax_smem.cu`:
- Three-pass algorithm (max → exp-sum → normalize)
- Pure shared-memory tree reduction for both max and sum
- Fixed block config: 256 threads per block, 4 rows per block. `THREADS_PER_BLOCK=256`, `ROWS_PER_BLOCK=4`, yielding 64 threads per row. Both kernels share this identical block size; this task permits exactly one core optimization (reduction strategy).
- FP16 I/O, FP32 internal
- Hardware `__expf()`
**Verify**: Code review against pseudocode in draft Appendix A.1.

### Step 2.2 — Compile baseline kernel
```bash
python scripts/compile.py --kernel rowwise_softmax_smem
```
**Verify**: `nvcc` exits 0, produces `kernels/rowwise_softmax_smem.so`.

### Step 2.3 — Validate baseline correctness
```bash
python scripts/validate.py --mode full-matrix \
    --kernels "rowwise_softmax_smem" \
    --shapes "64,64 256,128 1024,256 512,512 4096,512 1024,1024 256,4096 4096,4096" \
    --input-types "uniform large_magnitude outlier all_zero" \
    --atol 0.01 --rtol 0.001
```
**Verify**: All 32 tests pass. **Gate: do not proceed if any test fails.**

### Step 2.4 — Benchmark baseline kernel
```bash
python scripts/benchmark.py \
    --kernels "rowwise_softmax_smem" \
    --shapes "64,64 256,128 1024,256 512,512 4096,512 1024,1024 256,4096 4096,4096" \
    --warmup 10 --iters 100 \
    --output outputs/baseline_bench.json
```
**Verify**: `outputs/baseline_bench.json` contains 8 entries, all with valid `median_us > 0`.

---

## Phase 3: Candidate Kernel (Iteration 1)

### Step 3.1 — Implement `rowwise_softmax_warp_reduce`
Write `kernels/rowwise_softmax_warp_reduce.cu`:
- Copy baseline kernel verbatim, then replace shared-memory reduction steps with:
  - Warp-shuffle butterfly for intra-warp reduction (max and sum)
  - Shared-memory reduction of warp partials for cross-warp reduction
- **Do not change anything else**: same block size, same launch config, same I/O
**Verify**: `diff kernels/rowwise_softmax_smem.cu kernels/rowwise_softmax_warp_reduce.cu` shows only reduction-related lines changed (one-technique constraint check).

### Step 3.2 — Compile candidate kernel
```bash
python scripts/compile.py --kernel rowwise_softmax_warp_reduce
```
**Verify**: `nvcc` exits 0, produces `kernels/rowwise_softmax_warp_reduce.so`.

### Step 3.3 — Validate candidate correctness
```bash
python scripts/validate.py --mode full-matrix \
    --kernels "rowwise_softmax_warp_reduce" \
    --shapes "64,64 256,128 1024,256 512,512 4096,512 1024,1024 256,4096 4096,4096" \
    --input-types "uniform large_magnitude outlier all_zero" \
    --atol 0.01 --rtol 0.001
```
**Verify**: All 32 tests pass. **Gate: do not proceed if any test fails.**

### Step 3.4 — Benchmark candidate kernel
```bash
python scripts/benchmark.py \
    --kernels "rowwise_softmax_smem rowwise_softmax_warp_reduce" \
    --shapes "64,64 256,128 1024,256 512,512 4096,512 1024,1024 256,4096 4096,4096" \
    --warmup 10 --iters 100 \
    --output outputs/candidate_v1_bench.json
```
**Verify**: `outputs/candidate_v1_bench.json` contains 16 entries (8 shapes × 2 kernels).

### Step 3.5 — Evaluate against stop rule
Compare candidate vs. baseline median latencies per shape. Record best-so-far. Evaluate:
- Does candidate beat torch on any shape? → **Promote**. Write `outputs/promotion_report.md`. Done.
- Does candidate improve over baseline on any shape? → **Continue** to Phase 4.
- No improvement → **Record iteration**. If 3 consecutive no-improvement iterations → **Stop**. Write `outputs/experiment_conclusion.md`. Done.

---

## Phase 4: Iterate (up to 2 more iterations)

For each iteration (i = 2, 3):

### Step 4.i.1 — Refine candidate
Edit `kernels/rowwise_softmax_warp_reduce.cu` (or create `_v2`, `_v3`) with allowable refinements:
- Shared-memory padding to avoid bank conflicts
- Loop unroll (`#pragma unroll`)
- Register reuse improvements
- `__shfl_xor_sync` mask optimization

**One-technique constraint**: Only reduction-related code may change. Diff-check before proceeding.

### Step 4.i.2 — Compile, validate, benchmark
Repeat Steps 3.2–3.4 for the refined candidate.

### Step 4.i.3 — Evaluate stop rule
Same as Step 3.5. If still no improvement, increment the no-improvement counter. If counter reaches 3, stop.

---

## Phase 5: Finalize

### Step 5.1 — Archive results
```bash
# Copy all benchmark JSONs, candidate sources, and validation logs
cp outputs/*.json runs/
cp kernels/*.cu runs/
```

### Step 5.2 — Write final report
If promoted: `outputs/promotion_report.md`
If stopped: `outputs/experiment_conclusion.md`

Both reports include:
- Final benchmark table (all shapes, both kernels, torch)
- Per-shape speedup/delta vs baseline
- Per-shape speedup/delta vs torch
- Number of iterations attempted
- Key lessons learned

### Step 5.3 — Git tag
```bash
git tag -a experiment-complete -m "H200 FP16 rowwise softmax warp-reduce experiment"
```

---

## Shape Summary for Quick Reference

```
S1:   64 ×   64    (tiny)
S2:  256 ×  128    (small)
S3: 1024 ×  256    (moderate-narrow)
S4:  512 ×  512    (square-moderate)
S5: 4096 ×  512    (many-moderate)
S6: 1024 × 1024    (square-large)
S7:  256 × 4096    (wide — best case for warp-shuffle)
S8: 4096 × 4096    (large)
```

## Input Types for Quick Reference

```
T1: uniform          — torch.randn(M, N, dtype=float16)
T2: large_magnitude  — torch.randn(M, N, dtype=float16) * 10.0
T3: outlier          — uniform + one element per row = +50.0
T4: all_zero         — torch.zeros(M, N, dtype=float16)
```
