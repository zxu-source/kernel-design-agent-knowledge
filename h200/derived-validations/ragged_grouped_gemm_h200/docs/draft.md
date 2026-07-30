# Draft: BF16 Ragged Grouped GEMM on H200

## Scope and isolation

This is a new experiment under `ragged_grouped_gemm_h200/`.  It deliberately
does not modify the pre-existing `tasks/gemm/` FP16 Triton experiment or its
root-level documents.  Local WSL is the control plane; CUDA compilation,
correctness, benchmarking, and profiling are remote-H200-only through
`../scripts/h200-run.sh` / `../scripts/h200-raw.sh`.

Initial remote probe caveat: the raw H200 shell currently exposes `nvcc` and
two H200s, but not `conda`, `cmake`, or `ncu` on `PATH`; the existing
`h200-run.sh` consequently fails before it enters its documented
`metis_cuda12.9` environment.  This must be repaired or replaced by an
explicitly discovered environment path before formal CUDA/PyTorch/NCU evidence
is claimed.  No stale version information is treated as a current result.

## Operator understanding

For `E` experts the operator computes `C_e = A_e x B_e`, where `A_e` is
`[M_e,K]`, `B_e` is `[K,N]`, and `C_e` is `[M_e,N]`.  `K` and `N` are shared,
but `M_e` is ragged and may be zero.  The public API will use packed BF16 A/C
storage plus offsets, per-expert BF16 B storage, and device metadata.  Accumulation
is FP32.  Effective FLOPs are `sum_e(2*M_e*N*K)`; padded work is never counted
as useful FLOPs.

## Initial implementation plan (not implemented yet)

1. `candidate_00_baseline`: a CUDA C++ host launcher iterates nonempty experts
   and launches a safe BF16 Tensor-Core GEMM path per expert.  It is a genuine
   custom CUDA baseline, not a loop of `torch.matmul`.
2. `candidate_01_kernelwiki`: flatten each valid `(expert, m-tile, n-tile)` into
   a device work list.  A single persistent CUDA kernel consumes that list with
   a global atomic ticket, resolves packed offsets once per tile, and executes
   FP32-accumulated BF16 MMA tiles.  This removes per-expert launches and is
   intended to balance skewed expert loads.
3. If the portable MMA implementation is correct but not competitive, retain it
   as a failed/limited candidate and add a CUTLASS SM90 grouped-GEMM path using
   the same metadata contract.  TMA/WGMMA is a later, explicitly gated option:
   it requires CUTLASS availability, alignment validation, and a benchmarked
   benefit; it will not be claimed merely because the wiki mentions it.

## KernelWiki findings available before implementation

- `kernel-grouped-gemm` describes exactly this fixed-K/N, variable-M MoE
  structure and distinguishes contiguous packed offsets from padded masked
  layouts.
- CUTLASS PR 3091 provides a Hopper grouped example built from a persistent,
  group-aware scheduler and per-group TMA descriptor updates.  Its source notes
  BF16 support, FP32 accumulation, 16-byte contiguous-dimension alignment, and
  CTA M/N constraints.
- FlashInfer PR 1241 materializes group problem sizes, packed strides, and
  pointer arrays from an `m_indptr` offset array in a preparatory GPU kernel.
- DeepGEMM PR 304 exposes both contiguous and masked SM90 BF16 M-grouped paths
  and configures TMA descriptors, swizzles, and pipeline stages through a
  selected configuration.

These are observed facts, not a claim that their exact FP8/FP4 or SM100 code is
portable to this BF16 H200 task.  The traceable source-to-decision mapping is in
`kernelwiki_evidence.md`.

## Correctness plan

The harness will compare against FP32-reference `torch.matmul` (BF16 inputs)
for `E = 1,4,8,16,32,64`; uniform, skewed, sparse/empty, small-token, mixed,
and single-expert distributions; and at least `(1024,1024)`, `(4096,1024)`,
and `(4096,4096)` K/N pairs.  It will report max/mean absolute error, max
relative error, allclose, and NaN/Inf status.  Explicit guards cover `M_e=0`,
tails, nonaligned rows, range/overlap checks on offsets, and nondefault streams.

## Benchmark and profile plan

All candidates use the same generated inputs and CUDA-event timing: at least
20 warmups, 100 measured iterations, three trials, and median/min/mean/P90.
Rows include useful FLOPs only and separately state metadata-build cost.  The
comparison set is baseline, KernelWiki candidate, PyTorch loop, and (if built)
CUTLASS reference.  NCU permission will be probed first; a permission failure
will be recorded rather than substituted with fabricated metrics.

## Risks and fallback

- A hand-written WMMA kernel may be correct but inferior to CUTLASS on H200;
  that is an experimental result, not a reason to omit it.
- Persistent atomic scheduling can lose on tiny uniform workloads.  The plan
  will retain per-expert baseline and report these regressions.
- TMA/WGMMA requires suitable CUTLASS headers and alignment.  If unavailable,
  candidate_01 still tests the knowledge-derived flattened/persistent scheduling
  idea using standard CUDA MMA.
- Remote wrapper and shared-filesystem synchronization must be verified before
  formal H200 results are accepted.
