# CUTLASS PR #2719 — SM90 Array TMA GEMM PDL on H200 (DEFERRED)

Date: 2026-07-20. Status: **DEFERRED** (unmanageable scope for a single slot;
the SM90 PDL technique it enables is validated separately on H200 — see below).

## What PR #2719 does

Adds `cutlass::arch::launch_dependent_grids()` / `wait_on_dependent_grids()`
calls into the SM90 array-TMA (grouped) warp-specialized GEMM kernels so they
participate in Programmatic Dependent Launch (PDL) for back-to-back grouped-GEMM
launches. Two C++ header files only.

## What was confirmed on the H200

- The relevant CUTLASS C++ headers ARE present on the H200, bundled inside
  vLLM's deep_gemm third-party at
  `.../vllm/third_party/deep_gemm/include/cutlass/`, including:
  - `cutlass/gemm/kernel/sm90_gemm_array_tma_warpspecialized_cooperative.hpp`
    (one of the two files PR #2719 modifies), and
  - `cutlass/arch/grid_dependency_control.h` (the PDL primitive header that
    provides `launch_dependent_grids` / `wait_on_dependent_grids`).
- `grid_dependency_control.h` gates PDL behind
  `CUTLASS_ENABLE_GDC_FOR_SM90` + `__CUDA_ARCH_FEAT_SM90_ALL` (i.e. compile with
  `-arch=sm_90a -DCUTLASS_ENABLE_GDC_FOR_SM90`). So the mechanism is buildable
  on H200.
- **However, the bundled CUTLASS snapshot predates PR #2719**: the bundled
  `sm90_gemm_array_tma_warpspecialized_cooperative.hpp` contains NO
  `launch_dependent` / `wait_on_dependent` / `GridDependency` references. The
  literal PR code is therefore not present to test; reproducing #2719 means
  re-applying its change on top of a full CUTLASS SM90 grouped-GEMM build.

## Compile attempt (genuine, documented)

A minimal CUTLASS SM90 TF32 GEMM was written
(`data/crawl-runs/h200/cutlass2719_sm90_gemm.cu`,
sha256 `1630267bf6703f3be930e785b01081ef2f4819136b4f81346bf8532eb235228d`)
using the canonical `CollectiveBuilder` + `GemmUniversalAdapter` pattern and
compiled with `nvcc -O3 -std=c++17 -arch=sm_90a` against the bundled headers
(include root resolved; cublas reference via the `nvidia/cu13` pip package). The
compile reached CUTLASS template instantiation and failed on **version-specific
API differences** in this bundled snapshot:

- `cutlass::layout::RowTensorOp` and `cutlass::epilogue::collective::EpilogueOpTensorOp`
  do not exist (newer/older layout and epilogue-op tags).
- `cutlass::gemm::collective::CollectiveBuilder` reports "too many arguments"
  (the mainloop builder signature differs from the CUTLASS version the recipe
  was written for — no trailing scheduler/epilogue passthrough in this form).
- `cutlass::gemm::device::GemmUniversalAdapter<...>` is an incomplete type
  because `cutlass::gemm::kernel::GemmUniversal<...>` in this version requires
  an explicit TileScheduler template parameter.

Each correction requires a slow (~minutes) CUTLASS SM90 compile, and no
ready-made SM90 GEMM instantiation exists in the installed Python tree
(deep_gemm is shipped pre-compiled) to mirror the exact type recipe. Bringing a
faithful standalone CUTLASS SM90 grouped-GEMM-with-PDL repro up against this
pre-#2719 snapshot therefore exceeds a single batch slot and carries high
version-mismatch compile risk.

## Why this is acceptable (the technique is already validated on H200)

PR #2719's contribution is purely *enabling* PDL on the SM90 array-TMA GEMM
path. The PDL **technique** on SM90 (Hopper) — launch-attribute
`CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION` plus
`griddepcontrol.wait` / `griddepcontrol.launch_dependents` overlapping ramp-down
and ramp-up of consecutive dependent kernels — is independently validated on
this same H200 in the Triton PR #6394 result:
[`triton6394-pdl-h200-results.md`](triton6394-pdl-h200-results.md)
(1.07x–1.18x wall-clock under CUDA-graph replay, correctness delta = 0.0). The
CUTLASS PR reuses the identical hardware mechanism via
`cutlass::arch::launch_dependent_grids` / `wait_on_dependent_grids`.

## Recommendation / follow-up

A future round could (a) re-attempt against a CUTLASS snapshot that includes
#2719, or (b) author the GEMM in the CuTe DSL (`cutlass.cute`), which exposes
`griddepcontrol_launch_dependents` and avoids the C++ template-signature
version coupling. Until then, this entry is DEFERRED with the technique-level
validation in place; no CUTLASS-specific speedup or correctness claim is made.

## Files

| file | role | sha256 | status |
|---|---|---|---|
| `cutlass2719_sm90_gemm.cu` | minimal CUTLASS SM90 TF32 GEMM compile attempt | `1630267bf6703f3be930e785b01081ef2f4819136b4f81346bf8532eb235228d` | does not compile against this bundled CUTLASS snapshot (version-specific API) |
