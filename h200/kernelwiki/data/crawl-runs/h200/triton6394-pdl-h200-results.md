# Triton PR #6394 — Programmatic Dependent Launch (PDL) on H200

Date: 2026-07-20. Source-informed H200 microbenchmark (not a reproduction of
upstream tutorial 11). Hardware: NVIDIA H200, 132 SMs, compute capability 9.0.
Toolchain: Triton 3.6.0, PyTorch 2.11.0+cu130. The kernel and harness were
written from the merged PR's described API (`tl.extra.cuda.gdc_wait()` /
`gdc_launch_dependents()` plus the `launch_pdl` compile option); they are not
copied from the upstream tutorial.

## What was verified on H200

- The GDC intrinsics compile and **are emitted into PTX** as
  `griddepcontrol.wait` and `griddepcontrol.launch_dependents` when the kernel
  is launched with `launch_pdl=True` (verified by PTX dump).
- `launch_pdl=True` is the Triton 3.6.0 user-facing knob: it flips
  `metadata.launch_pdl` to `True`, which makes the backend driver set
  `CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION` on the launch.
  (`enable_pdl` is **not** a kwarg/decorator option in this build.)
- Correctness: a chain of N data-dependent elementwise kernels
  (`x = x*alpha + beta`) produces a result identical to the non-PDL chain and to
  the closed-form value `beta*(alpha^N - 1)/(alpha - 1)`; max abs delta = 0.0
  across all tested (BLOCK, N) configs.

## Method

A chain of N short, data-dependent Triton kernels over 1 M float32 elements.
PDL-on variant: every kernel launched with `launch_pdl=True`, calls
`gdc_wait()` at entry and `gdc_launch_dependents()` after its store, so the next
grid's ramp-up overlaps the current grid's store-drain/ramp-down. PDL-off
variant: same kernel body with the GDC paths compiled out. Timed with CUDA
events; reported value is the min of 10 trials after 5 warmup replays.

Two timing regimes:

1. **Python-dispatched** (`triton6394_pdl_chain.py`): each launch goes through
   the Python `kernel[grid](...)` path. CPU launch overhead (~7 us/launch)
   dominates and the GPU is idle between kernels, so there is no ramp-down left
   to overlap. Result: ~1.0x (0.99x–1.01x). This is a CPU-bound control, not a
   PDL failure.
2. **CUDA-graph replay** (`triton6394_pdl_graph.py`): the whole chain is
   captured into a CUDA graph and replayed, removing CPU dispatch from the
   measurement and isolating the GPU-side overlap PDL provides.

## Measured results (CUDA-graph replay, min of 10 trials)

| BLOCK | N (launches) | PDL-off (ms) | PDL-on (ms) | off us/launch | on us/launch | speedup | correctness delta |
|------:|-------------:|-------------:|------------:|--------------:|-------------:|--------:|------------------:|
|  1024 |          256 |       0.4875 |      0.4526 |         1.904 |        1.768 |  1.077x |                 0 |
|  1024 |         1024 |       1.9344 |      1.8120 |         1.889 |        1.770 |  1.068x |                 0 |
|  1024 |         4096 |       7.7448 |      7.2017 |         1.891 |        1.758 |  1.075x |                 0 |
|  2048 |          256 |       0.5036 |      0.4378 |         1.967 |        1.710 |  1.150x |                 0 |
|  2048 |         1024 |       2.0279 |      1.7208 |         1.980 |        1.680 |  1.178x |                 0 |
|  2048 |         4096 |       8.0451 |      7.1021 |         1.964 |        1.734 |  1.133x |                 0 |
|  4096 |          256 |       0.5209 |      0.4478 |         2.035 |        1.749 |  1.163x |                 0 |
|  4096 |         1024 |       2.0948 |      1.7902 |         2.046 |        1.748 |  1.170x |                 0 |
|  4096 |         4096 |       8.1828 |      7.1071 |         1.998 |        1.735 |  1.151x |                 0 |

## Conclusion (scoped to this benchmark)

- PDL on H200 / Triton 3.6.0 reduces per-launch GPU time of short back-to-back
  dependent kernels by a stable ~0.2 us (~1.9 us -> ~1.7 us), a **1.07x–1.18x
  wall-clock speedup** under CUDA-graph replay. The benefit grows with BLOCK
  size (larger kernels have more ramp overhead to overlap) and is consistent
  across chain length. This is consistent with the PR's "up to ~15% for
  back-to-back kernels" characterization.
- The speedup is **only visible when CPU launch overhead is removed** (graph
  capture/replay). Under naive Python dispatch the GPU is already idle between
  launches and PDL measures ~1.0x. Any production use must launch from a graph
  (or otherwise avoid CPU-bound dispatch) to realize the gain.
- No correctness regressions observed.

## Files

| file | role | sha256 |
|---|---|---|
| `triton6394_pdl_graph.py` | runnable harness (CUDA-graph timing, reported numbers) | `e8e3abb400d86c7823681cfa122739527f409063a1e8c5de0dd9a874b3c6cdbe` |
| `triton6394_pdl_chain.py` | runnable harness (Python-dispatch control, ~1.0x) | `3f7383905bc26cb8dd0c7ad247e77392ffc4c17c9382025eecce875542103a70` |
