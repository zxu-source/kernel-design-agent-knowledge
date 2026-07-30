# Hopper TMA (cp.async.bulk.tensor) 2D Tile Copy on H200

Date: 2026-07-20 (overnight). Source-informed H200 microbenchmark. Exercises the
H200's Tensor Memory Accelerator via `CUtensorMap` + `cp.async.bulk.tensor.2d`
+ mbarrier to copy a 2D tile from global to shared, then shared to global.
Validated for correctness and bandwidth vs a naive per-thread copy.

Hardware: NVIDIA H200, 132 SMs, cc 9.0. `nvcc -O3 -std=c++17 -arch=sm_90a -lcuda`.
TMA requires sm_90a and the CUDA driver API (`cuTensorMapEncodeTiled`, linked via
`-lcuda`). Timing: CUDA events, min of 20 trials after warmup.

## What was validated on H200

- The `CUtensorMap` is built with `cuTensorMapEncodeTiled` (2D, FLOAT32,
  globalDim in elements, globalStride in bytes, boxDim in elements, swizzle
  NONE) and copied to device memory; TMA reads the descriptor from generic memory.
- The kernel initializes a shared `mbarrier` (count=1), a single producer thread
  issues `cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes`
  and `mbarrier.arrive.expect_tx` with the byte count, then all threads
  `mbarrier.try_wait.parity(0)` until the transfer completes.
- **Correctness PASS**: TMA-copied output is **exactly** equal to the source
  (`max |tma - src| = 0.0`) and identical to the naive copy
  (`max |tma - naive| = 0.0`).

## Bandwidth — TMA ~2x faster than naive per-thread copy

| kernel | min (ms) | bandwidth |
|---|---:|---:|
| TMA (`cp.async.bulk.tensor.2d`) | 0.0107 | 1565 GB/s |
| naive (per-thread global->shared->global) | 0.0211 | 797 GB/s |

Config: 2048x2048 float tensor (16.8 MB), 128x128 tiles, 16x16 grid, 256
threads/block. TMA measured **1.96x** the bandwidth of the naive per-thread copy
(1565 vs 797 GB/s). TMA's advantage is hardware address generation + bulk
transfer that offloads the per-thread load/store instruction overhead, so the
copy saturates closer to the memory-system throughput.

## Conclusion (scoped)

- TMA (`CUtensorMap` + `cp.async.bulk.tensor.2d` + mbarrier) is **functionally
  validated on H200**: exact-copy correctness, and ~**2x the effective bandwidth**
  of a naive per-thread global->shared copy for this 2D tile pattern.
- This is a clean positive for the TMA primitive on Hopper. Two PTX details that
  caused initial failures (recorded for reuse): (1) the 2D coordinate vector must
  be `{dim0=x, dim1=y}` (fast, slow), not swapped; (2) the launch must link
  `-lcuda` for `cuTensorMapEncodeTiled`.
- Scope: a single global->shared->global tile copy at one tile/box size; a
  TMA-vs-`cp.async` (non-tensor) comparison and larger-box sweeps are left as
  refinements.

## File

| file | role | sha256 |
|---|---|---|
| `hopper_tma_copy.cu` | runnable harness (TMA 2D tile copy vs naive) | `6bae9f51de7c0299821eb727e3e5112b2d1b41b53e1589bcfed4015a7f8c6d5b` |
