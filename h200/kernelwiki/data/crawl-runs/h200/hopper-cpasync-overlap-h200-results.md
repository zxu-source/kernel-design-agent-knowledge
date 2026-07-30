# Hopper cp.async Multi-Stage Copy/Compute Overlap on H200 (record-negative)

Date: 2026-07-20 (overnight). Source-informed H200 microbenchmark. Tests whether
explicit multi-stage async-copy/compute overlap via `__pipeline_memcpy_async`
(cp.async + commit_group/wait_group) beats straightforward global loads for a
tiled global->shared->compute->global workload on H200 (SM90). This directly
extends Codex's earlier negative control (`cuda::memcpy_async` copy-only ->
0.957x, no compute to overlap) by adding real overlapped compute.

Hardware: NVIDIA H200, 132 SMs, cc 9.0. Compiled with `nvcc -O3 -std=c++17
-arch=sm_90`. Workload: 1056 blocks x 16 tiles/block, TILE=4096 floats, 256
threads/block, STAGES=3 circular shared buffers, 277 MB total. Per-tile compute
is register-only (`y = y*y + x` x REPS, single global store) so REPS adds pure
compute (memory-bound at low REPS, compute-bound at high REPS). Timing: CUDA
events, min of 10 trials after warmup. cp.async copy at 16-byte (float4)
granularity.

## Correctness — PASS

The pipelined kernel produced output **identical** to the serial kernel
(`max |serial - pipelined| = 0.0`) for every REPS. The async-copy pipeline is
functionally correct.

## Latency — pipelining is SLOWER for this pattern (record-negative)

| REPS/element | serial min (ms) | pipelined min (ms) | speedup (serial/pipelined) |
|--:|--:|--:|--:|
|   8 | 0.290 | 0.378 | 0.77x |
|  32 | 0.439 | 0.577 | 0.76x |
| 128 | 1.403 | 1.519 | 0.92x |
| 512 | 5.113 | 5.255 | 0.97x |

Explicit cp.async multi-stage pipelining measured **0.77x-0.97x** (i.e., slower)
versus straightforward global loads + `__syncthreads` + compute across the full
memory-bound to compute-bound range. This is a genuine negative for this
workload class:

- The serial pattern (`s[j] = in[global]; __syncthreads(); compute(s)`) already
  benefits from the hardware's automatic global-load latency hiding (the GPU
  overlaps independent in-flight global loads with compute via its load/store
  units and instruction scheduler), so there is little global latency left for
  explicit pipelining to hide.
- The cp.async pipeline bookkeeping (`__pipeline_commit` /
  `__pipeline_wait_prior` per tile, the 3-stage circular shared buffer, and the
  extra `__syncthreads` barriers) adds a fixed ~0.1-0.15 ms cost that exceeds
  the overlap benefit for these tile/stage sizes. Increasing REPS into the
  compute-bound regime (REPS=512) closes the gap toward ~0.97x but never crosses
  1.0x.

## Conclusion (scoped)

For a simple tiled elementwise **global->shared->compute->global** workload on
H200, explicit cp.async multi-stage copy/compute overlap does **not** improve
over straightforward global loads (0.77x-0.97x). This refines Codex's earlier
negative (`cuda::memcpy_async` copy-only -> 0.957x): adding real overlapped
compute still does not make the simple cp.async pipeline win. The cp.async/TMA
overlap technique pays off in **producer/consumer warp-specialized** kernels
(CUTLASS / Triton warp-specialized GEMM, where a dedicated producer warpgroup
issues TMA while a consumer warpgroup runs MMA, and the structure prevents the
hardware from auto-overlapping) — not in simple elementwise pipelines where the
hardware already hides global-load latency. No speedup is claimed for this
pattern class.

## File

| file | role | sha256 |
|---|---|---|
| `hopper_cpasync_overlap.cu` | runnable harness (cp.async multi-stage vs serial global loads) | `8e2ca35aea712e2397a1f7191ba4ed198d7d336d63d12ae9432c793798f5fe3c` |
