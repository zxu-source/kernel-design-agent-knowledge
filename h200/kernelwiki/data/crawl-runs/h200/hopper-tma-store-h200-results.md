# Hopper TMA STORE (cp.async.bulk.tensor shared->global) on H200

Date: 2026-07-20 (overnight). Companion to `hopper-tma-copy-h200-results.md`
(TMA LOAD). Each block fills a shared tile deterministically and writes it to
global either via TMA store (`cp.async.bulk.tensor.2d.global.shared` +
`commit_group`/`wait_group 0`) or via a naive per-thread store. Both produce
identical output; we compare the pure store bandwidth.

Hardware: NVIDIA H200, 132 SMs, cc 9.0. `nvcc -O3 -std=c++17 -arch=sm_90a -lcuda`.
Timing: CUDA events, min of 20 trials after warmup. 2048x2048 float tensor,
128x128 tiles, 16x16 grid.

## Correctness — PASS

TMA-stored output is **exactly** equal to the naive-stored output
(`max |tma - naive| = 0.0`). The TMA store path is functionally correct.

## Bandwidth — TMA store is SLOWER than naive (record-negative)

| kernel | min (ms) | bandwidth |
|---|---:|---:|
| TMA store (`cp.async.bulk.tensor.2d ... bulk_group`) | 0.0346 | 485 GB/s |
| naive per-thread store | 0.0100 | 1680 GB/s |

TMA store measured only **485 GB/s**, ~3.5x SLOWER than the naive per-thread
store (1680 GB/s) for this simple shared->global tile write.

## Conclusion (scoped) — TMA's benefit is asymmetric across load vs store

- TMA **load** (`cp.async.bulk.tensor` global->shared) is a clear win on H200:
  ~1.96x the bandwidth of a naive copy (1565 vs 797 GB/s) — see the companion
  TMA-load result. TMA offloads address generation + bulk fetch.
- TMA **store** (shared->global) is a **loss** for simple writes: naive
  per-thread stores are already optimal on H200 (the hardware coalesces and
  write-back-caches streaming stores efficiently at 1680 GB/s), and TMA's
  `bulk_group` / `commit_group` / `wait_group` machinery adds overhead that the
  simple store doesn't need. TMA store pays off only for specialized cases
  (TMA-bulk reductions, multicast, or stores integrated into an async
  software-pipelined mainloop), not for plain shared->global writes.
- PTX detail that caused an initial correctness failure (recorded for reuse):
  the 2D coordinate vector must be `{x, y}` (fast, slow); a swap silently
  produces wrong (not faulting) data — the same gotcha as the TMA load.

## File

| file | role | sha256 |
|---|---|---|
| `hopper_tma_store.cu` | runnable harness (TMA store vs naive store) | `3d94e08aecdc2575243f589162622f3f8fa28bf94c0548f518d9a9e49caf7e83` |
