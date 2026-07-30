# Hopper Thread-Block Cluster + DSMEM Broadcast on H200

Date: 2026-07-20 (overnight). Source-informed H200 microbenchmark. Validates
Hopper thread-block clusters (Cluster Launch Control's cluster formation, not
CLC dynamic scheduling) and Distributed Shared Memory (DSMEM) on H200 (SM90),
and characterizes whether a DSMEM one-to-many broadcast beats a global-bounce
buffer for inter-block data exchange.

Hardware: NVIDIA H200, 132 SMs, cc 9.0. `nvcc -O3 -std=c++17 -arch=sm_90`.
Clusters launched via `cudaLaunchKernelEx` with `cudaLaunchAttributeClusterDimension`.
DSMEM via `cooperative_groups::cluster_group::map_shared_rank`. Timing: CUDA
events, min of 20 trials after warmup.

## What was validated on H200

- **Cluster formation**: `cooperative_groups::this_cluster().num_blocks()` reports
  the configured cluster size (verified = 4 for CLUSTER=4, = 2 for CLUSTER=2) —
  clusters form correctly on H200.
- **DSMEM cross-block shared access is functional**: a 2-block probe confirmed
  block 0 can WRITE into block 1's shared memory via `map_shared_rank` (block 1
  read back the value 99 correctly, no fault). Key correctness requirement
  discovered: a `cluster.sync()` MUST follow the DSMEM push, otherwise peer
  blocks exit and reclaim their shared memory while the sender is still writing
  into it -> illegal-access launch fault.
- **Correctness PASS**: a 4-block-per-cluster broadcast (block 0's 8 KB tile
  pushed to 3 peers via DSMEM vs the same via a global-bounce buffer) produced
  output **identical** to the global-bounce path and to the source
  (`max |dsmem - global| = 0.0`, `max |dsmem - src| = 0.0`).

## Latency — DSMEM broadcast is SLOWER than global bounce (record-negative)

| config | total data | global bounce (ms) | DSMEM (ms) | global/dsmem |
|---|---:|---:|---:|---:|
| CLUSTER=4, TILE=8KB,  528 cl | ~17 MB out | 0.0153 | 0.0190 | 0.81x |
| CLUSTER=4, TILE=32KB, 2112 cl | ~277 MB out (past L2) | 0.1458 | 0.1756 | 0.83x |

DSMEM fan-out measured **0.81x-0.83x (slower)** than a global-bounce buffer,
both at L2-resident scale and at HBM-streaming scale. For a one-to-many
broadcast, both paths move the data over the on-chip network / through L2; DSMEM
adds `cluster.sync()` overhead and cross-SM shared-write latency without
avoiding meaningful traffic, so it loses to the simpler global bounce (which the
L2 serves efficiently, and which at scale is bandwidth-saturated either way).

## Conclusion (scoped)

- Hopper cluster launch and DSMEM are **functionally validated on H200**
  (clusters form; cross-block shared read/write works; broadcast output correct).
- The DSMEM **speedup** claim does not hold for one-to-many broadcast on H200
  (0.81x-0.83x vs global bounce). DSMEM's benefit is for **repeated peer-to-peer
  exchanges inside compute kernels** (e.g., cooperative reductions where blocks
  exchange partial results many times without a global intermediary), not for a
  bulk broadcast where a global buffer + L2 is competitive. No broadcast
  speedup is claimed.
- Correctness requirement (recorded for reuse): always `cluster.sync()` after a
  DSMEM write before any peer may consume the data or exit, to avoid the sender
  writing into reclaimed shared memory.

## File

| file | role | sha256 |
|---|---|---|
| `hopper_cluster_dsmem.cu` | runnable harness (cluster DSMEM broadcast vs global bounce) | `f2b5edd31ae2dfe8dbd7f1c4dfb3e9b3bc0643cc138a7cbea81bad6183be8b9f` |
