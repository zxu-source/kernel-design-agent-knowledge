---
id: hw-thread-block-cluster
title: "Thread-Block Clusters and Distributed Shared Memory (DSMEM)"
type: hardware
architectures: [sm90, sm100]
tags: [cluster, mbarrier]
confidence: source-reported
related: [hw-clc, hw-tma, technique-persistent-kernels]
sources: [doc-nvidia-tuning-guide]
aliases: [cluster, "thread block cluster", DSMEM, "distributed shared memory", cluster-launch-control]
blackwell_relevance: "Clusters (1-8 blocks, up to 16 non-portable) available on Hopper SM90 and unchanged on Blackwell SM100; DSMEM semantics identical. CLC dynamic scheduling builds on clusters."
h200_validation:
  date: '2026-07-20 (overnight)'
  evidence_file: data/crawl-runs/h200/hopper-cluster-dsmem-h200-results.md
  harness_dir: artifacts/kernels/hopper-cluster-dsmem/variants
  gpu: H200
  arch: sm90
  toolchain: "nvcc 13.0, -arch=sm_90"
  correctness: "PASS — clusters form (num_blocks()=4); DSMEM cross-block shared read/write via map_shared_rank works; broadcast output identical to global-bounce and to source (delta=0.0)"
  result: "record-negative for speedup — DSMEM 1-to-many broadcast 0.81x-0.83x SLOWER than a global-bounce buffer on H200 (both use the on-chip network/L2; DSMEM adds cluster-sync overhead)."
  scope: "Bulk broadcast pattern on H200; DSMEM still pays off for repeated peer-to-peer exchanges inside compute kernels."
---

## Overview

A **thread-block cluster** is a group of 1-8 (up to 16 non-portable) thread
blocks that are co-scheduled on adjacent SMs and can synchronize and share
memory directly. **Distributed Shared Memory (DSMEM)** lets a block in a cluster
read and write another block's shared memory via `cluster_group::map_shared_rank`,
forming a cluster-wide shared-memory address space without going through global
memory. Available on Hopper (SM90) and unchanged on Blackwell (SM100).

## How It Works

```cuda
namespace cg = cooperative_groups;
cg::cluster_group cluster = cg::this_cluster();
int rank = cluster.block_rank();           // 0..cluster_size-1
// map the calling block's shared pointer to peer `r`'s shared memory:
float* peer_sbuf = cluster.map_shared_rank(sbuf, r);
peer_sbuf[j] = value;                       // write into peer r's shared via DSMEM
cluster.sync();                             // REQUIRED before peer consumes/exits
```

Clusters are launched via `cudaLaunchKernelEx` with a
`cudaLaunchAttributeClusterDimension` attribute. `cluster.sync()` is a
cluster-wide barrier; cluster formation is confirmed by `cluster.num_blocks()`.

## H200 measured evidence (2026-07-20)

Validated on H200 (SM90), `nvcc -arch=sm_90`:

- **Cluster formation confirmed**: `cluster.num_blocks()` returns the configured
  size (4 for CLUSTER=4, 2 for CLUSTER=2).
- **DSMEM cross-block access works**: block 0 can WRITE into block 1's shared
  memory via `map_shared_rank`; the peer reads the value back correctly.
  **Critical correctness rule** (discovered empirically): a `cluster.sync()` MUST
  follow the DSMEM push before any peer block may consume the data or exit.
  Without it, peer blocks finish early and reclaim their shared memory while the
  sender is still writing into it, causing an illegal-access launch fault.
- **Broadcast correctness PASS**: a 4-block-per-cluster broadcast (block 0's
  tile pushed to 3 peers via DSMEM) produced output identical to a
  global-bounce-buffer path and to the source (max delta = 0.0).

**Latency — record-negative for broadcast speedup**: the DSMEM one-to-many
broadcast measured **0.81x-0.83x (slower)** than a global-bounce buffer, both at
L2-resident scale (17 MB) and at HBM-streaming scale (277 MB, past the ~50 MB
L2). For a bulk broadcast, both paths move data over the on-chip network /
through L2; DSMEM adds `cluster.sync()` overhead and cross-SM shared-write
latency without avoiding meaningful traffic, so it loses to the simpler global
buffer (which L2 serves efficiently).

**Takeaway:** clusters and DSMEM are **correct and usable** on H200, but DSMEM
is not a win for bulk one-to-many broadcast — a global buffer + L2 is faster.
DSMEM pays off for **repeated peer-to-peer exchanges inside compute kernels**
(e.g., cooperative reductions where blocks exchange partials many times without
a global intermediary). Full numbers:
[`data/crawl-runs/h200/hopper-cluster-dsmem-h200-results.md`](../../data/crawl-runs/h200/hopper-cluster-dsmem-h200-results.md).

## Related

- [clc](clc.md) — Cluster Launch Control dynamic scheduling within persistent kernels.
- [tma](tma.md) — TMA bulk async copy, often used together with clusters.
