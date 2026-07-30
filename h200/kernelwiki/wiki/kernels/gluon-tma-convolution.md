---
id: kernel-gluon-tma-convolution
title: "Gluon TMA Implicit-GEMM Convolution"
type: kernel
architectures: [sm100]
tags: [tma, tcgen05, tmem, warp-specialization, pipeline-stages, persistent-kernel]
confidence: source-reported
reproducibility: snippet
kernel_types: [gemm, fused-kernel]
languages: [triton, python]
related: [hw-tma, hw-tcgen05-mma, hw-tmem, technique-warp-specialization, technique-pipeline-stages, technique-persistent-kernels]
sources: [pr-triton-10030]
performance_claims: []
artifact_dir: artifacts/kernels/gluon-tma-convolution/variants
blackwell_relevance: "The upstream implementation is SM100-specific: it uses TMA/im2col, tcgen05 MMA, TMEM FP32 accumulation, and warp-specialized partitions."
---

## Overview

Triton PR #10030 expresses convolution forward, weight-gradient, and
input-gradient work as implicit GEMMs. The source separates load, MMA, and
epilogue work into warp-specialized partitions, uses a persistent scheduler to
assign tiles, and carries the accumulator through TMEM. This page records the
upstream structure; it has not been executed on the available SM90 H200.

## Pipeline Structure

```python
# Source-derived structural excerpt from PR #10030. The real kernel supplies
# TMA descriptors, mbarriers, and the partition argument object `p`.
@gluon.jit
def conv_pipeline(p):
    scheduler = PersistentTileScheduler.initialize(p.config.get_num_tiles())
    for tile_idx in range(scheduler.get_num_tiles()):
        prog = p.config.get_program(scheduler.get_tile_id(tile_idx))
        tma.async_load_im2col(p.in_desc, prog.input_coord, prog.im2col_offsets,
                              p.load_ready_bars.index(tile_idx), p.a_bufs.index(tile_idx))
        tma.async_load(p.weight_desc, prog.weight_coord,
                       p.load_ready_bars.index(tile_idx), p.b_bufs.index(tile_idx))
        mbarrier.wait(p.load_ready_bars.index(tile_idx), phase=0)
        tcgen05_mma(p.a_bufs.index(tile_idx), p.b_bufs.index(tile_idx),
                    p.acc_bufs.index(tile_idx), use_acc=False)
        tcgen05_commit(p.acc_ready_bars.index(tile_idx))
```

## Design Notes

- Host tensors use NHWC activations and OHWI filters; the logical GEMM maps
  output positions to M, output channels to N, and filter/channel reduction to K.
- TMA im2col supplies input tiles and handles convolution geometry. The upstream
  helpers pad channel dimensions or materialize aligned strides when required by
  TMA descriptors.
- The dgrad path decomposes stride-greater-than-one work into subproblems and
  uses split-K partials only when necessary.
- Upstream PR benchmarks are source-reported and are intentionally not copied
  here because they have not been reproduced in this workspace.
