---
id: technique-persistent-kernels
title: Persistent Kernels with CLC
type: technique
architectures:
- sm100
tags:
- persistent-kernel
- clc
- tile-scheduling
confidence: source-reported
reproducibility: snippet
prerequisites:
- hw-clc
related:
- hw-clc
- technique-tile-scheduling
- technique-persistent-matmul-resource-budgeting
- pattern-tail-effect
sources:
- doc-nvidia-tuning-guide
- blog-tcgen05-tutorial
- doc-cutlass-blackwell
- pr-triton-6394
- pr-triton-9248
- pr-triton-9279
artifact_dir: artifacts/kernels/persistent-kernels
---

## Overview

Persistent kernels launch exactly as many CTAs as SMs, and each CTA processes multiple output tiles in a loop rather than exiting after one tile. On Blackwell, the CLC (Cluster Launch Control) hardware unit replaces software-based tile scheduling with a hardware-assisted mechanism. Each CTA queries the CLC for its next tile assignment and can cancel itself when no work remains, using the `try_cancel` pattern.

## CLC Loop Pattern

The core persistent kernel loop on Blackwell uses CLC to dynamically assign tiles:

```cuda
// Persistent kernel with CLC tile scheduling (Blackwell SM100)
__global__ void __launch_bounds__(512)
persistent_gemm_clc(const __grid_constant__ GemmParams params)
{
    // CLC-managed persistent loop: each CTA processes multiple tiles
    while (true) {
        // Query CLC for next tile assignment
        // Returns tile coordinates (tile_m, tile_n) or signals termination
        TileCoord tile;
        bool has_work = clc_try_get_tile(&tile);

        if (!has_work) {
            // No more tiles to process -- CTA exits
            // clc_try_cancel atomically checks if all tiles are done
            if (clc_try_cancel()) {
                return;  // CTA terminates
            }
            continue;  // Race condition: another CTA may have generated work
        }

        // Standard GEMM tile computation
        int tile_m = tile.m;
        int tile_n = tile.n;

        // TMA producer loads A[tile_m, :] and B[:, tile_n] tiles
        // MMA consumer accumulates K-dimension
        // Epilogue writes C[tile_m, tile_n]
        compute_gemm_tile(params, tile_m, tile_n);
    }
}
```

At the PTX level, the CLC interaction uses dedicated instructions:

```ptx
// CLC tile acquisition in PTX
// clusterctl.try_cancel cancels the CTA if no work remains
.reg .pred  %has_work;
.reg .b32   %tile_m, %tile_n;

TILE_LOOP:
    // Attempt to get next tile from CLC hardware scheduler
    clusterctl.query.async.shared  [%smem_tile_desc];
    clusterctl.wait;

    // Check if valid tile was assigned
    ld.shared.b32  %has_work, [%smem_tile_desc + 0];
    @!%has_work bra TRY_CANCEL;

    // Extract tile coordinates
    ld.shared.b32  %tile_m, [%smem_tile_desc + 4];
    ld.shared.b32  %tile_n, [%smem_tile_desc + 8];

    // ... compute tile ...

    bra TILE_LOOP;

TRY_CANCEL:
    // Atomically try to cancel this CTA
    clusterctl.try_cancel  %cancelled;
    @!%cancelled bra TILE_LOOP;   // Another CTA may have pushed work
    ret;
```

## Comparison: CLC vs Static Stride (Hopper)

On Hopper (SM90), persistent kernels use a static stride pattern where each CTA computes tiles at fixed intervals:

```cuda
// Hopper-style static stride persistent kernel
__global__ void hopper_persistent_gemm(GemmParams params)
{
    int cta_id = blockIdx.x;
    int total_ctas = gridDim.x;
    int total_tiles = params.num_tiles_m * params.num_tiles_n;

    // Static stride: CTA i handles tiles i, i+total_ctas, i+2*total_ctas, ...
    for (int tile_idx = cta_id; tile_idx < total_tiles; tile_idx += total_ctas) {
        int tile_m = tile_idx / params.num_tiles_n;
        int tile_n = tile_idx % params.num_tiles_n;
        compute_gemm_tile(params, tile_m, tile_n);
    }
}
```

| Aspect | Hopper Static Stride | Blackwell CLC |
|--------|---------------------|---------------|
| Scheduling | Software loop with fixed stride | Hardware CLC unit assigns tiles |
| Load balancing | Fixed; uneven if tile costs vary | Dynamic; CLC rebalances automatically |
| Tail effect | Last wave may have partial occupancy | CLC minimizes by giving fast CTAs more tiles |
| Launch overhead | Grid launch for each new problem | CLC can chain multiple problems |
| Termination | Implicit when loop ends | Explicit `try_cancel` |
| L2 locality | Depends on stride pattern | CLC can apply swizzled raster |

## CUTLASS PersistentTileSchedulerSm100

CUTLASS 4.5.0 provides `PersistentTileSchedulerSm100` that wraps the CLC hardware:

```cuda
// CUTLASS SM100 persistent tile scheduler (simplified)
template <class TileShape>
struct PersistentTileSchedulerSm100 {

    // Initialize the CLC with the problem geometry
    CUTLASS_DEVICE static void init(
        dim3 problem_tiles,
        void* clc_smem_buffer)
    {
        if (threadIdx.x == 0) {
            // Program CLC with total tile count and scheduling policy
            clc_init(clc_smem_buffer,
                     problem_tiles.x,  // tiles along M
                     problem_tiles.y,  // tiles along N
                     ClcPolicy::SwizzledRaster);
        }
        __syncthreads();
    }

    // Shared storage for CTA-wide CLC result broadcast
    // __shfl_sync is warp-local and cannot reach warps 1-15.
    struct SharedClcState {
        int tile_m, tile_n;
        int valid;       // 1 = got tile, 0 = no more work
        int cancelled;
    };

    // Get next tile assignment from CLC
    CUTLASS_DEVICE static bool get_next_tile(
        void* clc_smem_buffer,
        SharedClcState& shared_clc,
        int& tile_m,
        int& tile_n)
    {
        if (threadIdx.x == 0) {
            int m, n;
            bool v = clc_query_tile(clc_smem_buffer, m, n);
            shared_clc.tile_m = m;
            shared_clc.tile_n = n;
            shared_clc.valid  = v ? 1 : 0;
        }
        __syncthreads();  // All warps see the result
        tile_m = shared_clc.tile_m;
        tile_n = shared_clc.tile_n;
        return shared_clc.valid != 0;
    }

    // Try to cancel the CTA when no more work
    CUTLASS_DEVICE static bool try_cancel(
        void* clc_smem_buffer,
        SharedClcState& shared_clc)
    {
        if (threadIdx.x == 0) {
            shared_clc.cancelled = clc_try_cancel(clc_smem_buffer) ? 1 : 0;
        }
        __syncthreads();
        return shared_clc.cancelled != 0;
    }
};
```

## Performance Impact

The tcgen05-tutorial progression demonstrates the impact of persistent kernels:

```
Without persistence (static grid):  940 TFLOPS  (62% of peak)
With CLC persistent scheduling:    1476 TFLOPS  (98% of cuBLAS)
```

The 57% improvement comes from:
1. **Eliminated tail effect**: CLC dynamically assigns tiles, so fast-completing CTAs absorb extra work rather than sitting idle while the last wave finishes.
2. **Reduced launch overhead**: A single kernel launch covers all tiles; no need to re-launch grids.
3. **Better L2 cache utilization**: CLC can apply a swizzled raster pattern that improves spatial locality across neighboring tiles.

## PDL/GDC Kernel Overlap (Hopper/Blackwell)

Programmatic Dependent Launch (PDL) with Grid Dependency Control (GDC) enables consecutive kernels in the same CUDA stream to overlap their ramp-down and ramp-up periods, hiding prologue latencies. Unlike CLC (which schedules tiles within a single persistent kernel), PDL orchestrates overlap between distinct kernel launches.

### GDC API Pattern (Triton)

From PR #6394 (Triton PDL support, merged 2025-04-29):

```python
@triton.jit
def kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr,
           USE_GDC: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    if USE_GDC:
        # Wait for ALL CTAs in the prior kernel to complete before loading.
        # Ensures memory writes from prior kernel are visible.
        tl.extra.cuda.gdc_wait()

    x = tl.load(x_ptr + offsets)
    y = tl.load(y_ptr + offsets)

    if USE_GDC:
        # Signal that dependent kernels can begin launching.
        # Provides no memory ordering — only launch scheduling.
        tl.extra.cuda.gdc_launch_dependents()

    tl.store(output_ptr + offsets, x + y)
```

Key design points:
- **`gdc_wait()`**: Memory-ordering barrier — guarantees all CTAs in the prior
  kernel have completed before this CTA proceeds. Any memory written by the
  prior kernel is visible after this point.
- **`gdc_launch_dependents()`**: Launch scheduler hint — the runtime may begin
  launching the next kernel once all CTAs have issued this (or exited).
- **Conservative approach**: Place `gdc_wait()` before any `tl.load` to
  conservatively assume the prior kernel may write to any memory location.
- **Composable with persistent kernels**: PDL overlaps the prologue/epilogue of
  distinct kernel launches; CLC (persistent kernels) handles tile scheduling
  within a single launch. Both patterns address latency hiding at different
  granularities.

### Performance Evidence

- **Vector-add microbenchmark**: Up to 15% speedup (Triton PR #6394)
- **Back-to-back LLM layers on GB200 NVL72**: Up to 33% benefit (Nvidia
  GTC 2025 session S72503)
- PDL benefits increase with kernel count (amortized launch overhead) and
  decrease with per-kernel duration (long kernels already dominate overhead)

## When to Use

- **Large GEMM problems**: Persistent kernels are most beneficial when the number of output tiles exceeds the SM count by at least 2-3x.
- **Grouped GEMMs / MoE**: CLC can chain multiple problem instances, eliminating inter-kernel launch gaps.
- **Back-to-back kernel chains**: PDL/GDC overlaps consecutive kernel
  launches; applies when the workload is a sequence of distinct kernels
  (e.g., transformer layer = QKV proj → attention → output proj → FFN).
- **Workloads with uneven tile cost**: CLC's dynamic scheduling naturally handles variable-cost tiles (e.g., triangular attention masks).

## Hopper High-Occupancy Persistent Matmul

On Hopper (SM90), Blackwell's CLC hardware is absent. Instead, Triton achieves
high occupancy by allowing **multiple CTAs to co-reside on a single SM**. The
key insight (from PR #9248 and #9279): when occupancy > 1, the hardware warp
scheduler can fill tensor-core bubbles while one CTA runs its epilogue, achieving
an effect similar to ping-pong scheduling without explicit software coordination.

### maxnreg Formula (PR #9248)

To fit N CTAs per SM, each CTA must use at most `1/N` of the SM's registers:

```python
occupancy_target = 16 // num_warps  # e.g., num_warps=4 → 4 CTAs per SM
reg_per_sm = 64 * 1024    # 64 KB registers per SM
threads_per_warp = 32
max_reg_per_thread = 256  # hardware limit
maxnreg = reg_per_sm // (num_warps * threads_per_warp * occupancy_target)
maxnreg = min(max_reg_per_thread, maxnreg)
```

This is passed via `target_kernel_kwargs=dict(maxnreg=maxnreg)`, which tells
the PTX compiler to limit register allocation. A lower `maxnreg` may cause
register spills but enables more CTAs per SM.

### SMEM Division by Occupancy (PR #9248)

Shared memory must also be divided:
```python
smem_capacity //= occupancy_target  # split among co-resident CTAs
```
If `occupancy_target=2`, each CTA gets only 114 KB (vs 228 KB). This constrains
tile sizes — larger tiles that use 1 CTA/SM may outperform smaller tiles with
2 CTAs/SM if the tile size reduction hurts arithmetic intensity.

### Epilogue/Prologue Overlap vs Occupancy (PR #9279)

A counter-intuitive finding: when occupancy > 1, **removing** explicit
epilogue-prologue overlap (the "flatten" optimization) actually improved
performance by 150 GBps (2650→2800 GBps) on H200. The hypothesis: the warp
scheduler already hides epilogue latency by switching to the other co-resident
CTA, making the software-level overlap redundant and sometimes harmful (extra
register pressure from the overlap mechanism).

**Guideline:** When using occupancy_target > 1 on Hopper, benchmark with and
without epilogue/prologue overlap. The occupancy-based hardware scheduling
may outperform explicit software pipelining of the epilogue.

### Performance Evidence

| PR | Technique | Improvement | Hardware |
|---|---|---|---|
| #9248 | occupancy_target + maxnreg | 2290→2640 GBps (15%) | H200, bf16×mxfp4 MoE |
| #9279 | disable epilogue overlap with occupancy | 2650→2800 GBps (5.7%) | H200, mixed precision matmul |

## Caveats

- CLC is SM100-only; Hopper kernels must use software-based scheduling.
- The `try_cancel` pattern introduces a potential race that must be handled with a retry loop.
- For very small problems (fewer tiles than SMs), CLC overhead may not justify the complexity. A simple single-wave grid launch suffices.

## Full Reference Implementation

Verbatim upstream code lives in [`artifacts/kernels/persistent-kernels/full/`](../../artifacts/kernels/persistent-kernels/full/); labeled derived variants (each with the required `// provenance: derived from ...; not upstream code` header) live in [`artifacts/kernels/persistent-kernels/variants/`](../../artifacts/kernels/persistent-kernels/variants/). Every file's SHA-256 and upstream-pinning metadata is in `PROVENANCE.yaml` inside each bundle.

Query via:

```bash
python3 scripts/get_page.py technique-persistent-kernels --include-code
```
