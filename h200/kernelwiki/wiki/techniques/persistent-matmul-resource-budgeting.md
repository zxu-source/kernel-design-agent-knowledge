---
id: technique-persistent-matmul-resource-budgeting
title: "Persistent FP32 Matmul Shared Memory Budgeting"
type: technique
architectures:
- sm90
tags:
- persistent-kernel
- pipeline-stages
- shared-memory-optimization
- gemm
confidence: source-reported
reproducibility: snippet
prerequisites:
- technique-persistent-kernels
- technique-pipeline-stages
related:
- technique-persistent-kernels
- technique-pipeline-stages
- technique-register-budgeting
sources:
- pr-triton-9393
- pr-triton-9967
evidence_basis:
- evidence_type: upstream-code
  source_id: pr-triton-9393
  description: >
    Merged Triton PR introducing three concrete SMEM budgeting heuristics
    for persistent FP32 matmul kernels: 32 KB overhead reservation, pipeline
    stage cap at 3, and small-matrix persistent disable.  Also fixes TF32
    rounding for non-TMA FP32 inputs (cvt.rna → cvt.rn on SM90+).  No
    public benchmark data accompanies the PR.
- evidence_type: h200-measured
  source_id: pr-triton-9393
  description: >
    H200 (SM90) measurement on 2026-07-20 with Triton 3.6.0 (see h200_validation):
    a persistent FP32/TF32 matmul (num_stages=3) produced output bit-identical to
    a static-grid variant sharing the same tiles/stages (max delta = 0.0) and
    matched the PyTorch TF32 reference. Persistence alone gave no latency gain
    (0.92x-1.00x), consistent with the PR making no performance claim.
h200_validation:
  date: '2026-07-20'
  source_id: pr-triton-9393
  evidence_file: data/crawl-runs/h200/triton9393-persistent-matmul-h200-results.md
  gpu: H200
  arch: sm90
  toolchain: "Triton 3.6.0, PyTorch 2.11.0+cu130"
  correctness: "PASS — persistent FP32/TF32 matmul bit-identical to static-grid variant (max delta = 0.0) and matches TF32 reference"
  result: "Persistence isolated (fixed grid + round-robin tiles, same tiles/stages as static) gave 0.92x-1.00x — no speedup; production gains need accompanying tile-ordering/TMA, not persistence alone."
  scope: "Minimal persistent vs static-grid FP32/TF32 matmul on H200; correctness validated, no latency improvement observed or claimed."
blackwell_relevance: >
  The SMEM budgeting heuristics (fixed overhead reservation, stage capping,
  small-problem guard) are directly applicable to Blackwell (SM100) persistent
  kernels, though CLC hardware scheduling on SM100 may reduce the fixed-overhead
  profile. The TF32 rounding fix (cvt.rn) applies to SM90+ including SM100.
version_sensitive:
  id: vs-triton-persistent-fp32-smem
  tool: triton
  claim_valid_for:
  - "Triton persistent matmul: FP32 SMEM headroom = 32 KB, num_stages ≤ 3, disable when m*n*k < 131072"
  last_verified_release: null
  last_verified_at: null
---

## Overview

When enabling persistent scheduling for FP32 matrix multiplication kernels,
the simple per-stage tile-size model underestimates shared memory consumption.
Persistent kernels carry additional SMEM overhead — mbarrier state, TMA
descriptor tables, tile-scheduling metadata — that is not proportional to
tile count and becomes significant when FP32's 4-byte elements push the SMEM
budget closer to capacity.

This page documents three concrete heuristics extracted from the Triton
persistent matmul implementation (PR #9393, merged into `triton-lang/triton`),
plus a TF32 rounding precision fix for the non-TMA data path.

**All heuristics below are supported by merged upstream code. No public
benchmark data accompanies the source PR. Treat as source-reported
implementation patterns, not performance-validated recommendations.**

## Heuristic 1: Reserve 32 KB SMEM Headroom for Persistent FP32

### The Problem

The per-stage SMEM budget model computes:

```
num_stages = smem_capacity / stage_size
```

where `stage_size` accounts for A-tile and B-tile buffers only.  For
persistent FP32 kernels, the actual SMEM footprint exceeds this model because
persistent scheduling requires additional fixed-cost allocations:

- **mbarrier objects** for pipeline synchronization (one per stage × two
  directions: load-complete and compute-complete)
- **TMA descriptor tables** (tensor map descriptors for A and B global
  memory regions)
- **Tile scheduling metadata** (work queues, tile counters, CLC or
  software-scheduler state)

### The Fix

Subtract a flat 32 KB from the available SMEM capacity before computing the
stage budget:

```python
# Triton opt_flags_nvidia.py (simplified from PR #9393)
if is_persistent and (lhs_dtype == FP32 or rhs_dtype == FP32):
    smem_capacity -= 32 * 1024  # metadata / barriers / TMA state
smem_capacity = max(smem_capacity, 0)
```

### How to Apply

In your own persistent kernel scheduler, reserve a fixed overhead allowance
before dividing the remaining SMEM among pipeline stages.  The 32 KB value is
Triton-specific and may differ by framework, but the principle — fixed
overhead is not captured by per-stage sizing — is universal.

### What Is Directly Supported by the Diff

- The exact 32 KB subtraction and the comment citing "metadata/barriers/TMA state."
- The guard applies only when `is_persistent` AND at least one operand is FP32.

### What Is Inferred (Not in the Diff)

- The exact breakdown of the 32 KB (how much for barriers vs. descriptors vs.
  scheduler state).
- Whether 32 KB is the correct value for non-Triton frameworks or for
  Blackwell (SM100).  The code target is Hopper (SM90).

## Heuristic 2: Cap Pipeline Stages at 3 for Persistent FP32

### The Problem

Even after reserving the 32 KB fixed overhead, persistent FP32 kernels can
fail at launch time with shared-memory out-of-resource (OOR) errors.  The
remaining headroom after the per-stage budget may be insufficient when the
scheduler allocates memory for runtime structures not visible in the static
tile-size calculation.

### The Fix

Clamp the computed stage count to at most 3 (instead of the default maximum
of 4):

```python
# Triton opt_flags_nvidia.py (simplified from PR #9393)
num_stages = min(smem_capacity // int(stage_size), 4)

# Keep one stage of headroom for persistent fp32 to avoid launch-time OOR.
if is_persistent and (lhs_dtype == FP32 or rhs_dtype == FP32):
    num_stages = min(num_stages, 3)
```

The net effect: persistent FP32 kernels run with at most 3 pipeline stages,
leaving one stage-equivalent of SMEM as safety margin against launch-time
allocation failures.

### When This Matters

- **FP32 inputs** increase per-element storage by 2× vs BF16/FP16 (4 bytes
  vs 2 bytes), pushing the SMEM budget closer to the 228 KB (SM90) or
  99 KB (SM89) limit.
- **Persistent scheduling** adds the fixed overhead from Heuristic 1.
- Combined, the remaining headroom is thin, and a 4th stage can push
  allocation past the limit.

### What Is Directly Supported by the Diff

- The `min(num_stages, 3)` clamp with the comment about launch-time OOR.
- The double guard: both the 32 KB subtraction AND the stage-3 cap are applied.

### What Is Inferred

- Whether 3 stages are sufficient for full latency hiding in all FP32
  configurations.  The clamp is a safety measure; optimality is not claimed.
- Whether 2 stages (double buffering) would also work — the code does not
  test this.

## Heuristic 3: Disable Persistent Scheduling for Tiny Matrices

### The Problem

For very small matrix multiplications, the fixed cost of persistent
scheduling (CTA initialization, tile-queue management, try_cancel or
software-scheduler overhead) exceeds the benefit of reduced launch overhead.

### The Fix

Disable persistent scheduling when the total element count is below a
threshold:

```python
# Triton opt_flags.py (from PR #9393)
# TMA is slower for batched matmuls with small m/n/k.
if m * n * k < 131072:
    is_persistent = False
```

### What Is Directly Supported by the Diff

- The exact `m * n * k < 131072` threshold.
- The comment attributing this to TMA overhead on small problems.

### What Is Inferred

- Whether 131072 is a universal threshold or Triton-specific.  Other
  frameworks or hardware generations may have different crossover points.
- Whether this applies to non-batched matmuls.  The comment specifically
  mentions batched matmuls; the code applies unconditionally.

## Heuristic 4: Explicit TF32 Rounding for Non-TMA FP32 Inputs

### The Problem

TMA hardware automatically converts FP32 inputs to TF32 precision (19 bits:
1 sign + 8 exponent + 10 mantissa) during the global→shared memory copy on
Hopper.  When the kernel uses a non-TMA load path (e.g., `tl.load` with a
strided pointer for the non-TMA operand in a mixed TMA/non-TMA persistent
kernel), no automatic conversion occurs.  This creates a **precision
mismatch**: TMA-loaded operands are TF32-rounded; directly-loaded operands
retain full FP32 precision, causing inconsistent accumulation behavior.

### The Fix

Explicitly round FP32 inputs to TF32 in the non-TMA load path, matching
TMA's hardware rounding mode:

```python
# Triton _p_matmul.py (from PR #9393)
# Inside the non-TMA load branch:
if x.dtype == tl.float32 and ALLOW_TF32:
    # since data are not loaded from TMA we need to explicitly round to tf32.
    x = round_f32_to_tf32(x)
```

### Rounding Mode: cvt.rn vs cvt.rna

The PTX rounding instruction was also corrected.  On SM < 9.0, the legacy
`cvt.rna.tf32.f32` (round-nearest-away-from-zero) is used.  On SM ≥ 9.0
(Hopper+), `cvt.rn.tf32.f32` (round-nearest-even, a.k.a. banker's rounding)
is used to **match TMA's hardware rounding behavior**:

```python
@triton.jit
def round_f32_to_tf32(x: tl.tensor):
    # use cvt.rn on Hopper+ to match the rounding of TMA.
    ASM: tl.constexpr = (
        "cvt.rn.tf32.f32 $0, $1;"
        if cuda_capability_geq(9, 0)
        else "cvt.rna.tf32.f32 $0, $1;"
    )
    return tl.inline_asm_elementwise(
        ASM, "=r, r", [x], dtype=tl.float32, is_pure=True, pack=1
    )
```

| Architecture | Rounding Mode | PTX Instruction | When to Use |
|---|---|---|---|
| SM < 9.0 (pre-Hopper) | Round-nearest-away-from-zero | `cvt.rna.tf32.f32` | Legacy path |
| SM ≥ 9.0 (Hopper, Blackwell) | Round-nearest-even | `cvt.rn.tf32.f32` | Matches TMA hardware rounding |

### Key Insight

The rounding mode change (`rna` → `rn`) is not cosmetic.  If the non-TMA
path used a different rounding mode than TMA, the dot-product accumulations
would diverge, producing numerically inconsistent results between persistent
and non-persistent kernel paths for the same inputs.

### What Is Directly Supported by the Diff

- The conditional `cvt.rn` on SM ≥ 9.0, `cvt.rna` otherwise.
- The comment "use cvt.rn on Hopper+ to match the rounding of TMA."
- The explicit `round_f32_to_tf32` call gated on `ALLOW_TF32` and
  `x.dtype == tl.float32` in the non-TMA load branch.
- The `round_f32_to_tf32` function was added to both `_matmul.py` and
  `_p_matmul.py` (separate JIT compilation units).

### What Is Inferred

- The exact numerical impact of the `rna` → `rn` change (no error analysis
  in the PR).
- Whether `cvt.rna` was causing measurable divergence in practice, or this
  was a proactive correctness fix.

## Combined Application

All four heuristics should be applied together for persistent FP32 matmul:

```python
def configure_persistent_fp32_matmul(m, n, k, lhs_dtype, rhs_dtype,
                                     smem_capacity, stage_size):
    """Apply PR #9393 heuristics for persistent FP32 matmul configuration."""
    is_fp32 = (lhs_dtype == FP32 or rhs_dtype == FP32)

    # Heuristic 3: disable persistent for tiny matrices
    if m * n * k < 131072:
        return {"persistent": False}

    if is_fp32:
        # Heuristic 1: reserve fixed SMEM overhead
        smem_capacity -= 32 * 1024
        smem_capacity = max(smem_capacity, 0)

    num_stages = min(smem_capacity // stage_size, 4)

    if is_fp32:
        # Heuristic 2: leave one stage of safety headroom
        num_stages = min(num_stages, 3)

    # Heuristic 4: apply TF32 rounding in non-TMA load path
    # (handled inside the kernel, not the scheduler)

    return {
        "persistent": True,
        "num_stages": max(num_stages, 1),
        "needs_tf32_rounding": is_fp32,
    }
```

## Blackwell TMEM Overflow (PR #9967)

On Blackwell (SM100), FP32 matmul outputs face an additional resource constraint:
**TMEM capacity**. FP32 values consume 4× the TMEM space of FP16, and without
bounds checking, certain tile configurations silently overflow TMEM, producing
undefined results. PR #9967 (co-authored with Thomas Raoux, merged 2026-04-09)
addresses this by adding TMEM overflow guards in `opt_flags.py`.

This is the **TMEM-side complement** to the SMEM heuristics in Heuristics 1-4
above. Both SMEM (shared memory) and TMEM (tensor memory) budgets must be
respected for FP32 persistent matmul to be correct on Blackwell.

**Key takeaway for framework authors:** When targeting Blackwell with FP32,
budget not only SMEM (as in Heuristic 1) but also TMEM. Overflow of either
is a silent correctness bug, not just a performance issue.

## H200 measured evidence (Triton, SM90)

The FP32 persistent path was exercised on an H200 (132 SMs, cc 9.0) with
Triton 3.6.0. A persistent matmul (`min(132, num_tiles)` programs looping over
output tiles round-robin, `num_stages=3` per Heuristic 2) was compared against a
static-grid matmul sharing the **same** tiles / num_warps / num_stages, so the
two differ only in grid/loop structure.

**Correctness — PASS.** The persistent FP32/TF32 matmul produced output
**bit-identical** to the static-grid variant (`max |standard - persistent| =
0.0`) across all tested shapes (256³ … 4096×4096×1024), and both matched the
PyTorch TF32 reference within expected TF32 rounding. This empirically confirms
the PR's core enablement — the FP32 persistent path runs correctly — and that
the `cvt.rn` TF32 rounding is consistent between the persistent and non-persistent
paths.

**Latency — no gain from persistence alone.** Isolating the scheduling effect
(same tiles and stages, round-robin tile assignment), persistence measured
**0.92x–1.00x** (slightly *slower* at larger shapes). This is consistent with
the PR making no performance claim, and it shows that the latency gains of
Triton's *production* persistent matmul come from accompanying optimizations
(swizzled/grouped tile ordering for L2 reuse, TMA, cross-tile software
pipelining) that are orthogonal to the fixed-grid structure — not from
persistence per se. Full numbers and harness:
[`data/crawl-runs/h200/triton9393-persistent-matmul-h200-results.md`](../../data/crawl-runs/h200/triton9393-persistent-matmul-h200-results.md).

## When to Use

- **FP32 persistent matmul on Hopper (SM90)**: These heuristics are directly
  extracted from Triton's production persistent matmul path.
- **Any framework implementing persistent scheduling for FP32 GEMM**: The
  principles (fixed overhead reservation, stage cap, small-problem guard,
  TF32 rounding consistency) apply across frameworks, though exact constants
  may differ.

## When NOT to Use

- **Non-persistent kernels**: These heuristics are specific to persistent
  scheduling where SMEM carries scheduler state.
- **BF16 / FP16 persistent matmul**: The 32 KB overhead and stage cap are
  FP32-specific.  Lower-precision types have smaller per-element footprints
  and may not need these constraints.
- **Blackwell (SM100) CLC-based persistent kernels**: The existing
  `technique-persistent-kernels` page covers CLC scheduling; these Triton
  heuristics target Hopper's software-based persistent scheduling.

## Relationship to Other Techniques

- **[`technique-persistent-kernels`](persistent-kernels.md)**: Covers the CLC
  hardware scheduling pattern on SM100.  This page covers the SMEM budgeting
  aspect of software persistent scheduling on SM90.  The two are
  complementary — CLC eliminates some scheduler state from SMEM, potentially
  changing the headroom requirements.
- **[`technique-pipeline-stages`](pipeline-stages.md)**: Covers the general
  multi-stage circular buffer pattern.  This page adds FP32-specific stage
  count constraints on top of the general model.
- **[`technique-register-budgeting`](register-budgeting.md)**: Covers register
  budgeting for occupancy.  This page covers the analogous problem for shared
  memory — budgeting for persistent scheduling overhead.

## Caveats

- **No benchmark validation**: The heuristics prevent launch failures; they
  do not guarantee optimal performance.  The stage-3 cap may leave latency
  on the table in some configurations.
- **Triton-specific constants**: The 32 KB and 131072 values are from
  Triton's implementation.  Other frameworks should calibrate their own
  values by measuring actual SMEM allocation at launch time.
- **Hopper target**: The code targets SM90 (Hopper).  Blackwell (SM100) has
  228 KB SMEM per SM, potentially relaxing these constraints.  CLC-based
  persistent scheduling may also change the fixed-overhead profile.
- **Version-sensitive**: These heuristics reflect Triton's persistent matmul
  implementation as of merge SHA `f2895fae` (2026-02-20).  Future Triton
  releases may adjust the constants or retire the heuristics if the compiler
  learns to model the overhead automatically.
