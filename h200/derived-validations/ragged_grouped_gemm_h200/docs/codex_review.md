# Codex independent review

## Overall verdict

PASS WITH CHANGES

## Critical correctness issues

The original WMMA persistent prototype was rejected. Its global ticket was
incremented by every thread, whereas the WMMA tile computation requires all
threads/warps of a CTA to cooperate on one task. This gave divergent tile IDs,
NaNs, and CPU-reference error as high as 0.996. The revised candidate elects
thread 0 to fetch one task, publishes it in shared memory, and uses CTA-wide
barriers before the next ticket. It was rebuilt and revalidated on H200.

The revised API now rejects negative expert counts. Capacity cannot be checked
inside the public API because it receives offsets but not allocation lengths;
the test harness explicitly checks monotonic packed ranges and output bounds.

## Performance issues

`candidate_01_kernelwiki` removes the per-expert launch pattern and wins for
uniform, skewed, and small-token tests, but loses badly for mixed 4096x4096
(40.322 ms vs 18.755 ms). This is consistent with atomic-queue overhead and
the scalar micro-kernel becoming the limit on a large case. The implementation
does not use Tensor Cores, WGMMA, TMA, or shared-memory pipelining; its low
effective TFLOPS must not be interpreted as an H200-optimized GEMM result.

## KernelWiki usage assessment

PASS. The task ran the five requested searches, opened relevant pages, read
provenance, and translated concrete concepts into code: packed offsets from
FlashInfer, flattened group tiles from the grouped-GEMM page, and a persistent
group-aware scheduler from CUTLASS. It explicitly did not copy upstream code.
The source-to-decision table distinguishes observations, inference, and new
design.

## Missing experiments

- PyTorch loop and CUTLASS/cuBLAS references are unavailable in the current
  remote image: no discovered conda/PyTorch environment and no cuBLAS headers.
- `ncu` is absent, so no profile counters were collected.
- No Tensor-Core implementation passed validation after the rejected WMMA
  prototype; a production follow-up should use a tested SM90 CUTLASS/CuTe path.

## Required changes

Completed: CTA-level ticket ownership, full validation rerun, negative-count
validation, failed-prototype retention, useful-FLOP accounting, and reporting
of the mixed-case regression.

## Optional improvements

Use device-resident immutable metadata to avoid per-call uploads, bucket small
experts separately, and reintroduce an SM90 CUTLASS WGMMA/TMA path only with
CPU/library-reference correctness tests and NCU availability.
