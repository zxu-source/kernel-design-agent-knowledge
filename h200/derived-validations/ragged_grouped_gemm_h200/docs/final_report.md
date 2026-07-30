# BF16 Ragged Grouped GEMM on H200

## 1. Task definition

This experiment implements packed BF16 ragged grouped GEMM:
`C_e[M_e,N] = A_e[M_e,K] x B_e[K,N]`, with FP32 accumulation, fixed K/N, and
variable/zero expert token counts. `candidate_00_baseline` launches per
nonempty expert. `candidate_01_kernelwiki` makes a host-built flattened queue
of `(expert,tile_m,tile_n)` work and consumes it in one bounded persistent
launch.

## 2. Environment

Formal execution used GPU 0 only (`CUDA_VISIBLE_DEVICES=0`) on NVIDIA H200,
132 SMs, driver 570.124.06. `nvcc` was CUDA 13.0.88; the test binary SHA256 was
`15f981fb218e36ae724ace755d09c9e98fc578b58afec3c739487b67d1839334` before
the nonfunctional newline-only logging fix. The ai4qz checkout was commit
`6c366a0`. The local KernelWiki checkout could not resolve a standalone git
commit because its submodule metadata was incomplete; its validated corpus is
recorded in `kernelwiki_evidence.md`.

The current remote shell had no discoverable conda/PyTorch/CMake environment,
no cuBLAS headers, and no `ncu`. Therefore the program uses a self-contained
CUDA-runtime build and cannot honestly report PyTorch-loop, CUTLASS, cuBLAS, or
NCU results for this run.

## 3. Baseline design

The baseline is a custom CUDA implementation, not a PyTorch loop. It launches
one 32x32 FP32-accumulating BF16 scalar tile kernel grid per nonempty expert.
It deliberately establishes correct packed-offset and boundary behavior before
introducing a scheduler. This reference micro-kernel is not a Tensor-Core
optimization.

## 4. KernelWiki retrieval process

The required five queries were executed from `skills/KernelWiki/` and returned
grouped GEMM/MoE/CUTLASS/FlashInfer/SGLang/vLLM/DeepGEMM material. Relevant
pages were opened with `get_page.py`; provenance and key code bundles were read
for CUTLASS PR 3091, FlashInfer PR 1241, SGLang PR 9199, vLLM PR 25990, and
DeepGEMM PR 304. KernelWiki validation passed 2,271 files and 365 bundles.

## 5. Retrieved evidence

The decisive evidence was: packed offset-derived problem/pointer metadata from
FlashInfer PR 1241; the variable-M MoE problem and contiguous layout from
`kernel-grouped-gemm`; and CUTLASS PR 3091's group-aware persistent scheduler.
DeepGEMM's SM90 BF16 contiguous/masked distinction guided the test semantics.
The exact evidence, provenance paths, limits, and confidence distinctions are
in `kernelwiki_evidence.md`.

## 6. Knowledge-to-design mapping

The code adopts the following independently implemented migration: contiguous
packed offsets, no work for `M_e=0`, host-built tile flattening, one CTA-level
atomic ticket per persistent work item, and metadata-build timing outside the
kernel measurement. It does not copy upstream code. SM100/FP8/FP4 references
were not presented as directly portable SM90 BF16 results.

## 7. Optimized implementation

The first WMMA persistent prototype failed validation: each thread fetched a
different ticket, violating cooperative tile ownership. It is retained in
`candidates.jsonl`. The corrected candidate uses thread 0 to claim and publish
one task in shared memory, CTA synchronization, and a safe FP32 scalar
micro-kernel. This isolates scheduler effectiveness, but means the promoted
candidate is not an optimized Tensor-Core GEMM.

## 8. Codex review findings

The independent review rejected the broken WMMA candidate, required CTA-level
ticket ownership and input-count validation, and verified their re-test. It
also identified the missing library/PyTorch/NCU comparisons and the mixed-case
regression. Full review: `codex_review.md`.

## 9. Correctness results

All validated cases passed: E=1,4,8,16,32,64; single, uniform, empty, small,
skewed, and mixed distributions; `M_e` spans 0,1,2,7,16,31,64,127,256,512.
For each case persistent output exactly matched the baseline (`max_abs=0`,
`max_rel=0`) and small cases passed an independent CPU FP32 comparison with
maximum absolute error <= 0.00194609. The test checks finite outputs and packed
offset monotonic/bounds constraints. The detailed log is `outputs/validation.txt`.

## 10. Benchmark results and iteration stop

This report was reopened after a premature stop. The authoritative aligned
three-candidate rerun and complete process are in `iteration_log.md` and
`runs/candidate_02_static_queue_benchmark.csv`. `candidate_02_static_queue`
met the stopping criterion (correct and faster than baseline on every
representative case), so no three-regression fallback was needed.

All timings use CUDA Events, 20 warmups, 100 iterations, three trials, and
median values. TFLOPS are useful FLOPs `sum(2*M_e*N*K)`, never padded FLOPs.
Metadata construction is reported separately in `notes`.

| Distribution | K,N | Baseline us | Persistent us | Persistent / baseline |
| --- | ---: | ---: | ---: | ---: |
| uniform, E=8, 512 tokens | 1024,1024 | 1946.08 | 1341.18 | 1.45x |
| skewed, E=16, 629 tokens | 2048,1024 | 9953.42 | 4819.66 | 2.07x |
| small, E=32, 224 tokens | 4096,1024 | 36011.90 | 6448.69 | 5.59x |
| mixed, E=8, 504 tokens | 4096,4096 | 18754.60 | 40322.30 | 0.47x |

The per-expert custom baseline is the only available reference in the current
image. PyTorch-loop and CUTLASS/cuBLAS comparisons are explicitly unavailable,
not omitted selectively. Full rows: `benchmark.csv`.

## 11. Profiling results

NCU profiling could not run because `ncu --version` and `ncu --query-metrics`
both returned `bash: ncu: command not found`. No counter data are claimed; the
verbatim failure is in `profile/ncu_unavailable.txt`.

## 12. Failed candidates

`candidate_01_wmma_persistent_prototype` is retained with its NaN/correctness
failure and root cause in `candidates.jsonl`. The final scalar persistent path
also has a material mixed-case regression and is not claimed as universally
best.

## 13. Was KernelWiki effective?

**Partially effective.** Retrieval was effective: it located source-backed
grouped-MoE scheduling, metadata, and layout evidence across at least five
upstream repositories. Understanding and migration were demonstrated by a real
packed-offset persistent implementation, successful H200 correctness tests, and
large gains on skewed/small expert workloads. It was not strongly effective for
general performance: without a validated SM90 WGMMA/TMA/CUTLASS path, queue
overhead and the scalar micro-kernel caused a 2.15x slowdown on the mixed large
case. The knowledge improved the scheduler design, but did not by itself make a
production-quality H200 GEMM.

## 14. Limitations

The API has no allocation-length parameters, so offset capacity validation is
performed in the harness rather than enforceable inside the launcher. There is
no current PyTorch or cuBLAS reference, no NCU installation, no Tensor Core
candidate passing validation, and no profile metrics. A next iteration should
restore the intended conda/CUTLASS environment, use a library reference, and
only then add an SM90 WGMMA/TMA pipeline.

## 15. Reproduction commands

```bash
cd /inspire/hdd/project/qianghuaxuexi/wangtongyu-25057/kda-workspace/ragged_grouped_gemm_h200_upload/ragged_grouped_gemm_h200
/usr/local/cuda/bin/nvcc -std=c++17 -arch=sm_90 -Iinclude src/ragged_grouped_gemm.cu tests/ragged_grouped_gemm_test.cu -o ragged_grouped_gemm_test
CUDA_VISIBLE_DEVICES=0 ./ragged_grouped_gemm_test
CUDA_VISIBLE_DEVICES=0 ./ragged_grouped_gemm_test --bench
```
