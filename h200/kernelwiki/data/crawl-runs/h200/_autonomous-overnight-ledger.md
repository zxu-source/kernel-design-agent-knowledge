# Autonomous Overnight H200 Kernel-Validation Ledger (2026-07-20)

Driver: a recurring cron re-invokes Claude every few minutes; each firing
processes the next unfinished item below through the auditable lane, then ends.
User is asleep; leave a final summary here when every item is DONE or DEFERRED.

## Hard rules for every firing
- H200 is SM90 (Hopper). Only test SM90-runnable code. SM100-only (tcgen05,
  TMEM, NVFP4/MXFP4, 2-SM coop) -> DEFER with reason. Never fabricate a speedup.
- First, run: `/home/kirin_14379/projects/ai4qz/.venv/bin/ai4qz check h200_ncu`.
  If not ok, write a NOTE line and end the firing (do not force work).
- Pick the highest-priority item not DONE/DEFERRED; mark it `IN-PROGRESS` first.
- Lane: query.py -> harness (prefer raw Hopper CUDA-C++ via `nvcc -O3 -std=c++17
  -arch=sm_90a` when a framework API is uncertain — most reliable; CUTLASS C++
  headers at vllm/third_party/deep_gemm/include, cuBLAS at
  nvidia/cu13/{include,lib}; Triton 3.6 / CUTLASS DSL 4.4 also available) ->
  upload gzip+base64, decode+run on H200 -> correctness + repeated latency
  (warmup+iters, CUDA events) -> promote | record-negative | defer.
- promote = result log in data/crawl-runs/h200/ + derived artifact bundle at
  artifacts/kernels/<slug>/variants/ with PROVENANCE sha256 + wiki page H200
  evidence section. captured_at for NEW source pages: '2026-04-27'. H200
  validation dates: '2026-07-20 (overnight)'.
- Checkpoint EVERY firing: `python3 scripts/validate.py` (must be 0 errors) then
  `python3 scripts/generate-indices.py`. Then mark the item DONE/DEFERRED here
  with a one-line result. Keep each firing to ~2-3 operators then end.
- Do NOT commit, push, delete, or clone repos. `rm` is harness-blocked -> use
  `mv` or the data/crawl-runs/h200/ convention. _probe_*.py discovery scripts
  may accumulate; leave them.

## Progress
- [DONE] Triton #6394 PDL: promoted; 1.07-1.18x under graph replay; correctness PASS. (batch 1)
- [DONE] Triton #9393 persistent FP32/TF32 matmul: correctness PASS (bit-identical); latency 0.92-1.00x (record-negative). (batch 1)
- [DEFERRED] CUTLASS #2719 SM90 array-TMA GEMM PDL: bundled CUTLASS predates PR; version-API compile fail; technique covered by #6394. (batch 1)
- [DEFERRED] Triton #6299 warp-specialization matmul: Triton 3.6 WS is compiler-auto (no enable_warp_specialization kwarg); per-PR compiler internals not A/B-testable from Python. Scoped note added to wiki/techniques/warp-specialization.md. (batch 2, overnight)
- [DONE] Triton #6660 software-pipelined attention: from-scratch Triton FA-2 forward correctness PASS vs torch SDPA (err 7.6e-6..1.5e-5); 215-429 TFLOPS on H200; torch SDPA ~1.5-2.1x faster. New page wiki/kernels/triton-fa2-hopper.md + bundle + result log. (batch 2, overnight)
- [DEFERRED] Hopper WGMMA TF32 GEMM vs SIMT: deferred to a focused firing — raw wgmma.mma_async PTX needs careful shared-memory descriptor encoding (14-bit descriptor, register-operand A, fence/commit_group/wait_group), high one-shot risk for an autonomous firing. Revisit with a dedicated attempt. (batch 2, overnight)
- [DONE/record-negative] Hopper cp.async multi-stage copy/compute overlap: correctness PASS (piped==serial, delta=0); pipelining 0.77-0.97x SLOWER than straightforward global loads for the elementwise pattern (hardware already hides global-load latency); refines Codex's cuda::memcpy_async negative. Attached to wiki/techniques/pipeline-stages.md + bundle + result log. (batch 3, overnight)
- [DONE] Hopper thread-block cluster + DSMEM broadcast: cluster formation validated (num_blocks()=4); DSMEM cross-block read/write via map_shared_rank WORKS (correctness PASS, delta=0); found + documented the required post-push cluster.sync() (else peer frees shared mid-write -> fault); DSMEM broadcast 0.81-0.83x SLOWER than global bounce (record-negative for bulk broadcast). New page wiki/hardware/thread-block-cluster.md + bundle + result log. (batch 3, overnight)
- [DONE] Hopper FP8 e4m3 matmul (Triton): correctness PASS (rel err 1-3% vs dequantized fp32 ref = fp8 quantization floor); naive Triton fp8 underutilizes fp8 wgmma (156-187 TFLOPS, ~2x SLOWER than bf16 330-501) — kernel-quality limitation, not fp8 hardware statement; BK=256 tuning hit shared-mem OOR. Correctness validated; throughput NOT representative (needs production fp8 GEMM). Attached to wiki/kernels/fp8-block-scale-gemm.md + bundle + result log. (batch 4, overnight)
- [DONE] Hopper TMA (cp.async.bulk.tensor.2d via CUtensorMap + mbarrier): correctness PASS (delta=0); TMA 1565 GB/s ~1.96x naive per-thread copy (797 GB/s) on H200. Found + documented the {x,y} coord-order requirement and the -lcuda link requirement. Attached to wiki/hardware/tma.md + bundle + result log. (batch 3, overnight)
- [DONE/record-negative] Hopper TMA STORE (cp.async.bulk.tensor shared->global): correctness PASS (delta=0); TMA store 485 GB/s ~3.5x SLOWER than naive store (1680 GB/s) -> TMA benefit is asymmetric (load-only). Attached to wiki/hardware/tma.md + bundle + result log. (open slot, overnight)
- [DEFERRED] Hopper WGMMA TF32/bf16 GEMM: raw wgmma A-register-fragment layout is the blocker; deferred to interactive morning work. (overnight)
- [DEFERRED] CUTLASS DSL #3091 grouped GEMM: CuTe DSL API learning curve; deferred to morning. (overnight)
- [DEFERRED] CUTLASS #2881 TMA prefetch: CUTLASS C++ version-API blocker (same as #2719); deferred. (overnight)

## Worklist (priority order) — claim next unfinished
BATCH 2
- [x] Triton #6299 warp-specialization matmul (Triton 3.6). A/B WS on/off if the API exposes a toggle; else correctness of WS matmul vs torch + observation. Attach to wiki/techniques/warp-specialization.md.  -> DEFERRED (compiler-auto, no toggle)
- [x] Triton #6660 software-pipelined attention (Triton 3.6). FA-style attention correctness vs torch.nn.functional.scaled_dot_product_attention + latency at a few (seqlen, d). New wiki kernel page or attach to flash-attention-4.md.  -> DONE (new page wiki/kernels/triton-fa2-hopper.md)
- [x] Hopper WGMMA TF32 (and/or bf16) GEMM vs SIMT GEMM (raw CUDA C++, nvcc -arch=sm_90a, wgmma.mma_async). Reliable, no framework risk. New wiki hardware/technique page for wgmma if absent. -> DEFERRED: raw wgmma needs the A-matrix register-fragment layout (per-thread (row,col) mapping across the 4-warp warpgroup) derived from PTX docs / CUTLASS atom recipe; high one-shot correctness risk autonomously. Better attempted interactively (morning).

BATCH 3
- [x] Hopper TMA async copy (cp.async.bulk.tensor via tensormap) vs cp.async vs naive (raw CUDA, nvcc sm_90a). Reliable. Attach to wiki/hardware/tma.md. -> DONE (correctness PASS, delta=0; TMA 1565 GB/s ~1.96x naive 797 GB/s; new evidence on wiki/hardware/tma.md)
- [x] Hopper mbarrier async-copy/compute overlap (raw CUDA, nvcc sm_90a): copy+compute overlapped via cuda::barrier/mbarrier vs serial. Reliable. -> DONE/record-negative (0.77-0.97x; hardware hides global-load latency for elementwise patterns)
- [x] Thread-block Cluster distributed shared memory on Hopper (cluster_size=2, raw CUDA, nvcc sm_90a): demonstrate DSMEM + correctness. Reliable. -> DONE (cluster+DSMEM validated correct; DSMEM broadcast 0.81-0.83x vs global bounce = record-negative for bulk broadcast; new page wiki/hardware/thread-block-cluster.md)

BATCH 4+
- [x] CUTLASS DSL #3091 Hopper grouped GEMM via cutlass.cute (CuTe DSL) if API cooperative; else defer. -> DEFERRED: CuTe DSL (nvidia_cutlass_dsl) is an MLIR-based Python DSL needing its atom/collective-builder API learned; not tractable in one autonomous firing. Morning: try a minimal cutlass.cute GEMM, else use deep_gemm's installed SM90 kernels.
- [x] CUTLASS #2881 TMA prefetch (C++ compile-risk like #2719; defer if it fails). -> DEFERRED: CUTLASS C++ CollectiveBuilder/GemmUniversal template signatures in the bundled (vLLM/deep_gemm) snapshot differ from the standard recipe (same version-API blocker as #2719); a TMA-prefetch repro would need that resolved first.
- [x] Hopper FP8 (e8m0 block-scale / e4m3) via wgmma or Triton, if a clean path exists; else defer. -> DONE (Triton fp8 e4m3 correctness PASS; naive kernel underutilizes fp8 units, throughput not representative; attached to wiki/kernels/fp8-block-scale-gemm.md)
- [x] Any further raw Hopper primitive with a clean A/B knob. -> DONE: TMA store (cp.async.bulk.tensor shared->global) vs naive store; correctness PASS, record-negative (0.29x; TMA benefit is load-only). See wiki/hardware/tma.md.

## Morning summary (fill in when worklist exhausted)

Status: **worklist exhausted** — every item DONE or DEFERRED. Repo at 0 validate
errors, 386 asset bundles (started at 378 before this H200 push; +8 derived
runnable bundles). No commits made (all local for review).

### Tally (8 primitives validated on H200 + 5 deferred)

**PROMOTED — validated correctness + real measured gain:**
1. Triton #6394 PDL — 1.07-1.18x (CUDA-graph replay), correctness PASS.
2. Hopper TMA LOAD (cp.async.bulk.tensor) — 1.96x (1565 vs 797 GB/s), correctness PASS.

**CORRECTNESS-VALIDATED + honestly characterized (no fabricated speedup):**
3. Triton #9393 persistent FP32/TF32 matmul — correctness PASS (bit-identical); persistence-alone 0.92-1.00x (record-negative).
4. Triton #6660 FA-2 attention — correctness PASS vs torch SDPA; 215-429 TFLOPS; torch SDPA 1.5-2.1x faster (characterized).
5. cp.async copy/compute overlap — correctness PASS; 0.77-0.97x (record-negative; hw hides global-load latency).
6. thread-block cluster + DSMEM — correctness PASS; DSMEM broadcast 0.81-0.83x (record-negative for bulk broadcast). Found the post-push cluster.sync() rule.
7. FP8 e4m3 matmul (Triton) — correctness PASS (1-3% rel err = fp8 floor); naive kernel underutilizes fp8 units (record-negative, kernel-quality limited).
8. TMA STORE (cp.async.bulk.tensor shared->global) — correctness PASS; 0.29x (record-negative; TMA benefit is load-only).

**DEFERRED (need interactive / focused work):**
- CUTLASS #2719 SM90 array-TMA GEMM PDL — bundled CUTLASS predates PR + version-API compile fail; technique covered by PDL.
- Triton #6299 warp-spec matmul — Triton 3.6 WS is compiler-auto, no user toggle.
- WGMMA TF32/bf16 GEMM — raw wgmma A-register-fragment layout is the blocker (PTX-doc/CUTLASS-atom recipe); attempt interactively.
- CUTLASS DSL #3091 grouped GEMM — CuTe DSL API learning curve.
- CUTLASS #2881 TMA prefetch — CUTLASS C++ version-API blocker (same as #2719).

### Key empirical takeaways for the wiki
- PDL pays off only under graph capture (CPU-dispatch-free); ~0.2 us/launch saved.
- Persistence alone does not speed up matmul (needs swizzling/TMA).
- cp.async / DSMEM / TMA-store do NOT beat simple alternatives for elementwise/bulk patterns on H200 — the hardware already hides latency / coalesces well. These techniques pay off only inside warp-specialized producer/consumer GEMM pipelines.
- TMA LOAD is a clear ~2x win over naive copies.
- Naive Triton fp8 tl.dot does not hit the fast fp8 wgmma path; production fp8 GEMM needed.
- Cluster + DSMEM validated correct (post-push cluster.sync() is mandatory).

### New files (this H200 push)
Result logs (data/crawl-runs/h200/): triton6394-pdl, triton6393-persistent-matmul, cutlass2719-sm90-tma-gemm-defer, triton6660-flash-attention, hopper-cpasync-overlap, hopper-cluster-dsmem, hopper-fp8-matmul, hopper-tma-copy, hopper-tma-store (all `-h200-results.md`).
Runnable harnesses: triton6394_pdl_{graph,chain}.py, triton9393_persistent_matmul.py, triton6660_flash_attention.py, hopper_{cpasync_overlap,cluster_dsmem,tma_copy,tma_store}.cu, hopper_fp8_matmul.py, cutlass2719_sm90_gemm.cu (defer evidence). All sha256-tracked in artifacts/kernels/<slug>/variants/.
New wiki pages: kernels/triton-fa2-hopper.md, hardware/thread-block-cluster.md. Updated: hardware/pdl-gdc.md (->verified), hardware/tma.md (load+store), techniques/{persistent-matmul-resource-budgeting,pipeline-stages,warp-specialization}.md, kernels/fp8-block-scale-gemm.md.

### Cron
This recurring cron is being deleted (worklist exhausted); it would otherwise auto-expire in 7 days. To resume focused work on WGMMA / CUTLASS items, re-invoke manually in the morning.

