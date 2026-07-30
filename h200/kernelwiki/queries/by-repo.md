# Query: By Repository

> Auto-generated. Do not edit manually.

<a id="dao-ailabflash-attention"></a>
## Dao-AILab/flash-attention
28 PRs

| PR | Title | Date | Techniques | Tags |
|-----|-------|------|------------|------|
| [#2717](../sources/prs/flash-attention/PR-2717.md) | [CuTe, SM100] Fix FP8 e4m3 accuracy: make max_offset dtype-aware to avoid P saturation | 2026-07-18 |  | fp8 |
| [#2685](../sources/prs/flash-attention/PR-2685.md) | ci: install cutlass-dsl/quack at runtime to decouple from the baked image | 2026-06-27 |  | gemm |
| [#2675](../sources/prs/flash-attention/PR-2675.md) | [AMD ROCm] Enable RDNA backward and adopt CK unified workspace | 2026-06-24 |  | gemm |
| [#2671](../sources/prs/flash-attention/PR-2671.md) | Fix CuTe SM120 compile-time argument handling | 2026-06-21 |  | gemm |
| [#2666](../sources/prs/flash-attention/PR-2666.md) | fix(hd256/sm100): make q/k/v contiguous before dedicated hd256 kernel | 2026-06-19 |  | gemm |
| [#2661](../sources/prs/flash-attention/PR-2661.md) | Enable 2CTA for SM100 block-sparse backward | 2026-06-16 |  | 2sm-cooperative, flash-attention |
| [#2662](../sources/prs/flash-attention/PR-2662.md) | [cute] Fix int32 overflow in SM100 LPT tile scheduler for long context | 2026-06-16 | tile-scheduling | tile-scheduling |
| [#2636](../sources/prs/flash-attention/PR-2636.md) | ci(fa4): enforce cutlass-dsl/quack dep floors and rebake cu130 image | 2026-06-10 |  | gemm |
| [#2515](../sources/prs/flash-attention/PR-2515.md) | Fix ZeroDivisionError in num_splits_heuristic for empty Q workloads | 2026-04-28 |  | gemm |
| [#2481](../sources/prs/flash-attention/PR-2481.md) | [cute,bwd] fix PDL race in bwd_preprocess, which corrupting dpsum on SM90+ | 2026-04-21 |  | gemm |
| [#2402](../sources/prs/flash-attention/PR-2402.md) | feat(cute): implement softcap backward pass, correct math formula, and resolve JIT cache bug | 2026-03-28 |  | gemm |
| [#2343](../sources/prs/flash-attention/PR-2343.md) | [ROCm] Auto-detect Triton backend if C++ extension is missing | 2026-03-13 |  | gemm |
| [#2282](../sources/prs/flash-attention/PR-2282.md) | Add FA4 publishing strategy | 2026-02-27 |  | gemm |
| [#2242](../sources/prs/flash-attention/PR-2242.md) | Fix Hopper tests | 2026-02-08 |  | gemm |
| [#2189](../sources/prs/flash-attention/PR-2189.md) | [Cute][Flex] Fix expanded tensor bug | 2026-01-16 |  | gemm |
| [#2178](../sources/prs/flash-attention/PR-2178.md) |  [AMD] Triton Backend for ROCm #3 | 2026-01-14 | kernel-fusion | decode, fp8, kernel-fusion |
| [#2109](../sources/prs/flash-attention/PR-2109.md) | [Cute,Fwd,Sm100] fp8 e4m3 and e5m2 support | 2025-12-29 |  | attention, flash-attention, fp8 |
| [#2070](../sources/prs/flash-attention/PR-2070.md) | Add score-mod bwd support  | 2025-12-15 |  | tma |
| [#1985](../sources/prs/flash-attention/PR-1985.md) | [Cute] Block sparse support Sm100 | 2025-11-05 |  | gemm |
| [#1949](../sources/prs/flash-attention/PR-1949.md) | Fix hopper cuda 13 build | 2025-10-21 |  | gemm |
| [#1952](../sources/prs/flash-attention/PR-1952.md) | [NVIDIA] cutlass v4.3.0 | 2025-10-21 |  | gemm |
| [#1904](../sources/prs/flash-attention/PR-1904.md) | C++11 fix warnings | 2025-09-23 |  | gemm |
| [#1893](../sources/prs/flash-attention/PR-1893.md) | Improve causal backward determinism perf with SPT schedule | 2025-09-17 | epilogue-fusion, tile-scheduling | epilogue-fusion, tile-scheduling, tma |
| [#1868](../sources/prs/flash-attention/PR-1868.md) | flash-attn-cute bwd sm90 | 2025-09-06 |  | gemm |
| [#1769](../sources/prs/flash-attention/PR-1769.md) | Add torch.compile support to flash attention 3 | 2025-07-22 |  | attention |
| [#1604](../sources/prs/flash-attention/PR-1604.md) | Support hdimQK != hdimV backward | 2025-04-21 | epilogue-fusion | epilogue-fusion, tma |
| [#1511](../sources/prs/flash-attention/PR-1511.md) | Fix CUDA 12.1 build | 2025-02-26 |  | tma |
| [#1442](../sources/prs/flash-attention/PR-1442.md) | Replace c10::optional with std::optional in flash_attn | 2025-01-13 | kernel-fusion | attention, kernel-fusion |

<a id="nvidiatransformerengine"></a>
## NVIDIA/TransformerEngine
2 PRs

| PR | Title | Date | Techniques | Tags |
|-----|-------|------|------------|------|
| [#3194](../sources/prs/TransformerEngine/PR-3194.md) | Work around nvcc 13.0 parse failure in group_quantize_fp8.cuh | 2026-07-08 |  | fp8, quantization |
| [#3186](../sources/prs/TransformerEngine/PR-3186.md) | [Common] Pass cu_seqlens and token-unit ragged offsets directly to cuDNN SDPA fprop | 2026-07-07 | kernel-fusion | fp8, kernel-fusion |

<a id="nvidiacudnn-frontend"></a>
## NVIDIA/cudnn-frontend
15 PRs

| PR | Title | Date | Techniques | Tags |
|-----|-------|------|------------|------|
| [#398](../sources/prs/cudnn-frontend/PR-398.md) | SDPA bwd: linear-workspace deterministic path on Hopper (cuDNN 9.25+) | 2026-07-16 |  | attention, flash-attention |
| [#366](../sources/prs/cudnn-frontend/PR-366.md) | Expose cu_seq_len_q/kv on the sdpa_fp8 python binding | 2026-07-08 |  | attention, fp8 |
| [#353](../sources/prs/cudnn-frontend/PR-353.md) | cuDNN Frontend v1.26.0 | 2026-07-07 | kernel-fusion | attention, gemm, grouped-gemm |
| [#333](../sources/prs/cudnn-frontend/PR-333.md) | Add block-sparse attention CuTe DSL kernels for Hopper and Blackwell | 2026-07-01 | pipeline-stages, tile-scheduling | attention, pipeline-stages, sparse-attention |
| [#335](../sources/prs/cudnn-frontend/PR-335.md) | Bypass OSS d=256 path on cuDNN 9.23+ | 2026-07-01 |  | gemm |
| [#285](../sources/prs/cudnn-frontend/PR-285.md) | Skip TensorIR MemBound / compile-time-const samples on consumer Blackwell (SM12x) | 2026-06-05 | kernel-fusion | kernel-fusion |
| [#263](../sources/prs/cudnn-frontend/PR-263.md) | DSA: fix CuTe DSL guards and add SM90 indexer forward | 2026-06-01 |  | attention, decode, sparse-attention |
| [#215](../sources/prs/cudnn-frontend/PR-215.md) | cuDNN Frontend v1.22.0 Release | 2026-04-02 | kernel-fusion | attention, flash-attention, fp8 |
| [#205](../sources/prs/cudnn-frontend/PR-205.md) | cuDNN Frontend v1.19.0 Release | 2026-03-06 | persistent-kernel | attention, flash-attention, fp8 |
| [#166](../sources/prs/cudnn-frontend/PR-166.md) | cuDNN Frontend v1.14.1 | 2025-09-04 |  | attention, flash-attention, fp8 |
| [#159](../sources/prs/cudnn-frontend/PR-159.md) | Fix hopper bprop surface check | 2025-08-16 |  | attention, flash-attention |
| [#144](../sources/prs/cudnn-frontend/PR-144.md) | Update/1.2.0 | 2025-05-27 |  | attention, flash-attention |
| [#141](../sources/prs/cudnn-frontend/PR-141.md) | v1.12.0 release | 2025-05-15 | kernel-fusion | attention, decode, flash-attention |
| [#118](../sources/prs/cudnn-frontend/PR-118.md) |  cudnn frontend v1.8 release | 2024-10-23 | kernel-fusion | attention, flash-attention, fp8 |
| [#92](../sources/prs/cudnn-frontend/PR-92.md) | cudnn FE 1.6.0  | 2024-08-09 | kernel-fusion | attention, flash-attention, fp8 |

<a id="nvidiacutlass"></a>
## NVIDIA/cutlass
93 PRs

| PR | Title | Date | Techniques | Tags |
|-----|-------|------|------------|------|
| [#3392](../sources/prs/cutlass/PR-3392.md) | Fix per-group norm constant indexing in block-scale-factor epilogue stores (grouped/ptr-array GEMM) | 2026-07-16 | epilogue-fusion, kernel-fusion | block-scale, epilogue-fusion, gemm |
| [#3381](../sources/prs/cutlass/PR-3381.md) | SM100 GEMM AR, GEMM RS Update | 2026-07-13 |  | gemm |
| [#3378](../sources/prs/cutlass/PR-3378.md) | [Hopper CuTeDSL] Add FP8 GEMM with GELU epilogues | 2026-07-09 | epilogue-fusion, persistent-kernel | epilogue-fusion, fp8, gemm |
| [#3376](../sources/prs/cutlass/PR-3376.md) | fix arguments for tensorop_gemm | 2026-07-08 |  | gemm |
| [#3369](../sources/prs/cutlass/PR-3369.md) | Fix grouped GEMM issue | 2026-07-07 |  | gemm, grouped-gemm |
| [#3370](../sources/prs/cutlass/PR-3370.md) | [CuTeDSL] Gate Blackwell example tuning by CUDA backend | 2026-07-07 |  | attention, gemm, grouped-gemm |
| [#3366](../sources/prs/cutlass/PR-3366.md) | add a markdown doc about how to measure gemm performance | 2026-07-03 |  | gemm |
| [#3344](../sources/prs/cutlass/PR-3344.md) | [CLI] Recover ssd and blockwise group gemm perf | 2026-06-23 |  | attention, gemm, grouped-gemm |
| [#3322](../sources/prs/cutlass/PR-3322.md) | Update tensorop_gemm.py | 2026-06-16 |  | gemm |
| [#3303](../sources/prs/cutlass/PR-3303.md) | Fix dense_gemm_fp8_2xacc validation codes | 2026-06-05 |  | fp8, gemm |
| [#3292](../sources/prs/cutlass/PR-3292.md) | add tileN = 8,16 for SM120 blockscale GEMM. | 2026-06-02 | epilogue-fusion | epilogue-fusion, gemm |
| [#3286](../sources/prs/cutlass/PR-3286.md) | Fix cross-execution-space error: remove CUTLASS_HOST_DEVICE from CudaHostAdapter::memsetDevice | 2026-05-29 |  | gemm |
| [#3283](../sources/prs/cutlass/PR-3283.md) | Enable tcgen05 blockscaled ops on Thor SM110 | 2026-05-28 | persistent-kernel | gemm, persistent-kernel, tcgen05 |
| [#3280](../sources/prs/cutlass/PR-3280.md) | [SM120] Add ptr-array TMA collective for tensor/token-scaled FP8 grouped GEMM | 2026-05-27 |  | fp8, gemm, tma |
| [#3274](../sources/prs/cutlass/PR-3274.md) | [CLI] add mixed precision blockscaled gemm example | 2026-05-26 | persistent-kernel | gemm, persistent-kernel |
| [#3272](../sources/prs/cutlass/PR-3272.md) | Add Blackwell GeForce blockscaled GEMM examples | 2026-05-25 | persistent-kernel | gemm, persistent-kernel |
| [#3262](../sources/prs/cutlass/PR-3262.md) | Fix hopper grouped gemm example `elect_sync` api use | 2026-05-22 |  | gemm, grouped-gemm |
| [#3251](../sources/prs/cutlass/PR-3251.md) | [CLI] Update FMHA & improve perf | 2026-05-20 |  | attention, flash-attention |
| [#3206](../sources/prs/cutlass/PR-3206.md) | [CuTeDSL] Update atomic_max_float32 to atomic_fmax in blockscaled GEMM | 2026-05-07 | persistent-kernel | gemm, persistent-kernel |
| [#3184](../sources/prs/cutlass/PR-3184.md) | Add Snake activation functor for EVT | 2026-04-24 | epilogue-fusion | epilogue-fusion |
| [#3176](../sources/prs/cutlass/PR-3176.md) | Small Tile N BlockScaled GEMM + Grouped GEMM on SM12x | 2026-04-19 | kernel-fusion | gemm, kernel-fusion, tma |
| [#3149](../sources/prs/cutlass/PR-3149.md) | [Hopper CuTeDSL] Add FP8 GEMM with 2xAcc | 2026-04-05 |  | fp8, gemm |
| [#3130](../sources/prs/cutlass/PR-3130.md) | Update blackwell tutorial to be compatible with 4.5-dev version | 2026-03-25 | persistent-kernel | gemm, persistent-kernel |
| [#3124](../sources/prs/cutlass/PR-3124.md) | [CuTeDSL] Add SM103 grouped block-scaled GEMM kernel and tests | 2026-03-23 |  | block-scale, gemm, grouped-gemm |
| [#3106](../sources/prs/cutlass/PR-3106.md) | [CLI] add cutedsl fp16 gemm tutorial from 2 to 6 | 2026-03-13 |  | gemm |
| [#3092](../sources/prs/cutlass/PR-3092.md) | Support for Group GEMM in CUTLASS Profiler for GeForce and Spark | 2026-03-07 |  | gemm, grouped-gemm |
| [#3091](../sources/prs/cutlass/PR-3091.md) | [Hopper CuTeDSL] Add grouped GEMM kernel example | 2026-03-06 |  | gemm, grouped-gemm |
| [#3087](../sources/prs/cutlass/PR-3087.md) | CuTeDSL] Fix Boolean.__dsl_and__ to emit arith.andi directly for i1 operands | 2026-03-01 |  | gemm |
| [#3055](../sources/prs/cutlass/PR-3055.md) | Replace std::min with cute::min in sm120 blockwise scaling device functions | 2026-02-23 |  | gemm, tma |
| [#3027](../sources/prs/cutlass/PR-3027.md) | Replace fence proxy to the latest routine code in examples/distributed/all_reduce_tma.py | 2026-02-13 |  | tma |
| [#3021](../sources/prs/cutlass/PR-3021.md) | [Cute-DSL] Add option for issue_clc_query without multicast | 2026-02-11 |  | clc |
| [#2995](../sources/prs/cutlass/PR-2995.md) | [CuTeDSL] Fix: SM100 block-scale gemm overlapping accumulator | 2026-02-03 | double-buffering, pipeline-stages, epilogue-fusion | block-scale, cute-dsl, double-buffering |
| [#2990](../sources/prs/cutlass/PR-2990.md) | fix performance regression in cute-dsl examples for 4.4-ctk13.1 release  | 2026-01-28 |  | gemm |
| [#2988](../sources/prs/cutlass/PR-2988.md) | fix performance inssues in cute-dsl examples for 4.4-ctk13.1 release | 2026-01-27 |  | gemm, grouped-gemm |
| [#2977](../sources/prs/cutlass/PR-2977.md) | Fix error in Blackwell document of referring to Mxf4 format as NVF4 | 2026-01-23 |  | gemm |
| [#2970](../sources/prs/cutlass/PR-2970.md) | [CuTeDSL] Distributed example, using TMA load to access remote memory rank-by-rank, reducing in cta, broadcast result to all ranks by multimem TMA store | 2026-01-21 |  | tma |
| [#2965](../sources/prs/cutlass/PR-2965.md) | [Bug Fix]Set NumSplitsM to 1 when TileShapeM < 128 in sm90 fp8 blockwise scaling CollectiveMma | 2026-01-19 |  | fp8, gemm, tma |
| [#2945](../sources/prs/cutlass/PR-2945.md) | Fix out-of-bounds TMA access in wgmma_tma_sm90 tutorial | 2026-01-10 |  | tma, wgmma |
| [#2946](../sources/prs/cutlass/PR-2946.md) | [Cutlass gemm] Fix SM100 FP8 nosmem epilogue-fusion shape_div 'Divisibility Condition' for non-multiple-of-64 N tiles | 2026-01-10 | epilogue-fusion | gemm, fp8, epilogue-fusion |
| [#2930](../sources/prs/cutlass/PR-2930.md) | Minor fix for testing of blockscaled dense GEMM with TMA prefetch | 2026-01-05 | persistent-kernel | gemm, persistent-kernel, tma |
| [#2928](../sources/prs/cutlass/PR-2928.md) | Replace CUDA driver API with runtime API | 2026-01-04 |  | tma |
| [#2921](../sources/prs/cutlass/PR-2921.md) | Fix incorrect tensor layout strides in Blackwell MMA tutorial comments | 2026-01-03 |  | gemm |
| [#2894](../sources/prs/cutlass/PR-2894.md) | Fix CUDA version checking in examples | 2025-12-21 | warp-specialization | flash-attention, fp8, gemm |
| [#2889](../sources/prs/cutlass/PR-2889.md) | add missing condition for sync | 2025-12-19 |  | gemm |
| [#2881](../sources/prs/cutlass/PR-2881.md) | new example with TMA prefetch feature targeting for DRAM latency boun… | 2025-12-16 | persistent-kernel | gemm, persistent-kernel, tma |
| [#2865](../sources/prs/cutlass/PR-2865.md) | [Bug Fix]Bypass launch grids for SM120 Kernel with SM90 Mainloop & SM100 TileScheduler | 2025-12-09 |  | gemm, tma |
| [#2826](../sources/prs/cutlass/PR-2826.md) | add pytest support for tutorial gemm | 2025-12-01 |  | gemm |
| [#2790](../sources/prs/cutlass/PR-2790.md) | Blockscaled Ragged Contiguous Grouped Gemm for MoEs | 2025-11-21 |  | gemm, moe |
| [#2780](../sources/prs/cutlass/PR-2780.md) | add a notebook for tour to sol gemm | 2025-11-18 |  | gemm |
| [#2750](../sources/prs/cutlass/PR-2750.md) | Add tutorial fp16_gemm_1 | 2025-11-05 |  | gemm |
| [#2746](../sources/prs/cutlass/PR-2746.md) | Support for GEMM-K=0 for Blackwell Grouped GEMMs | 2025-11-04 | warp-specialization, persistent-kernel, pipeline-stages | grouped-gemm, gemm, pipeline-stages |
| [#2740](../sources/prs/cutlass/PR-2740.md) | Fix register index bug in mma.sync.aligned.m16n8k16 | 2025-11-01 |  | gemm |
| [#2719](../sources/prs/cutlass/PR-2719.md) | Support PDL for SM90 Array TMA GEMM | 2025-10-24 |  | gemm, tma |
| [#2713](../sources/prs/cutlass/PR-2713.md) | DistGEMM bug fixes | 2025-10-22 |  | gemm |
| [#2686](../sources/prs/cutlass/PR-2686.md) | [Bug Fix]Fixed compilation error when using StreamK scheduler + PDL(SM90) | 2025-10-10 | tile-scheduling | gemm, tile-scheduling |
| [#2684](../sources/prs/cutlass/PR-2684.md) | minor fix for sm90 mixed input building issue | 2025-10-04 |  | gemm |
| [#2639](../sources/prs/cutlass/PR-2639.md) | doc change for 4.2 | 2025-09-15 |  | gemm |
| [#2599](../sources/prs/cutlass/PR-2599.md) | fix gqa issue for blackwell fmha.py | 2025-08-28 |  | flash-attention |
| [#2582](../sources/prs/cutlass/PR-2582.md) | Fix Copy_Atom type mismatch in sgemm_sm80.cu | 2025-08-20 |  | gemm |
| [#2538](../sources/prs/cutlass/PR-2538.md) | Made documentation updates in batched_gemm.cu | 2025-08-04 |  | gemm |
| [#2492](../sources/prs/cutlass/PR-2492.md) | fix: examples/cute/tutorial/blackwell/04_mma_tma_2sm_sm100.cu GridDim miscalculated | 2025-07-23 |  | 2sm-cooperative, tma |
| [#2472](../sources/prs/cutlass/PR-2472.md) | Add Blackwell MLA forward (shape: d=192, dv=128) implementation | 2025-07-16 | warp-specialization, pipeline-stages, double-buffering | mla, attention, prefill |
| [#2466](../sources/prs/cutlass/PR-2466.md) | Example 77 add blackwell flash-attention bwd for MLA shape | 2025-07-14 | warp-specialization, double-buffering, pipeline-stages | mla, attention, flash-attention |
| [#2378](../sources/prs/cutlass/PR-2378.md) | support fp16 accmulator for sm89 fp8 mma | 2025-06-07 |  | fp8, gemm |
| [#2366](../sources/prs/cutlass/PR-2366.md) | [ex77] fix mla split; add fwd lse; add bwd varlen | 2025-06-04 | epilogue-fusion, kernel-fusion, tile-scheduling | epilogue-fusion, flash-attention, kernel-fusion |
| [#2333](../sources/prs/cutlass/PR-2333.md) | Fix epilogue::thread::Convert cannot be used with DefaultEpilogue | 2025-05-26 | epilogue-fusion | epilogue-fusion |
| [#2324](../sources/prs/cutlass/PR-2324.md) | Ensure compatibility with "-Wimplicit-fallthrough" when compiling with Clang | 2025-05-22 | swizzling | swizzling, tma |
| [#2291](../sources/prs/cutlass/PR-2291.md) | Correct divmod order in example 77 (blackwell fmha) | 2025-05-13 | tile-scheduling | flash-attention, tile-scheduling |
| [#2292](../sources/prs/cutlass/PR-2292.md) | Handle get_masked_trip_count for small length in fmha example | 2025-05-13 | kernel-fusion | flash-attention, kernel-fusion |
| [#2275](../sources/prs/cutlass/PR-2275.md) | Fix group scale gemm when K==128 | 2025-05-02 |  | fp8, gemm, tma |
| [#2276](../sources/prs/cutlass/PR-2276.md) | [CUTLASS] Add GNA to PUBLICATIONS.md | 2025-05-02 |  | gemm |
| [#2267](../sources/prs/cutlass/PR-2267.md) | war to fix blackwell grouped groupwise hang | 2025-04-29 | tile-scheduling | gemm, tile-scheduling |
| [#2270](../sources/prs/cutlass/PR-2270.md) | hopper-blockwise-generalization-optimization | 2025-04-29 |  | fp8, gemm, tma |
| [#2256](../sources/prs/cutlass/PR-2256.md) | Use cudaMemcpyAsync in gemm grouped with kRequiresPrecomputation sche… | 2025-04-21 |  | gemm |
| [#2219](../sources/prs/cutlass/PR-2219.md) | [SM90] Change register allocation for TileN=208 to avoid spills | 2025-04-04 |  | gemm, tma |
| [#2220](../sources/prs/cutlass/PR-2220.md) | Set EpiTile correctly when TileN is not divisible by 32 | 2025-04-04 | epilogue-fusion | epilogue-fusion |
| [#2199](../sources/prs/cutlass/PR-2199.md) | fix_missing_stdint | 2025-03-26 |  | gemm |
| [#2167](../sources/prs/cutlass/PR-2167.md) | Fix sm100 gemm wrong static constexpr that breaks compilation on Windows | 2025-03-13 |  | gemm, tma |
| [#2172](../sources/prs/cutlass/PR-2172.md) | Fix SM90 beta=1 hang and stream-K launch errors | 2025-03-13 | tile-scheduling | gemm, tile-scheduling, tma |
| [#2161](../sources/prs/cutlass/PR-2161.md) | Blockwise Improvement and Programmatic Dependent Launch | 2025-03-10 | persistent-kernel, tile-scheduling | pdl, gdc, gemm |
| [#2134](../sources/prs/cutlass/PR-2134.md) | Flash MLA Support - Step 2 | 2025-02-26 |  | mla, tma |
| [#2139](../sources/prs/cutlass/PR-2139.md) | Blockwise and Groupwise GEMM for Blackwell and Improvements for Hopper | 2025-02-26 | warp-specialization, fine-grained-quantization | gemm, grouped-gemm, fp8 |
| [#2130](../sources/prs/cutlass/PR-2130.md) | Flash MLA support | 2025-02-24 |  | mla, tma |
| [#2123](../sources/prs/cutlass/PR-2123.md) | Hopper Grouped GEMM support for FP8 Accum | 2025-02-20 |  | fp8, gemm, grouped-gemm |
| [#2110](../sources/prs/cutlass/PR-2110.md) | Treat negative zero as equivalent to positive zero in sm90_sparse_gemm_compressor.hpp | 2025-02-13 |  | gemm |
| [#2095](../sources/prs/cutlass/PR-2095.md) | Improvements for: Groupwise scaling along M for FP8 gemm | 2025-02-10 | warp-specialization | fp8, gemm, tma |
| [#2037](../sources/prs/cutlass/PR-2037.md) | Groupwise scaling along M for FP8 gemm | 2025-01-13 | warp-specialization | fp8, gemm, tma |
| [#2033](../sources/prs/cutlass/PR-2033.md) | [EVT] Add support for Row/Col broadcast PtrArray | 2025-01-08 | epilogue-fusion, kernel-fusion | epilogue-fusion, kernel-fusion, tma |
| [#1993](../sources/prs/cutlass/PR-1993.md) | bugfix generic-k code in top-k with softmax | 2024-12-17 | epilogue-fusion, kernel-fusion | epilogue-fusion, gemm, kernel-fusion |
| [#1972](../sources/prs/cutlass/PR-1972.md) | Improve mixed dtype GEMM | 2024-12-04 | kernel-fusion | fp8, gemm, kernel-fusion |
| [#1932](../sources/prs/cutlass/PR-1932.md) | Blockwise Scaling for FP8 | 2024-11-08 | pipeline-stages, warp-specialization | fp8, gemm, pipeline-stages |
| [#1883](../sources/prs/cutlass/PR-1883.md) | Improve sm90 mixed dtype kernel | 2024-10-18 | epilogue-fusion | epilogue-fusion, fp8, gemm |
| [#1864](../sources/prs/cutlass/PR-1864.md) | Add GMMA shape m64n40k16 | 2024-10-11 |  | gemm |

<a id="deepseek-aideepgemm"></a>
## deepseek-ai/DeepGEMM
11 PRs

| PR | Title | Date | Techniques | Tags |
|-----|-------|------|------------|------|
| [#384](../sources/prs/DeepGEMM/PR-384.md) | Sync nv_dev with upstream #377 | 2026-07-17 | epilogue-fusion | attention, epilogue-fusion, fp4 |
| [#377](../sources/prs/DeepGEMM/PR-377.md) | Public release 26/07 | 2026-07-14 |  | attention, fp4, fp8 |
| [#369](../sources/prs/DeepGEMM/PR-369.md) | support indexer_n_heads = 16 | 2026-06-26 |  | attention, fp4, fp8 |
| [#364](../sources/prs/DeepGEMM/PR-364.md) | Multiple updates and refactorings | 2026-06-23 | epilogue-fusion | attention, epilogue-fusion, fp4 |
| [#347](../sources/prs/DeepGEMM/PR-347.md) | Multiple updates and refactorings | 2026-06-01 |  | attention, fp4, fp8 |
| [#349](../sources/prs/DeepGEMM/PR-349.md) | Sync nv_dev with upstream #347 | 2026-06-01 |  | attention, fp4, fp8 |
| [#343](../sources/prs/DeepGEMM/PR-343.md) | Fix a race condition in contiguous k-grouped GEMM where in-flight tensormaps are updated in-place | 2026-05-29 |  | fp8, gemm |
| [#328](../sources/prs/DeepGEMM/PR-328.md) | Sync nv_dev with upstream #316 (Mega MoE optimizations & benchmarks) | 2026-05-07 |  | attention, fp4, fp8 |
| [#324](../sources/prs/DeepGEMM/PR-324.md) | feat: add sm120 support for DeepGEMM  | 2026-05-01 |  | attention, fp4, fp8 |
| [#314](../sources/prs/DeepGEMM/PR-314.md) | fix: SM90 paged MQA logits IMA on next_n=4 (nv_dev rebase) | 2026-04-22 | epilogue-fusion, kernel-fusion | attention, epilogue-fusion, fp4 |
| [#304](../sources/prs/DeepGEMM/PR-304.md) | [Public release 26/04] Introducing Mega MoE, FP4 Indexer and other features/fixes | 2026-04-17 | kernel-fusion, fine-grained-quantization, communication-overlap | gemm, moe, fused-kernel |

<a id="flashinfer-aiflashinfer"></a>
## flashinfer-ai/flashinfer
644 PRs

| PR | Title | Date | Techniques | Tags |
|-----|-------|------|------------|------|
| [#4073](../sources/prs/flashinfer/PR-4073.md) | perf(jit): prune arch gencode that dispatch can never load | 2026-07-20 |  | gemm |
| [#4014](../sources/prs/flashinfer/PR-4014.md) | fix(cudnn): include attn scale in the prefill/decode graph-cache keys | 2026-07-17 |  | attention, decode, prefill |
| [#4025](../sources/prs/flashinfer/PR-4025.md) | fix(moe): restore legacy fused_moe.core imports for interleave helpers | 2026-07-17 | kernel-fusion | kernel-fusion, moe |
| [#4029](../sources/prs/flashinfer/PR-4029.md) | perf(gemm): Improve mm_fp4 cute-dsl autotune time via disk-cache and parallel compilation | 2026-07-17 |  | fp4, gemm |
| [#4033](../sources/prs/flashinfer/PR-4033.md) | fix(benchmark): don't fail non-trtllm mm_fp4 backends on unaligned n | 2026-07-17 |  | fp4, gemm |
| [#4040](../sources/prs/flashinfer/PR-4040.md) | feat: CuTe-DSL Fused SwiGLU NVFP4 Quantization | 2026-07-17 | kernel-fusion | fp4, kernel-fusion, nvfp4 |
| [#4001](../sources/prs/flashinfer/PR-4001.md) | perf(kda): use a one-warp/grouped recurrent decode hybrid | 2026-07-16 |  | decode |
| [#4004](../sources/prs/flashinfer/PR-4004.md) | Yanqinz/fix autotuner using autotuning plan from other runners | 2026-07-16 | kernel-fusion | kernel-fusion, moe |
| [#3979](../sources/prs/flashinfer/PR-3979.md) | chore: fix cute-dsl moe test | 2026-07-15 | kernel-fusion | kernel-fusion, moe |
| [#3980](../sources/prs/flashinfer/PR-3980.md) | feat(moe_ep): restructure MegaMoE kernel sources into kernel_src + latest CuTeDSL MegaMoE kernels with improved performance | 2026-07-15 | epilogue-fusion, kernel-fusion, persistent-kernel | epilogue-fusion, fp4, fp8 |
| [#3983](../sources/prs/flashinfer/PR-3983.md) | feat(moe): add B12x backends to unified MoE API and add corresponding tests | 2026-07-15 | kernel-fusion | kernel-fusion, moe |
| [#3955](../sources/prs/flashinfer/PR-3955.md) | feat: expose trtllm-gen block-sparse attention for decode | 2026-07-14 |  | attention, decode, flash-attention |
| [#3964](../sources/prs/flashinfer/PR-3964.md) | fix (test): mock MoE EP arch validation in split unit tests | 2026-07-14 |  | moe |
| [#3973](../sources/prs/flashinfer/PR-3973.md) | Update moe cubins to fix 2CTA hanging issue | 2026-07-14 |  | 2sm-cooperative, moe |
| [#3975](../sources/prs/flashinfer/PR-3975.md) |  mamba checkpointing SSU: two-kernel split + ring-buffer cache for checkpointing SSU | 2026-07-14 |  | gemm |
| [#3976](../sources/prs/flashinfer/PR-3976.md) | feat(moe): add deterministic CuTe DSL NVFP4 finalize mode | 2026-07-14 | kernel-fusion | fp4, gemm, grouped-gemm |
| [#3940](../sources/prs/flashinfer/PR-3940.md) | Add BF16 weight preparation and unified fuzzer coverage | 2026-07-13 | kernel-fusion | kernel-fusion, moe |
| [#3941](../sources/prs/flashinfer/PR-3941.md) | docs: add fused_moe LoRA delta APIs to rst | 2026-07-13 | kernel-fusion | kernel-fusion, moe |
| [#3942](../sources/prs/flashinfer/PR-3942.md) | docs: cover remaining @flashinfer_api symbols missing from rst | 2026-07-13 | kernel-fusion | decode, gemm, kernel-fusion |
| [#3947](../sources/prs/flashinfer/PR-3947.md) | perf(jit): drop dead SM12x gencode from SM10x-serving modules | 2026-07-13 | kernel-fusion | gemm, kernel-fusion, moe |
| [#3948](../sources/prs/flashinfer/PR-3948.md) | perf(gemm): update mm_fp4 cute-dsl tactic heuristic and autotune over a top-N ranked list | 2026-07-13 |  | fp4, gemm |
| [#3949](../sources/prs/flashinfer/PR-3949.md) | perf(attention): reuse TRT-LLM-gen counter buffers | 2026-07-13 |  | attention, decode, mla |
| [#3924](../sources/prs/flashinfer/PR-3924.md) | Add mm_fp8 into benchmark | 2026-07-11 |  | fp8, gemm |
| [#3925](../sources/prs/flashinfer/PR-3925.md) | fix(moe): allocate trtllm-gen expert_weights as bf16 in all launchers | 2026-07-11 | kernel-fusion | kernel-fusion, moe |
| [#3927](../sources/prs/flashinfer/PR-3927.md) | feat(quantization): support configurable scale-factor layouts for MXFP4 | 2026-07-11 |  | fp4, quantization |
| [#3908](../sources/prs/flashinfer/PR-3908.md) | Ameyn/gdn wy perf followup | 2026-07-10 |  | decode |
| [#3911](../sources/prs/flashinfer/PR-3911.md) | docs: fix 3 documentation issues from nightly doc-check | 2026-07-10 | kernel-fusion | kernel-fusion, moe |
| [#3912](../sources/prs/flashinfer/PR-3912.md) | fix: autotuner memory leak follow up | 2026-07-10 | kernel-fusion | attention, kernel-fusion, moe |
| [#3914](../sources/prs/flashinfer/PR-3914.md) | fix (test): release CUDA cache before prefill compile subprocess | 2026-07-10 |  | attention, prefill |
| [#3916](../sources/prs/flashinfer/PR-3916.md) | fix(mla): allow native-width block tables for TRTLLM-GEN | 2026-07-10 |  | attention, mla |
| [#3921](../sources/prs/flashinfer/PR-3921.md) | feat(attention): accept token-unit batch offsets on the cudnn prefill path | 2026-07-10 |  | attention, prefill |
| [#3922](../sources/prs/flashinfer/PR-3922.md) | fix(cute_dsl): migrate off APIs removed in nvidia-cutlass-dsl 4.6 (keeps 4.5.x compat) | 2026-07-10 |  | gemm |
| [#3888](../sources/prs/flashinfer/PR-3888.md) | fix(mla): enable cute-dsl/auto on SM103 + cluster-aware cutlass split_kv | 2026-07-09 |  | attention, mla |
| [#3891](../sources/prs/flashinfer/PR-3891.md) | Add FP8 groupwise MoE GEMM entry (cute SM120 backend) | 2026-07-09 | epilogue-fusion | epilogue-fusion, fp8, gemm |
| [#3892](../sources/prs/flashinfer/PR-3892.md) | feat(moe): wire in-kernel routing (FromLogits) into unified MoE API + routing axes in fuzzer | 2026-07-09 | kernel-fusion | fp4, kernel-fusion, moe |
| [#3897](../sources/prs/flashinfer/PR-3897.md) | feat(attention): enable SM121 (GB10/DGX Spark) for NVFP4 attention | 2026-07-09 |  | attention, fp4, nvfp4 |
| [#3880](../sources/prs/flashinfer/PR-3880.md) | fix(comm): make TRT-LLM reductions partial-warp safe | 2026-07-08 | kernel-fusion | kernel-fusion |
| [#3884](../sources/prs/flashinfer/PR-3884.md) | docs: fix 8 new documentation issues from nightly doc-check (v0.6.14) | 2026-07-08 | kernel-fusion | kernel-fusion, moe |
| [#3857](../sources/prs/flashinfer/PR-3857.md) | [feat] New CuTe DSL JIT kernels for FMHA: Dense & MXFP8 & NVFP4 formats | 2026-07-07 | persistent-kernel, tile-scheduling | attention, flash-attention, fp4 |
| [#3858](../sources/prs/flashinfer/PR-3858.md) | feat: SM100 CUTLASS NVFP4 SVDQuant fused GEMM (mm_nvfp4_svdquant) | 2026-07-07 | kernel-fusion | fp4, gemm, kernel-fusion |
| [#3863](../sources/prs/flashinfer/PR-3863.md) | fix(attention): clamp CTA_TILE_Q for fp8 h512 ragged prefill on 99KB-smem GPUs | 2026-07-07 |  | attention, fp8, prefill |
| [#3866](../sources/prs/flashinfer/PR-3866.md) | perf(gdn): optimize delta rule cp prefill for ultra low parallelism case | 2026-07-07 |  | prefill |
| [#3867](../sources/prs/flashinfer/PR-3867.md) | Add BF16 unified MoE conformance coverage | 2026-07-07 |  | moe |
| [#3870](../sources/prs/flashinfer/PR-3870.md) | fix(autotuner): time tactics with %globaltimer under Confidential Computing | 2026-07-07 |  | gemm |
| [#3871](../sources/prs/flashinfer/PR-3871.md) | Fix: graph-safe uniform multi-token decode on FA2 tensor-core path | 2026-07-07 |  | attention, decode, prefill |
| [#3872](../sources/prs/flashinfer/PR-3872.md) | doc: Adding owners for moe_ep | 2026-07-07 |  | moe |
| [#3844](../sources/prs/flashinfer/PR-3844.md) | fix(gdn): reject invalid sm120 tma lowering | 2026-07-06 |  | tma |
| [#3846](../sources/prs/flashinfer/PR-3846.md) | perf(moe): optimize one-sided EP4 NVFP4 dispatch | 2026-07-06 |  | fp4, moe, nvfp4 |
| [#3847](../sources/prs/flashinfer/PR-3847.md) | [feat] Add cutedsl split K for MXFP8 dense gemm | 2026-07-06 |  | fp8, gemm |
| [#3852](../sources/prs/flashinfer/PR-3852.md) | feat: MegaMoE Kernel integration in moe_ep | 2026-07-06 | epilogue-fusion, persistent-kernel, swizzling | epilogue-fusion, fp4, fp8 |
| [#3853](../sources/prs/flashinfer/PR-3853.md) | docs: install flashinfer-cubin from flashinfer.ai index; drop cubin PyPI upload | 2026-07-06 |  | gemm |
| [#3840](../sources/prs/flashinfer/PR-3840.md) | fix(moe): fail loudly instead of IMA when no routing tier covers (numExperts, topK) | 2026-07-05 | kernel-fusion | kernel-fusion, moe |
| [#3837](../sources/prs/flashinfer/PR-3837.md) | [Bugfix] Skip autotuning for small token counts on SM100 to prevent TMA crash | 2026-07-04 | kernel-fusion | kernel-fusion, moe, tma |
| [#3838](../sources/prs/flashinfer/PR-3838.md) | fix(attention): correct SM120 NVFP4 qk_correction layout, row-sum reduction, and lse | 2026-07-04 | epilogue-fusion | attention, epilogue-fusion, fp4 |
| [#3821](../sources/prs/flashinfer/PR-3821.md) | feat: make moe_ep (EP) part of the default install; drop nccl submodule | 2026-07-03 |  | moe |
| [#3822](../sources/prs/flashinfer/PR-3822.md) | Optimize FP4 V-scale dequant for h512 FA2 prefill | 2026-07-03 |  | attention, fp4, prefill |
| [#3804](../sources/prs/flashinfer/PR-3804.md) | Enable cuBLASLt BF16 GEMM on SM80+ | 2026-07-02 |  | gemm |
| [#3813](../sources/prs/flashinfer/PR-3813.md) | Feat/vllm moe ep api | 2026-07-02 |  | moe |
| [#3789](../sources/prs/flashinfer/PR-3789.md) | Add cold-L2 and CUDA graph to mm_bf16 API | 2026-07-01 |  | gemm |
| [#3794](../sources/prs/flashinfer/PR-3794.md) | feat(cute-dsl): sliding window and attention sinks for GQA decode (non-paged + paged) | 2026-07-01 | kernel-fusion | attention, decode, kernel-fusion |
| [#3058](../sources/prs/flashinfer/PR-3058.md) | Support lse in trtllm paged attn kernels | 2026-04-14 |  | attention, decode, flash-attention |
| [#3066](../sources/prs/flashinfer/PR-3066.md) | feat: Add b12x CuTe DSL fused MoE for SM120 | 2026-04-14 | kernel-fusion | fp4, kernel-fusion, moe |
| [#3051](../sources/prs/flashinfer/PR-3051.md) | feat: Add backend="b12x" for mm_fp4 on SM120 | 2026-04-13 |  | fp4, gemm |
| [#3032](../sources/prs/flashinfer/PR-3032.md) | fused_moe: pre-filter SM89 tactics with zero occupancy on SM120 Blackwell (fix review feedback on #2764) | 2026-04-10 | kernel-fusion | gemm, kernel-fusion, moe |
| [#3021](../sources/prs/flashinfer/PR-3021.md) | fix: extend moe alltoall top-k specializations | 2026-04-09 |  | moe |
| [#3024](../sources/prs/flashinfer/PR-3024.md) | [feat] Add routing_replay_out support to MoE kernels and Python API | 2026-04-09 | kernel-fusion | kernel-fusion, moe |
| [#3025](../sources/prs/flashinfer/PR-3025.md) | Prevent MoE autotuner buffer overflow on large token buckets | 2026-04-09 | kernel-fusion | kernel-fusion, moe |
| [#3026](../sources/prs/flashinfer/PR-3026.md) | perf: Port TRT-LLM SM120/SM121 FP4 CUTLASS GEMM optimizations. Add PDL | 2026-04-09 |  | fp4, gemm |
| [#3014](../sources/prs/flashinfer/PR-3014.md) | perf: Optimize CUTLASS MoE helper kernels for small-batch decode workloads | 2026-04-08 | kernel-fusion | decode, kernel-fusion, moe |
| [#3017](../sources/prs/flashinfer/PR-3017.md) | [chore] Install nvidia-cutlass-dsl[cu13] for cu130+ | 2026-04-08 |  | gemm |
| [#3001](../sources/prs/flashinfer/PR-3001.md) | [feat] Add blackwell GDN prefill kernel | 2026-04-07 | tile-scheduling | gated-delta-net, prefill, tile-scheduling |
| [#3007](../sources/prs/flashinfer/PR-3007.md) | fix: use sym_int64 for strides in rmsnorm CuTe DSL kernels to prevent int32 overflow | 2026-04-07 | kernel-fusion | kernel-fusion |
| [#3008](../sources/prs/flashinfer/PR-3008.md) | feat: add PDL support to rmsnorm_fp4quant and add_rmsnorm_fp4quant CuTe DSL kernels | 2026-04-07 |  | fp4 |
| [#2988](../sources/prs/flashinfer/PR-2988.md) | [Fmha] support nvfp4 output keepsMmaAb generation kernels | 2026-04-06 |  | attention, flash-attention, fp4 |
| [#2994](../sources/prs/flashinfer/PR-2994.md) |   Fix MXFP4/MXFP8 failures in SM120 FAST_BUILD and expand all_tiles[]                                                   | 2026-04-06 |  | fp4, fp8 |
| [#2996](../sources/prs/flashinfer/PR-2996.md) | fix: tinygemm2 hang issue due to barrier sync | 2026-04-06 |  | gemm |
| [#2982](../sources/prs/flashinfer/PR-2982.md) | feat(comm): add MOE Finalize/Reduction patterns to unified allreduce_fusion API | 2026-04-05 | kernel-fusion | kernel-fusion, moe |
| [#2984](../sources/prs/flashinfer/PR-2984.md) | fix: restore SM120 CUTLASS MoE tile candidate removed by #2927 (test_trtllm_cutlass_fused_moe.py) | 2026-04-05 | kernel-fusion | kernel-fusion, moe |
| [#2966](../sources/prs/flashinfer/PR-2966.md) | Fused moe all-reduce routed scaling factor + quant support | 2026-04-03 | kernel-fusion | kernel-fusion, moe |
| [#2974](../sources/prs/flashinfer/PR-2974.md) | test: skip unsupported mm_mxfp8 configurations on SM12x | 2026-04-03 |  | fp8, gemm |
| [#2954](../sources/prs/flashinfer/PR-2954.md) | Only swizzle on v block scale; rename kv_block_scales to kv_cache_sf | 2026-04-02 | swizzling | attention, block-scale, decode |
| [#2960](../sources/prs/flashinfer/PR-2960.md) | Update NVSHMEM interface to use NVSHMEM4Py instead of custom bindings | 2026-04-02 |  | gemm |
| [#2963](../sources/prs/flashinfer/PR-2963.md) | test: xfail cuDNN FP8 prefill on Blackwell with CUDA <= 12.9 | 2026-04-02 |  | attention, fp8, prefill |
| [#2965](../sources/prs/flashinfer/PR-2965.md) | Add flashinfer.fused_rmsnorm_silu() with native kernel backend | 2026-04-02 | kernel-fusion | kernel-fusion |
| [#2940](../sources/prs/flashinfer/PR-2940.md) | CuTe DSL FP4 GEMM Heuristic | 2026-04-01 |  | fp4, gemm |
| [#2942](../sources/prs/flashinfer/PR-2942.md) | [Perf] Refactor MoE autotuning to set valid topk ids in routed MoE tuning | 2026-04-01 | kernel-fusion | kernel-fusion, moe |
| [#2945](../sources/prs/flashinfer/PR-2945.md) | fix: use float instead of double in sampling binary search to avoid FP64 bottleneck on SM103 | 2026-04-01 |  | gemm |
| [#2926](../sources/prs/flashinfer/PR-2926.md) | feat: add Relu2 (squared ReLU) activation support in CUTLASS MoE backend | 2026-03-31 | epilogue-fusion | epilogue-fusion, gemm, moe |
| [#2927](../sources/prs/flashinfer/PR-2927.md) | feat: SM121 (GB10) tile filtering and autotuner robustness | 2026-03-31 |  | gemm |
| [#2916](../sources/prs/flashinfer/PR-2916.md) | fix: Fix autotuner crash on meta-device tensor in trtllm_fp4_block_scale_routed_moe | 2026-03-30 | kernel-fusion | block-scale, fp4, kernel-fusion |
| [#2913](../sources/prs/flashinfer/PR-2913.md) | [NVIDIA] fix(jit): enable GDC for CUTLASS fused MoE PDL — prevent random crashes on SM12x | 2026-03-29 | kernel-fusion | fp8, gemm, kernel-fusion |
| [#2908](../sources/prs/flashinfer/PR-2908.md) | feat(gdn): state checkpointing in chunk_gated_delta_rule | 2026-03-28 |  | gated-delta-net, prefill, tma |
| [#2901](../sources/prs/flashinfer/PR-2901.md) | feat: add pdl support for cute dsl mla decode kernel support | 2026-03-27 |  | attention, decode, fp8 |
| [#2902](../sources/prs/flashinfer/PR-2902.md) | feat: add MXFP8 GEMM support for SM120 | 2026-03-27 |  | fp8, gemm |
| [#2904](../sources/prs/flashinfer/PR-2904.md) | perf: Optimize CuTe-DSL fp4 and fp8 quantization kernels | 2026-03-27 |  | fp4, fp8, nvfp4 |
| [#2898](../sources/prs/flashinfer/PR-2898.md) | fix: snap weight_scale_vec_size to handle block_scale_interleave padding for SM120 | 2026-03-26 | kernel-fusion | block-scale, fp4, kernel-fusion |
| [#2872](../sources/prs/flashinfer/PR-2872.md) | fix: add cute dsl moe utils to AOT | 2026-03-24 |  | moe |
| [#2876](../sources/prs/flashinfer/PR-2876.md) | [fix] bugfix 2856: Fix pre-allocated out shape check in trtllm_batch_decode_with_kv_cache_mla for q_len_per_req > 1 | 2026-03-24 |  | attention, decode, mla |
| [#2882](../sources/prs/flashinfer/PR-2882.md) | Fix silent bug with FP8 per tensor non-gated MoE | 2026-03-24 | kernel-fusion | fp8, kernel-fusion, moe |
| [#2864](../sources/prs/flashinfer/PR-2864.md) | Add support for Relu2 in BF16 fused MoE | 2026-03-23 | kernel-fusion | gemm, kernel-fusion, moe |
| [#2865](../sources/prs/flashinfer/PR-2865.md) | Mamba SSU: horizontal MTP kernel (+ DSTATE=96 support) | 2026-03-23 |  | gemm |
| [#2853](../sources/prs/flashinfer/PR-2853.md) | fix: int32 overflow in `trtllm_fp4_block_scale_moe` causing "Unsupported hidden state scale shape" for EP32+ configs | 2026-03-22 | kernel-fusion | block-scale, fp4, kernel-fusion |
| [#2836](../sources/prs/flashinfer/PR-2836.md) | [Fmha] Sparse MLA decode kernel selection heuristics | 2026-03-20 |  | decode, flash-attention, mla |
| [#2838](../sources/prs/flashinfer/PR-2838.md) | feat: Add CuTe-DSL backend for NVFP4 quantization | 2026-03-20 |  | fp4, nvfp4, quantization |
| [#2842](../sources/prs/flashinfer/PR-2842.md) | perf: Optimize GDN MTP decode kernel (v15) — eliminate ilp=1 fallback… | 2026-03-20 |  | decode |
| [#2844](../sources/prs/flashinfer/PR-2844.md) | read real strides for kv and block scale | 2026-03-20 |  | decode, flash-attention, prefill |
| [#2821](../sources/prs/flashinfer/PR-2821.md) | fix: Autotuner _find_nearest_profile non-power-of-2 num_tokens, create launchers for all supported tileN in trtllm fused MoE | 2026-03-19 | kernel-fusion | kernel-fusion, moe |
| [#2828](../sources/prs/flashinfer/PR-2828.md) | [Spark unit test] Adjust tolerance for test_xqa, test_logits_processor | 2026-03-19 |  | attention |
| [#2811](../sources/prs/flashinfer/PR-2811.md) | CuteDSL MoE fix redundant output buffer zeroing | 2026-03-18 | kernel-fusion | gemm, grouped-gemm, kernel-fusion |
| [#2802](../sources/prs/flashinfer/PR-2802.md) | [fix] Bugfix 1367: fix VariableBlockSparseAttention buffer overflow by dynamically resizing kv_lens_buffer | 2026-03-17 |  | attention, decode, prefill |
| [#2805](../sources/prs/flashinfer/PR-2805.md) | [CuTe DSL] Add modular FMHA prefill and MLA decode attention kernels | 2026-03-17 | epilogue-fusion, kernel-fusion, persistent-kernel | attention, decode, epilogue-fusion |
| [#2810](../sources/prs/flashinfer/PR-2810.md) | feat(gdn): add padding index guard for bf16 decode kernel | 2026-03-17 |  | decode |
| [#2792](../sources/prs/flashinfer/PR-2792.md) | feat: Support padding tokens with seqlen=0 for rope+quant+kv cache update fusion kernel | 2026-03-16 | kernel-fusion | attention, kernel-fusion |
| [#2798](../sources/prs/flashinfer/PR-2798.md) | Upgrade cutlass 4.2.1 -> 4.4.2 | 2026-03-16 |  | gemm, moe, tma |
| [#2777](../sources/prs/flashinfer/PR-2777.md) | perf: Performance tune cute dsl RMSNorm variants | 2026-03-13 | kernel-fusion | kernel-fusion |
| [#2780](../sources/prs/flashinfer/PR-2780.md) | fix(jit): enable GDC for CUTLASS GEMM PDL — SM100 flag only | 2026-03-13 |  | gemm |
| [#2781](../sources/prs/flashinfer/PR-2781.md) | tests: skip sliding window + fp8 to prevent hang in fmha_v2 unit tests | 2026-03-13 |  | attention, flash-attention, fp8 |
| [#2770](../sources/prs/flashinfer/PR-2770.md) | feat: Expose TRT-LLM FMHA style paged KV Cache and page table layout | 2026-03-12 |  | attention, decode, flash-attention |
| [#2750](../sources/prs/flashinfer/PR-2750.md) | [Spark unit test debugging] Fix for tests/attention/test_trtllm_gen_mla.py | 2026-03-11 |  | attention, mla |
| [#2751](../sources/prs/flashinfer/PR-2751.md) | [Spark unit test debugging] Fix for tests/gemm/test_groupwise_scaled_gemm_fp8.py | 2026-03-11 |  | fp8, gemm |
| [#2752](../sources/prs/flashinfer/PR-2752.md) | [feat] Add air top-p algorithm | 2026-03-11 |  | gemm |
| [#2757](../sources/prs/flashinfer/PR-2757.md) | feat: Add FP4 KV cache quant/dequant kernels  | 2026-03-11 |  | fp4, quantization |
| [#2738](../sources/prs/flashinfer/PR-2738.md) | Support for MXFP4 and NVFP4 group GEMMs on GeForce and Spark | 2026-03-10 |  | fp4, fp8, gemm |
| [#2739](../sources/prs/flashinfer/PR-2739.md) | Support in-place update for `trtllm_fp8_block_scale_moe` | 2026-03-10 | kernel-fusion | block-scale, fp8, kernel-fusion |
| [#2740](../sources/prs/flashinfer/PR-2740.md) | misc: Update gemm/batched gemm cubins from trtllm-gen, gemm header refactor | 2026-03-10 | kernel-fusion | gemm, kernel-fusion, moe |
| [#2743](../sources/prs/flashinfer/PR-2743.md) | Add cute dsl mla decode op | 2026-03-10 |  | attention, decode, fp8 |
| [#2744](../sources/prs/flashinfer/PR-2744.md) | [feat] Add 2048 experts and 32 Top K  | 2026-03-10 | kernel-fusion | kernel-fusion, moe |
| [#2725](../sources/prs/flashinfer/PR-2725.md) | fix: Add SM120 (RTX Blackwell desktop) support for NVFP4 MoE kernels | 2026-03-09 | kernel-fusion | fp4, kernel-fusion, moe |
| [#2727](../sources/prs/flashinfer/PR-2727.md) | [gdn] support non-contiguous state for decoding | 2026-03-09 |  | decode |
| [#2716](../sources/prs/flashinfer/PR-2716.md) | fix(jit): GEMM kernels produce NaN under concurrency — missing GDC flags cause PDL synchronization barriers to compile as no-ops | 2026-03-07 |  | gemm |
| [#2702](../sources/prs/flashinfer/PR-2702.md) | Add NVFP4 KV cache quantization support for SM100 | 2026-03-06 |  | attention, decode, flash-attention |
| [#2707](../sources/prs/flashinfer/PR-2707.md) | feat: Add support for TRTLLM MXFP8 non-gated MoE with ReLU2 | 2026-03-06 | kernel-fusion | fp8, kernel-fusion, moe |
| [#2709](../sources/prs/flashinfer/PR-2709.md) | Mamba2 SSD Combined Forward Pass (Blackwell CuTe DSL Kernel) | 2026-03-06 | tile-scheduling | tile-scheduling |
| [#2700](../sources/prs/flashinfer/PR-2700.md) | Add varlen and speculative decoding support to selective state update | 2026-03-05 |  | gemm |
| [#2670](../sources/prs/flashinfer/PR-2670.md) | fix: reduce smem allocation for tinygemm2 kernel in SM120 | 2026-03-03 |  | gemm |
| [#2677](../sources/prs/flashinfer/PR-2677.md) | feat: add support for more MLA head dimensions | 2026-03-03 |  | attention, flash-attention, mla |
| [#2679](../sources/prs/flashinfer/PR-2679.md) | feat(gdn): add BF16 state kernel with MTP support beyond T>4 with intermediate caching. | 2026-03-03 |  | decode |
| [#2666](../sources/prs/flashinfer/PR-2666.md) | benchmarks: Add FP8 input / BF16 output in ragged prefill benchmark | 2026-03-02 |  | attention, fp8, prefill |
| [#2667](../sources/prs/flashinfer/PR-2667.md) | perf: Update trtllm-gen batched GEMM kernels - faster, more NVFP4 tile dims, MXFP8 with relu2 act | 2026-03-02 |  | fp4, fp8, gemm |
| [#2661](../sources/prs/flashinfer/PR-2661.md) | feat: implement deterministic topk | 2026-03-01 |  | gemm |
| [#2653](../sources/prs/flashinfer/PR-2653.md) | [feat] trtllm-gen mxfp8 gemm | 2026-02-28 | kernel-fusion | attention, decode, fp4 |
| [#2654](../sources/prs/flashinfer/PR-2654.md) | fix: Add fused MOE and GEMM AOT modules for SM121 | 2026-02-28 | kernel-fusion | gemm, kernel-fusion, moe |
| [#2660](../sources/prs/flashinfer/PR-2660.md) | feat: support mxfp4 & mxfp8 entrypoint for blackwell cutedsl dense gemm | 2026-02-28 |  | fp4, fp8, gemm |
| [#2650](../sources/prs/flashinfer/PR-2650.md) | Enable sm120f compilation | 2026-02-27 |  | fp4, quantization |
| [#2642](../sources/prs/flashinfer/PR-2642.md) | [fp8_blockwise]Fix int32 overflow in TRTLLM fused MoE activation kernel | 2026-02-26 | kernel-fusion | fp8, kernel-fusion, moe |
| [#2644](../sources/prs/flashinfer/PR-2644.md) | feat: FP32 dtype output for BF16 matmuls (CUTLASS & cuDNN) | 2026-02-26 |  | gemm |
| [#2645](../sources/prs/flashinfer/PR-2645.md) | int16 Block-Scaled State and Stochastic Rounding for SSU (mamba) | 2026-02-26 |  | block-scale |
| [#2635](../sources/prs/flashinfer/PR-2635.md) | benchmark: Add MXFP4/MXFP8 quantization mode support to FP4 MoE benchmark | 2026-02-25 |  | fp4, fp8, moe |
| [#2627](../sources/prs/flashinfer/PR-2627.md) | fix: trtllm_mxint4_block_scale_moe unit test to index output list | 2026-02-24 | kernel-fusion | block-scale, kernel-fusion, moe |
| [#2628](../sources/prs/flashinfer/PR-2628.md) | benchmark: Enable speculative decode microbenchmarking for paged decode | 2026-02-24 |  | attention, decode |
| [#2629](../sources/prs/flashinfer/PR-2629.md) | fix: cute dsl nvfp4 moe routing index error | 2026-02-24 | kernel-fusion | fp4, kernel-fusion, moe |
| [#2631](../sources/prs/flashinfer/PR-2631.md) | fix: add SM121 support to SM120 version guards | 2026-02-24 | kernel-fusion | attention, decode, flash-attention |
| [#2618](../sources/prs/flashinfer/PR-2618.md) | perf(gdn): optimize MTP kernel with ILP rows and SMEM v caching | 2026-02-22 |  | decode |
| [#2619](../sources/prs/flashinfer/PR-2619.md) | feat: add pool+indices support to gated_delta_rule_decode_pretranspose (bf16 path)  | 2026-02-22 |  | decode, gated-delta-net |
| [#2607](../sources/prs/flashinfer/PR-2607.md) | support qk_nope_head_dim for 192 check for GLM-5 | 2026-02-21 |  | attention, mla |
| [#2610](../sources/prs/flashinfer/PR-2610.md) | Ameyn/gdn bf16 tolerance parallel reduction | 2026-02-21 |  | decode |
| [#2605](../sources/prs/flashinfer/PR-2605.md) | [bugfix] Fix FilteredTopK overflow correctness | 2026-02-20 |  | gemm |
| [#2587](../sources/prs/flashinfer/PR-2587.md) | feat: trtllm tinygemm2 in flashinfer as bf16 routergemm | 2026-02-19 |  | gemm |
| [#2588](../sources/prs/flashinfer/PR-2588.md) | Perf: Optimize GDN decode pretranspose kernel for all batch sizes | 2026-02-19 |  | decode |
| [#2591](../sources/prs/flashinfer/PR-2591.md) | Mamba SSU: better automatic kernel selection + algorithm selection optionally exposed to the user. | 2026-02-19 |  | gemm |
| [#2581](../sources/prs/flashinfer/PR-2581.md) | Implement `cutlass_fused_moe` mxfp8 | 2026-02-18 | kernel-fusion | fp8, gemm, kernel-fusion |
| [#2585](../sources/prs/flashinfer/PR-2585.md) | tests: add bias testing to nvfp4 moe | 2026-02-18 | kernel-fusion | fp4, kernel-fusion, moe |
| [#2573](../sources/prs/flashinfer/PR-2573.md) | [Bug] Fix spark unit test failures for test_add_rmsnorm_fp4_quant_cute_dsl | 2026-02-17 |  | fp4 |
| [#2574](../sources/prs/flashinfer/PR-2574.md) | feat: add is_sm12x_supported() helper for SM12x family detection | 2026-02-17 |  | attention, flash-attention, gemm |
| [#2564](../sources/prs/flashinfer/PR-2564.md) | fix: W4A8 autotune crash in cutlass_fused_moe profiler workspace | 2026-02-14 | kernel-fusion | kernel-fusion, moe |
| [#2559](../sources/prs/flashinfer/PR-2559.md) | fix: allow fmha_v2_prefill_deepseek on SM121 (DGX Spark) | 2026-02-13 |  | attention, flash-attention, prefill |
| [#2560](../sources/prs/flashinfer/PR-2560.md) | fix: guard CUTLASS FMHA against SM12x and fix fmha_v2 SM121a check | 2026-02-13 |  | attention, flash-attention, prefill |
| [#2543](../sources/prs/flashinfer/PR-2543.md) | misc: point triton blackwell-ptxas to local cuda ptxas | 2026-02-12 |  | gemm |
| [#2547](../sources/prs/flashinfer/PR-2547.md) | feat: Enable TRTLLM-Gen Skip-Softmax attention for MLA | 2026-02-12 |  | attention, flash-attention, mla |
| [#2549](../sources/prs/flashinfer/PR-2549.md) | Add gen_gemm_sm100_module_cutlass_mxfp8 to jit-cache | 2026-02-12 |  | fp8, gemm |
| [#2538](../sources/prs/flashinfer/PR-2538.md) | tests: bmm_fp8 for SM110 | 2026-02-11 |  | fp8, gemm |
| [#2540](../sources/prs/flashinfer/PR-2540.md) | feat: cute dsl mmfp4 for blackwell | 2026-02-11 |  | fp4, gemm |
| [#2533](../sources/prs/flashinfer/PR-2533.md) | fix: include fp8_blockscale_gemm_90 in AOT jit-cache | 2026-02-10 |  | fp8, gemm |
| [#2536](../sources/prs/flashinfer/PR-2536.md) | fallback to fa2 (instead of fa3) for unsupported configuration (bf16 Q, Fp8 KV) | 2026-02-10 |  | fp8 |
| [#2525](../sources/prs/flashinfer/PR-2525.md) | feat: BF16 GEMM benchmarking support | 2026-02-09 |  | gemm |
| [#2530](../sources/prs/flashinfer/PR-2530.md) | pick fa2 for BatchDecodeWithPagedKVCacheWrapper auto backend | 2026-02-09 |  | attention, decode |
| [#2520](../sources/prs/flashinfer/PR-2520.md) | Support NVFP4 KV cache decode on SM120 | 2026-02-08 |  | attention, decode, fp4 |
| [#2521](../sources/prs/flashinfer/PR-2521.md) | Feat/gdn decode pooled | 2026-02-08 |  | decode |
| [#2505](../sources/prs/flashinfer/PR-2505.md) | Feat: Trtllm-gen MxFP8 MoE integration | 2026-02-06 | kernel-fusion | fp8, gemm, kernel-fusion |
| [#2509](../sources/prs/flashinfer/PR-2509.md) | perf: cache cudaGetDeviceProperties in gdn_prefill to avoid per-call overhead | 2026-02-06 |  | prefill |
| [#2498](../sources/prs/flashinfer/PR-2498.md) | Ameyn/gdn decode cutedsl kernel | 2026-02-05 |  | decode, prefill |
| [#2503](../sources/prs/flashinfer/PR-2503.md) | refactor: Port upstream CUTLASS fixes and refactor grouped_gemm_nt_masked GEMM module location | 2026-02-05 |  | gemm, grouped-gemm |
| [#2495](../sources/prs/flashinfer/PR-2495.md) | fix: add support check for gemm config for cutlass moe | 2026-02-04 |  | gemm, moe |
| [#2476](../sources/prs/flashinfer/PR-2476.md) | fix: blockscale moe routine supports non-DS routing | 2026-02-03 |  | moe |
| [#2477](../sources/prs/flashinfer/PR-2477.md) | feat: Add TRTLLM-Gen Skip-Softmax kernels for prefill and decode | 2026-02-03 |  | attention, decode, flash-attention |
| [#2479](../sources/prs/flashinfer/PR-2479.md) | fix: Fix memory bandwidth calculation in MLA benchmarks | 2026-02-03 |  | attention, mla |
| [#2460](../sources/prs/flashinfer/PR-2460.md) | perf: add fp4 GEMM tile configs and streamK scheduler for SM120 | 2026-02-02 |  | fp4, gemm |
| [#2462](../sources/prs/flashinfer/PR-2462.md) | feat: Support Fused MoE non gated Relu2 NVFP4 & FP8 and support Nemotron, fixed | 2026-02-02 | kernel-fusion | fp4, fp8, gemm |
| [#2464](../sources/prs/flashinfer/PR-2464.md) | feat: Add MXFP8 GEMM mm_mxfp8 (cutlass) | 2026-02-02 |  | fp8, gemm |
| [#2456](../sources/prs/flashinfer/PR-2456.md) | fix: fix illegal memory access for NaN input in sampling kernels | 2026-01-31 |  | gemm |
| [#2443](../sources/prs/flashinfer/PR-2443.md) | Add cute-dsl backends to mxfp[8,4]_quantization for future refactor | 2026-01-30 |  | fp4, fp8, quantization |
| [#2444](../sources/prs/flashinfer/PR-2444.md) | MTP for mamba  | 2026-01-30 |  | gemm |
| [#2445](../sources/prs/flashinfer/PR-2445.md) | bugfix: fix stub generation directory in fused_moe module | 2026-01-30 | kernel-fusion | kernel-fusion, moe |
| [#2446](../sources/prs/flashinfer/PR-2446.md) | feat: Add TRTLLM fmha_v2 library for SM90 attention with Skip-Softmax  | 2026-01-30 | epilogue-fusion, kernel-fusion | attention, epilogue-fusion, flash-attention |
| [#2432](../sources/prs/flashinfer/PR-2432.md) | fix: Sampling: CUDA Graph fix | 2026-01-29 |  | gemm |
| [#2441](../sources/prs/flashinfer/PR-2441.md) | fix: Fix NaN output in mxfp8_quantize for very small input values | 2026-01-29 |  | fp8, quantization |
| [#2428](../sources/prs/flashinfer/PR-2428.md) | refactor: refactoring cuda code to cute-dsl (part 1) | 2026-01-28 | kernel-fusion | fp4, kernel-fusion |
| [#2421](../sources/prs/flashinfer/PR-2421.md) | refactor: simplify fp4 rmsnorm | 2026-01-27 |  | fp4 |
| [#2422](../sources/prs/flashinfer/PR-2422.md) | refactor: reduce hopper's gdn prefill compilation time and fix docstring. | 2026-01-27 | tile-scheduling | prefill, tile-scheduling, tma |
| [#2416](../sources/prs/flashinfer/PR-2416.md) | feat: update trtllm-gen MoE cubins | 2026-01-26 |  | gemm, moe, tma |
| [#2415](../sources/prs/flashinfer/PR-2415.md) | Remove cudaMalloc/Free in GDN prefill kernel | 2026-01-25 |  | prefill |
| [#2405](../sources/prs/flashinfer/PR-2405.md) | perf: improve gdn decode cute-dsl kernels | 2026-01-23 |  | decode |
| [#2387](../sources/prs/flashinfer/PR-2387.md) | A Blackwell-optimized version of selective_state_update (decode) | 2026-01-22 | warp-specialization, pipeline-stages, double-buffering | tcgen05, decode |
| [#2398](../sources/prs/flashinfer/PR-2398.md) | feat: cuteDSL fp4 moe for better DSR1 performance. | 2026-01-22 | kernel-fusion, pipeline-stages | fp4, gemm, grouped-gemm |
| [#2404](../sources/prs/flashinfer/PR-2404.md) | perf: mm_fp4 heuristic prioritizes CUTLASS over cuDNN on SM103 | 2026-01-22 |  | fp4, gemm |
| [#2395](../sources/prs/flashinfer/PR-2395.md) | feat: Add output_both_sf_layouts option to add_rmsnorm_fp4quant API | 2026-01-21 |  | fp4 |
| [#2376](../sources/prs/flashinfer/PR-2376.md) | feat: BF16 GEMM using cuDNN backend | 2026-01-20 |  | gemm |
| [#2378](../sources/prs/flashinfer/PR-2378.md) | bugfix: hotfix of PR 2366 (mamba kernel) | 2026-01-20 |  | gemm |
| [#2380](../sources/prs/flashinfer/PR-2380.md) | fix: ensure each CTA processes full numHeadsQPerKv for trtllm decode kernel | 2026-01-20 |  | decode, flash-attention |
| [#2385](../sources/prs/flashinfer/PR-2385.md) | fix: In-place Residual Update for add_rmsnorm_fp4quant | 2026-01-20 |  | fp4 |
| [#2370](../sources/prs/flashinfer/PR-2370.md) | feat: [Qwen3-Next] Add Cute DSL GDN decode kernel and  tests | 2026-01-18 |  | decode, prefill |
| [#2366](../sources/prs/flashinfer/PR-2366.md) | Enable fp16/bf16/f32 support for selective_state_update (mamba) | 2026-01-16 |  | gemm |
| [#2362](../sources/prs/flashinfer/PR-2362.md) | benchmarks: Add norm and quantization routines to microbenchmark harness. | 2026-01-15 |  | quantization |
| [#2352](../sources/prs/flashinfer/PR-2352.md) | Added the cudnn backend Ragged KV Cache wrapper | 2026-01-14 |  | attention, prefill |
| [#2343](../sources/prs/flashinfer/PR-2343.md) | Optimize quantization function in large problem size | 2026-01-13 | kernel-fusion | fp4, kernel-fusion, moe |
| [#2334](../sources/prs/flashinfer/PR-2334.md) | Support both 3D and 4D kv_cache shapes in MLA APIs | 2026-01-12 |  | mla |
| [#2327](../sources/prs/flashinfer/PR-2327.md) | [perf] Improve gemm_fp8_nt_groupwise (cutlass backend) by 10-40% for batch sizes <= 32 | 2026-01-11 |  | fp8, gemm |
| [#2328](../sources/prs/flashinfer/PR-2328.md) | fix: guard batchWarpReduceSum with ENABLE_FP8 to fix compilation without FP8 | 2026-01-11 |  | fp8 |
| [#2330](../sources/prs/flashinfer/PR-2330.md) | feat: expose swizzled_input_sf parameter for CUTLASS fused MOE | 2026-01-11 | kernel-fusion, swizzling | kernel-fusion, moe, swizzling |
| [#2325](../sources/prs/flashinfer/PR-2325.md) | bugfix: fix multi-cta top-k implementation when k value is different for different row | 2026-01-10 |  | gemm |
| [#2323](../sources/prs/flashinfer/PR-2323.md) | [ML3] Optimized Router Gemm | 2026-01-09 |  | gemm |
| [#2308](../sources/prs/flashinfer/PR-2308.md) | Fix: FilteredTopKUnifiedKernel read value out of length | 2026-01-08 |  | gemm |
| [#2302](../sources/prs/flashinfer/PR-2302.md) | fix: Decode benchmark's fa2_tc uses backend=fa2 in wrapper | 2026-01-07 |  | attention, decode |
| [#2303](../sources/prs/flashinfer/PR-2303.md) | [Perf][Feature] Add SM103-specific schedulers for NVFP4 CUTLASS kernels | 2026-01-07 |  | fp4, gemm, nvfp4 |
| [#2304](../sources/prs/flashinfer/PR-2304.md) | feat: Support Fused MoE non gated Relu2 NVFP4 & FP8 and support Nemotron | 2026-01-07 | kernel-fusion | fp4, fp8, gemm |
| [#2301](../sources/prs/flashinfer/PR-2301.md) | Selective State Update kernel (mamba) | 2026-01-06 |  | gemm |
| [#2281](../sources/prs/flashinfer/PR-2281.md) | feat: IdType indices in sampling kernels | 2026-01-02 |  | gemm |
| [#2279](../sources/prs/flashinfer/PR-2279.md) | [WIP] Refactor: simplify torch -> cute-dsl boilerplate and enable tvm-ffi for cute-dsl kernels | 2026-01-01 |  | fp4 |
| [#2276](../sources/prs/flashinfer/PR-2276.md) | feat: add GDN Attention | 2025-12-31 | tile-scheduling | attention, prefill, tile-scheduling |
| [#2277](../sources/prs/flashinfer/PR-2277.md) | Tiny fix bench tgv gemm | 2025-12-31 |  | gemm |
| [#2268](../sources/prs/flashinfer/PR-2268.md) | [performance]optimize for nvfp4 | 2025-12-25 | kernel-fusion | fp4, kernel-fusion, nvfp4 |
| [#2265](../sources/prs/flashinfer/PR-2265.md) | [TRTLLM-Gen Fmha] add optimized trtllm-gen decode kernels for high throughput + speculative decoding | 2025-12-24 |  | attention, decode, flash-attention |
| [#2266](../sources/prs/flashinfer/PR-2266.md) | test: use .float() in in F.cosine_similarity() in bmm_fp8 test | 2025-12-24 |  | fp8, gemm |
| [#2260](../sources/prs/flashinfer/PR-2260.md) | fix: Add global scale support and optional output allocation for RMSNorm+FP4Quant fusion kernels | 2025-12-23 | kernel-fusion | fp4, kernel-fusion |
| [#2261](../sources/prs/flashinfer/PR-2261.md) | Fix CUTLASS FP8 gemm correctness issue on SM120/SM121 for shapes where N is not divisible by ScaleGranularityN. | 2025-12-23 |  | fp8, gemm |
| [#2255](../sources/prs/flashinfer/PR-2255.md) | fix: support int64 IdType for RoPE part argument in `rope_quantize_fp8_append_paged_kv_cache` | 2025-12-22 |  | attention, fp8, quantization |
| [#2256](../sources/prs/flashinfer/PR-2256.md) | feat: Add support for bmm mxfp8 | 2025-12-22 |  | fp8, gemm |
| [#2254](../sources/prs/flashinfer/PR-2254.md) | feat: support non-contiguous query for trtllm-gen attention backend | 2025-12-21 |  | attention, flash-attention |
| [#2243](../sources/prs/flashinfer/PR-2243.md) | feat: RMSNorm/Fused RMSNorm + FP8 Quantization kernels | 2025-12-19 | kernel-fusion | fp8, kernel-fusion, quantization |
| [#2244](../sources/prs/flashinfer/PR-2244.md) | Remove cudaStreamSynchronize from gemm_groupwise_sm120.cuh for CUDA graph compatibility | 2025-12-19 |  | gemm |
| [#2245](../sources/prs/flashinfer/PR-2245.md) | test: Fix MNNVL tests to skip when container lacks SYS_PTRACE capability | 2025-12-19 |  | moe |
| [#2247](../sources/prs/flashinfer/PR-2247.md) | feat: Support numLocalTokens=0 for moe All-to-all | 2025-12-19 |  | moe |
| [#2237](../sources/prs/flashinfer/PR-2237.md) | [feat] Integrate SGLang concat_mla_k kernel into flashinfer | 2025-12-18 |  | mla |
| [#2241](../sources/prs/flashinfer/PR-2241.md) | Fp8 attention are now part of cuDNN 9.17.1 | 2025-12-18 |  | attention, fp8, prefill |
| [#2233](../sources/prs/flashinfer/PR-2233.md) | feat: Fused RMSNorm + FP4 Quantization Kernels in CuTe-DSL | 2025-12-17 | kernel-fusion | fp4, kernel-fusion, quantization |
| [#2234](../sources/prs/flashinfer/PR-2234.md) | fix: add DeepSeek routing for Bf16xBf16 and MxIntxBf16 TRT-LLM Gen MoE | 2025-12-17 | kernel-fusion | kernel-fusion, moe |
| [#2235](../sources/prs/flashinfer/PR-2235.md) | refactor: pull trtllm-gen batch-gemm/gemm headers from artifactory; update tma descriptor shape init | 2025-12-17 | kernel-fusion | gemm, kernel-fusion, moe |
| [#2217](../sources/prs/flashinfer/PR-2217.md) | feat: Support unpadded output hidden size for trtllm_fp4_block_scale_moe | 2025-12-14 | kernel-fusion | block-scale, fp4, kernel-fusion |
| [#2214](../sources/prs/flashinfer/PR-2214.md) | misc: support checks for gemm | 2025-12-13 |  | gemm |
| [#2215](../sources/prs/flashinfer/PR-2215.md) | feat: further optimize top-k and add fused top-k page construction kernels for DSA | 2025-12-13 | kernel-fusion | kernel-fusion |
| [#2211](../sources/prs/flashinfer/PR-2211.md) | Move the run function definition out of BatchedGemmInterface | 2025-12-12 |  | gemm |
| [#2193](../sources/prs/flashinfer/PR-2193.md) | feat: unit-test and api change, w4a8 grouped-gemm fused MoE for SM90 | 2025-12-10 | kernel-fusion | gemm, grouped-gemm, kernel-fusion |
| [#2194](../sources/prs/flashinfer/PR-2194.md) | Permute page table in benchmarking | 2025-12-10 |  | attention |
| [#2190](../sources/prs/flashinfer/PR-2190.md) | Fix for moe on sm110 | 2025-12-09 |  | gemm, moe |
| [#2181](../sources/prs/flashinfer/PR-2181.md) | Rename noauxtc to fused_topk_deepseek | 2025-12-05 | kernel-fusion | kernel-fusion, moe |
| [#2175](../sources/prs/flashinfer/PR-2175.md) | fix: compile flags for trtllm fmha_v2  | 2025-12-04 |  | attention, flash-attention, prefill |
| [#2163](../sources/prs/flashinfer/PR-2163.md) | refactor: Move mla code from decode.py to mla.py and add to documentation | 2025-12-03 |  | attention, decode, mla |
| [#2165](../sources/prs/flashinfer/PR-2165.md) | Add data type check for deepseek fp4 moe | 2025-12-03 | kernel-fusion | fp4, kernel-fusion, moe |
| [#2171](../sources/prs/flashinfer/PR-2171.md) | Fix gemm allreduce two shot | 2025-12-03 |  | gemm |
| [#2157](../sources/prs/flashinfer/PR-2157.md) | fix xqa mha_sm90.cu | 2025-12-02 |  | gemm |
| [#2159](../sources/prs/flashinfer/PR-2159.md) | feat: MxInt4 x Bf16 TRT-LLM Gen MoE support | 2025-12-02 | kernel-fusion | fp4, gemm, kernel-fusion |
| [#2148](../sources/prs/flashinfer/PR-2148.md) | Enable Hopper FA3 FP8 attention in decode.py | 2025-11-28 |  | attention, decode, fp8 |
| [#2149](../sources/prs/flashinfer/PR-2149.md) | enable sm103 moe dsl backend | 2025-11-28 |  | gemm, moe |
| [#2142](../sources/prs/flashinfer/PR-2142.md) | feat: TRTLLM FMHAv2 backend for ctx attention | 2025-11-25 | epilogue-fusion, kernel-fusion | attention, epilogue-fusion, flash-attention |
| [#2137](../sources/prs/flashinfer/PR-2137.md) | fix: some bugs of headDim 256 trtllm-gen fmha kernels.  | 2025-11-24 |  | attention, flash-attention |
| [#2138](../sources/prs/flashinfer/PR-2138.md) | feat: add trtllm-gen per-tensor sparseMla kernels. | 2025-11-24 |  | attention, decode, flash-attention |
| [#2134](../sources/prs/flashinfer/PR-2134.md) | fix(trtllm): reset negative strideBatch to 0 for ragged KV layout to … | 2025-11-23 |  | attention, flash-attention |
| [#2131](../sources/prs/flashinfer/PR-2131.md) | make DeepGEMM swapAB available for linear gemm SM90 | 2025-11-22 |  | fp8, gemm |
| [#2126](../sources/prs/flashinfer/PR-2126.md) | fix flaky xqa test | 2025-11-21 |  | attention, mla |
| [#2129](../sources/prs/flashinfer/PR-2129.md) | fix: Fix bench_mm_fp8.py | 2025-11-21 |  | fp8 |
| [#2130](../sources/prs/flashinfer/PR-2130.md) | A unified API for the MNNVL and single-node/multi-GPU AllReduce kernels. | 2025-11-21 | kernel-fusion | kernel-fusion |
| [#2117](../sources/prs/flashinfer/PR-2117.md) | update xqa license | 2025-11-20 |  | mla, tma |
| [#2118](../sources/prs/flashinfer/PR-2118.md) | Refactor trtllm_mnnvl_allreduce | 2025-11-20 |  | gemm |
| [#2119](../sources/prs/flashinfer/PR-2119.md) | perf: bunch of features and optimizations for top-k (sampling + sparse attention) | 2025-11-20 |  | attention |
| [#2125](../sources/prs/flashinfer/PR-2125.md) | feat: support variable sequence length in decode kernel of trtllm-gen attention | 2025-11-20 |  | attention, decode, flash-attention |
| [#2109](../sources/prs/flashinfer/PR-2109.md) | feat: support more head dim in RoPE kernel | 2025-11-19 |  | attention |
| [#2110](../sources/prs/flashinfer/PR-2110.md) | add tensor scale input for xqa | 2025-11-19 |  | attention, decode, mla |
| [#2111](../sources/prs/flashinfer/PR-2111.md) | refactor: update fa3 codebase and fix hopper unittest [part 1] | 2025-11-19 | epilogue-fusion | attention, epilogue-fusion, fp8 |
| [#2114](../sources/prs/flashinfer/PR-2114.md) | feature: make the LSE returned by MLA support base 2 or e  #2113 | 2025-11-19 |  | attention, mla |
| [#2102](../sources/prs/flashinfer/PR-2102.md) | Port TRT-LLM communication kernels to flashinfer | 2025-11-18 |  | moe |
| [#2105](../sources/prs/flashinfer/PR-2105.md) | enable xqa speculative decoding | 2025-11-18 |  | attention, decode |
| [#2100](../sources/prs/flashinfer/PR-2100.md) | [DSR1] Added MLA test | 2025-11-17 |  | attention, mla |
| [#2097](../sources/prs/flashinfer/PR-2097.md) | refactor: update dpsk fused_moe test [2] | 2025-11-16 | kernel-fusion | fp8, kernel-fusion, moe |
| [#2095](../sources/prs/flashinfer/PR-2095.md) | perf: enable pdl for cutlass fp4 gemm | 2025-11-15 |  | fp4, gemm |
| [#2090](../sources/prs/flashinfer/PR-2090.md) | refactor: pass hopper deepgemm include directory through python | 2025-11-14 | kernel-fusion | gemm, kernel-fusion, moe |
| [#2092](../sources/prs/flashinfer/PR-2092.md) | perf: TRT-LLM Gen finalize kernel optimization | 2025-11-14 | kernel-fusion | kernel-fusion, moe |
| [#2084](../sources/prs/flashinfer/PR-2084.md) | [API change] Allow using torch.Tensor for scales for trtllm-gen attention | 2025-11-13 |  | attention, decode, flash-attention |
| [#2088](../sources/prs/flashinfer/PR-2088.md) | refactor: update dpsk fused_moe test [1] | 2025-11-13 | kernel-fusion | fp8, kernel-fusion, moe |
| [#2079](../sources/prs/flashinfer/PR-2079.md) | [Feature] Support batch prefill for POD Attention | 2025-11-12 |  | attention, decode, prefill |
| [#2081](../sources/prs/flashinfer/PR-2081.md) | enable xqa fp8 output | 2025-11-12 |  | attention, decode, fp8 |
| [#2082](../sources/prs/flashinfer/PR-2082.md) | Patch sm103 for 3xfp4 moe generation | 2025-11-12 | kernel-fusion | fp4, kernel-fusion, moe |
| [#2075](../sources/prs/flashinfer/PR-2075.md) | unittest: improve the efficiency of xqa unittests | 2025-11-11 |  | attention, decode |
| [#2076](../sources/prs/flashinfer/PR-2076.md) | fix: fix test_trtllm_gen_attention when max_seq_len < page_size | 2025-11-11 |  | attention |
| [#2070](../sources/prs/flashinfer/PR-2070.md) | feat: BF16 GEMM using CUTLASS backend for SM100 | 2025-11-10 |  | gemm |
| [#2072](../sources/prs/flashinfer/PR-2072.md) | [Test] Optimize test_trtllm_gen_fused_moe.py | 2025-11-10 | kernel-fusion | kernel-fusion, moe |
| [#2058](../sources/prs/flashinfer/PR-2058.md) | perf: Optimize helper max/minmax function in sampling.cuh | 2025-11-07 |  | gemm |
| [#2061](../sources/prs/flashinfer/PR-2061.md) | Fix moe fp8 failure for sm121 | 2025-11-07 |  | fp8, moe |
| [#2062](../sources/prs/flashinfer/PR-2062.md) | Fix: several bugs/issues with trtllm-gen attention kernels.  | 2025-11-07 |  | attention, flash-attention |
| [#2063](../sources/prs/flashinfer/PR-2063.md) | perf: TRT-LLM MoE Block-FP8 activation optimization | 2025-11-07 | kernel-fusion | fp8, kernel-fusion, moe |
| [#2051](../sources/prs/flashinfer/PR-2051.md) | Add support for topkPacked input in block-level renormalize | 2025-11-06 | kernel-fusion | kernel-fusion, moe |
| [#2052](../sources/prs/flashinfer/PR-2052.md) | test: Skip test_fp8_quantize.py on Hopper | 2025-11-06 |  | fp8, quantization |
| [#2053](../sources/prs/flashinfer/PR-2053.md) | feat: add xqa mla backend | 2025-11-06 |  | attention, decode, mla |
| [#2055](../sources/prs/flashinfer/PR-2055.md) | misc: Add XQA decode to microbenchmark for sm90 and sm120 | 2025-11-06 |  | attention, decode |
| [#2044](../sources/prs/flashinfer/PR-2044.md) | perf: improve sampling/mask/softmax performance (part 1/2) | 2025-11-05 |  | tma |
| [#2047](../sources/prs/flashinfer/PR-2047.md) | Rebase FP8 SM100 Cutlass FMHA Attention to main (original PR#1238) | 2025-11-05 |  | attention, flash-attention, fp8 |
| [#2048](../sources/prs/flashinfer/PR-2048.md) | Fix dtype of output scales from mnnvl_moe_alltoallv_prepare_without_allgather | 2025-11-05 |  | moe |
| [#2049](../sources/prs/flashinfer/PR-2049.md) | [BUG] Fix trtllm-gen fp4 moe renormalize routing | 2025-11-05 | kernel-fusion | fp4, kernel-fusion, moe |
| [#2033](../sources/prs/flashinfer/PR-2033.md) | use scalar for kv_scale in xqa | 2025-11-04 |  | attention, decode, mla |
| [#2035](../sources/prs/flashinfer/PR-2035.md) | Added an initial implementation of Q and KV Cache in fp8 and to use t… | 2025-11-04 |  | attention, fp8, prefill |
| [#2037](../sources/prs/flashinfer/PR-2037.md) | feat: Add flashinfer.rope.rope_quantize_fp8_append_paged_kv_cache (fused RoPE + Q + KV cache, supports MLA/GQA/MHA)  | 2025-11-04 | kernel-fusion | attention, fp8, kernel-fusion |
| [#2025](../sources/prs/flashinfer/PR-2025.md) | perf: Speed up fp4 quantization for small batch with swizzling for cutlass MoE | 2025-11-03 | swizzling | fp4, moe, quantization |
| [#2028](../sources/prs/flashinfer/PR-2028.md) | [NVIDIA] Thor & Spark Support | 2025-11-03 |  | gemm |
| [#2029](../sources/prs/flashinfer/PR-2029.md) | feat: suitable_auto_backends to prune auto backends, bmm_fp8 refactor, heuristic_func intake | 2025-11-03 |  | fp8, gemm |
| [#2030](../sources/prs/flashinfer/PR-2030.md) | Enable renormalize(naive) routing for fp8 per-tensor | 2025-11-03 | kernel-fusion | fp8, kernel-fusion, moe |
| [#2019](../sources/prs/flashinfer/PR-2019.md) | [DSV3] Optimized Router Gemm | 2025-10-31 |  | fp4, gemm |
| [#2020](../sources/prs/flashinfer/PR-2020.md) | update trtllm cutlass moe  | 2025-10-31 | epilogue-fusion, kernel-fusion, warp-specialization | epilogue-fusion, fp4, fp8 |
| [#2011](../sources/prs/flashinfer/PR-2011.md) | Feature: Support non-gated activation in cutlass fused MoE nvfp4 | 2025-10-30 | kernel-fusion | fp4, kernel-fusion, moe |
| [#2012](../sources/prs/flashinfer/PR-2012.md) | fix: Enable SM121 for mm_fp4 | 2025-10-30 |  | fp4, gemm |
| [#2014](../sources/prs/flashinfer/PR-2014.md) | [feat] Refactor trtllmgen MOE and add Bf16 trtllmgen moe | 2025-10-30 | kernel-fusion | gemm, kernel-fusion, moe |
| [#2001](../sources/prs/flashinfer/PR-2001.md) | feat: add xqa backend and completes NHD/HND coverage for trtllm-gen/xqa backend | 2025-10-29 |  | attention, decode, flash-attention |
| [#2002](../sources/prs/flashinfer/PR-2002.md) | Fix trtllm-gen attention illegal memory access | 2025-10-29 |  | attention, decode |
| [#1994](../sources/prs/flashinfer/PR-1994.md) | minor fix for xqa | 2025-10-28 |  | attention, mla |
| [#1995](../sources/prs/flashinfer/PR-1995.md) | Bugfix: Change get() -> GetDLTensorPtr() in cutlass FusedMoE validations | 2025-10-28 | kernel-fusion | kernel-fusion, moe |
| [#1999](../sources/prs/flashinfer/PR-1999.md) | unittest: Add head dim 256 test cases and mark as xfail | 2025-10-28 |  | attention |
| [#1982](../sources/prs/flashinfer/PR-1982.md) | fix: correct PDL parameter handling in RopeQuantize kernel | 2025-10-26 |  | attention, fp8, quantization |
| [#1979](../sources/prs/flashinfer/PR-1979.md) | feat: Add backend='auto' to mm_fp4 and enable autotune for backend='cudnn' | 2025-10-25 |  | fp4, gemm |
| [#1980](../sources/prs/flashinfer/PR-1980.md) | feat: autotune tile_tokens_dim in trtllm-gen MOE | 2025-10-25 | kernel-fusion | kernel-fusion, moe |
| [#1976](../sources/prs/flashinfer/PR-1976.md) | fix: Make attention microbenchmark correctly use page table | 2025-10-24 |  | attention |
| [#1978](../sources/prs/flashinfer/PR-1978.md) | fix: Skipping attention sink Blackwell test outside of Blackwell | 2025-10-24 |  | attention |
| [#1969](../sources/prs/flashinfer/PR-1969.md) | feat: enable deepgemm jit for fp8 block-scale on SM90 | 2025-10-23 |  | block-scale, fp8, gemm |
| [#1973](../sources/prs/flashinfer/PR-1973.md) | Feature: Add support for L40 FusedMoE in cutlass path | 2025-10-23 | kernel-fusion, warp-specialization | gemm, kernel-fusion, moe |
| [#1959](../sources/prs/flashinfer/PR-1959.md) | fix: Add cutlass as an mm_fp4 backend in compute capability 12.0 in benchmark code | 2025-10-21 |  | fp4 |
| [#1961](../sources/prs/flashinfer/PR-1961.md) | Fix: Verify scales are not None for Cutlass FP8 FusedMoE | 2025-10-21 | kernel-fusion | fp8, kernel-fusion, moe |
| [#1963](../sources/prs/flashinfer/PR-1963.md) | fix: ensure SM120/121 SFA/SFB contiguity | 2025-10-21 |  | gemm |
| [#1954](../sources/prs/flashinfer/PR-1954.md) | Feature: Support Relu2 activation in fused MoE | 2025-10-20 | epilogue-fusion, kernel-fusion | epilogue-fusion, gemm, kernel-fusion |
| [#1955](../sources/prs/flashinfer/PR-1955.md) | Update trtllm-gen fused moe routing kernel and add more kernels | 2025-10-20 | kernel-fusion | gemm, kernel-fusion, moe |
| [#1942](../sources/prs/flashinfer/PR-1942.md) | Add realistic bench for persistent kernel  | 2025-10-17 | persistent-kernel | attention, persistent-kernel |
| [#1926](../sources/prs/flashinfer/PR-1926.md) | Add layernorm op for inputs of mixed dtype | 2025-10-14 |  | gemm |
| [#1927](../sources/prs/flashinfer/PR-1927.md) | silu_and_mul nvfp4 quanization fusion rework | 2025-10-14 | kernel-fusion | fp4, kernel-fusion, nvfp4 |
| [#1924](../sources/prs/flashinfer/PR-1924.md) | MLA RoPE + quantization fused kernel: shape generalization for MHA / GQA | 2025-10-13 | kernel-fusion | attention, fp8, kernel-fusion |
| [#1912](../sources/prs/flashinfer/PR-1912.md) | fix: Fix trtllm-gen prefill IMA when batch_size==1 | 2025-10-10 |  | attention, flash-attention, prefill |
| [#1882](../sources/prs/flashinfer/PR-1882.md) | feat: Add FP4 TRTLLM-Gen throughput MOE batched gemms | 2025-10-07 | kernel-fusion | fp4, gemm, kernel-fusion |
| [#1883](../sources/prs/flashinfer/PR-1883.md) | misc: fix some B200 GEMM bench | 2025-10-07 |  | fp8, gemm |
| [#1878](../sources/prs/flashinfer/PR-1878.md) | Tune kernel compilation parameters for https://github.com/flashinfer-ai/flashinfer/pull/1850  | 2025-10-06 |  | attention, flash-attention, tma |
| [#1862](../sources/prs/flashinfer/PR-1862.md) | raise error for group_gemm_fp8_nt_groupwise then num_groups > 1 on sm120/121 | 2025-10-04 |  | fp4, fp8, gemm |
| [#1865](../sources/prs/flashinfer/PR-1865.md) | Bugfix: fix o_strides in persistent kernel  | 2025-10-04 | persistent-kernel | attention, persistent-kernel |
| [#1850](../sources/prs/flashinfer/PR-1850.md) | Add head_dim=64 for tcgen05 tcgen05 flash-attention implementation | 2025-10-03 | warp-specialization, persistent-kernel, pipeline-stages | tcgen05, flash-attention, attention |
| [#1826](../sources/prs/flashinfer/PR-1826.md) | Bugfix: Fix data hazard in persistent reduce | 2025-10-01 | persistent-kernel | attention, persistent-kernel |
| [#1829](../sources/prs/flashinfer/PR-1829.md) | feat: trtrllm-gen global scaled FP8 GEMMs | 2025-10-01 | kernel-fusion | fp8, gemm, kernel-fusion |
| [#1831](../sources/prs/flashinfer/PR-1831.md) | Update the routing for TRTLLMGEN to support kimi k2 and qwen | 2025-10-01 | kernel-fusion | kernel-fusion, moe |
| [#1835](../sources/prs/flashinfer/PR-1835.md) | [Quantization] Add per-expert global scaling factor for fp4 batched quantize | 2025-10-01 |  | fp4, quantization |
| [#1810](../sources/prs/flashinfer/PR-1810.md) | tests: Update support for tgv_gemm to SM100 only and add to ut | 2025-09-30 |  | gemm |
| [#1812](../sources/prs/flashinfer/PR-1812.md) | tests: upgrade cutlass, fix import and skip non-SM100 cutedsl two shot allreduce | 2025-09-30 |  | gemm |
| [#1817](../sources/prs/flashinfer/PR-1817.md) | fix: fp4 moe on sm120 | 2025-09-30 | kernel-fusion | fp4, kernel-fusion, moe |
| [#1819](../sources/prs/flashinfer/PR-1819.md) | feat:enable fp8 blockscale moe for fused cultass for sm90 | 2025-09-30 | kernel-fusion | fp8, gemm, kernel-fusion |
| [#1809](../sources/prs/flashinfer/PR-1809.md) | Support checks PoC | 2025-09-29 |  | fp4, gemm |
| [#1769](../sources/prs/flashinfer/PR-1769.md) | feat: add xqa fp8 mha and fp8 kv cache | 2025-09-25 |  | attention, fp8, mla |
| [#1771](../sources/prs/flashinfer/PR-1771.md) | Waive / disable test_mla_decode_kernel.py::test_mla_decode_kernel for not sm80  | 2025-09-25 |  | decode, mla |
| [#1774](../sources/prs/flashinfer/PR-1774.md) | Masked batch nvfp4 quantization | 2025-09-25 |  | fp4, nvfp4, quantization |
| [#1764](../sources/prs/flashinfer/PR-1764.md) | fix: fix cannot import name 'cuda' from 'cuda' in CUDA13 | 2025-09-24 |  | gemm |
| [#1766](../sources/prs/flashinfer/PR-1766.md) | Added xfail for mx_fp4 matmul on SM120 | 2025-09-24 |  | fp4, gemm |
| [#1767](../sources/prs/flashinfer/PR-1767.md) | tests: skip non SM100/103 for grouped deepgemm | 2025-09-24 |  | fp8, gemm |
| [#1768](../sources/prs/flashinfer/PR-1768.md) | add test case for trtllm gen fused moe with kimi k2 problem sizes | 2025-09-24 | kernel-fusion | kernel-fusion, moe |
| [#1757](../sources/prs/flashinfer/PR-1757.md) | fix: should pass global_override_indptr_cpu in fast_decode_plan param list | 2025-09-23 |  | decode |
| [#1752](../sources/prs/flashinfer/PR-1752.md) | tests: xfail attention sink UT for sliding window + non causal case | 2025-09-22 |  | attention |
| [#1754](../sources/prs/flashinfer/PR-1754.md) | tests: xfail moe quantization classes mxfp8_bf16 UTs on sm103  | 2025-09-22 | kernel-fusion | fp8, kernel-fusion, moe |
| [#1755](../sources/prs/flashinfer/PR-1755.md) | Fix tests/test_trtllm_gen_attention.py::test_trtllm_batch_prefill, ::test_trtllm_batch_decode mismatch error | 2025-09-22 |  | attention, decode, prefill |
| [#1745](../sources/prs/flashinfer/PR-1745.md) | feat: port fast_decode_plan from sgl | 2025-09-21 |  | decode |
| [#1723](../sources/prs/flashinfer/PR-1723.md) | Fix DeepSeek quality for TRTLLM fused MoE routing | 2025-09-19 | kernel-fusion | kernel-fusion, moe |
| [#1724](../sources/prs/flashinfer/PR-1724.md) | bugfix: partially fix tests/test_trtllm_gen_fused_moe.py unit test failure | 2025-09-19 | kernel-fusion | kernel-fusion, moe |
| [#1725](../sources/prs/flashinfer/PR-1725.md) | TVM: support TVM binding for GroupedGemm | 2025-09-19 |  | fp8, gemm, grouped-gemm |
| [#1727](../sources/prs/flashinfer/PR-1727.md) | fix: put sampling kernel launch into macro | 2025-09-19 |  | gemm |
| [#1695](../sources/prs/flashinfer/PR-1695.md) | [cute_dsl] add gemm + all reduce (two_shot) | 2025-09-18 | kernel-fusion, tma-multicast | cute-dsl, gemm, kernel-fusion |
| [#1716](../sources/prs/flashinfer/PR-1716.md) | perf: Add tuning config for cutlass moe for a hardware | 2025-09-18 | kernel-fusion | kernel-fusion, moe |
| [#1698](../sources/prs/flashinfer/PR-1698.md) | hotfix: Hotfix for `test_pod_kernels.py` on B300 | 2025-09-17 |  | gemm |
| [#1706](../sources/prs/flashinfer/PR-1706.md) | feat: Benchmark mm_fp4 mxfp4 support and gemm autotune support.  Restore mm_fp4 API behavior | 2025-09-17 |  | fp4, gemm |
| [#1707](../sources/prs/flashinfer/PR-1707.md) | bugfix: increase workspace to make trtllm gen attention unit test pass | 2025-09-17 |  | attention |
| [#1710](../sources/prs/flashinfer/PR-1710.md) | test: skip the unsupported test cases for sm120/121 | 2025-09-17 | kernel-fusion | attention, fp4, fp8 |
| [#1681](../sources/prs/flashinfer/PR-1681.md) | perf: improve attention of tcgen05 flash-attention | 2025-09-16 | warp-specialization, persistent-kernel | tcgen05, flash-attention, attention |
| [#1685](../sources/prs/flashinfer/PR-1685.md) | perf: Port the separate reduce kernel mode from trtllm. | 2025-09-16 |  | attention, flash-attention |
| [#1688](../sources/prs/flashinfer/PR-1688.md) | gemm: Enabled alpha with the mx_fp4 format | 2025-09-16 |  | fp4, gemm |
| [#1694](../sources/prs/flashinfer/PR-1694.md) | Update deepgemm backend for 103a | 2025-09-16 |  | fp8, gemm |
| [#1696](../sources/prs/flashinfer/PR-1696.md) | Support Kimi-K2 for TRT: templatize number of experts | 2025-09-16 | kernel-fusion | kernel-fusion, moe |
| [#1679](../sources/prs/flashinfer/PR-1679.md) | [misc] add a wrapper class for attention sink jit args | 2025-09-15 |  | attention |
| [#1682](../sources/prs/flashinfer/PR-1682.md) | Update TGV GEMM default kernel and TGV code cleanup. | 2025-09-15 |  | gemm |
| [#1668](../sources/prs/flashinfer/PR-1668.md) | TGV GEMM as a BF16 backend alternative to cuBLAS | 2025-09-14 | persistent-kernel, tile-scheduling | gemm, fp8, tcgen05 |
| [#1677](../sources/prs/flashinfer/PR-1677.md) | Support output signals for overlapping for cutedsl gemm | 2025-09-14 |  | gemm |
| [#1670](../sources/prs/flashinfer/PR-1670.md) | feat: Add `variant.OutputTransform()` to decode kernels | 2025-09-11 |  | attention, decode |
| [#1674](../sources/prs/flashinfer/PR-1674.md) | test: better fp8 quantization init for fused_moe test | 2025-09-11 | kernel-fusion | fp8, kernel-fusion, moe |
| [#1675](../sources/prs/flashinfer/PR-1675.md) | feat: Batch-size invariant FA2 Prefill & Decode | 2025-09-11 |  | attention, decode, prefill |
| [#1665](../sources/prs/flashinfer/PR-1665.md) | test: update fused_moe test to random scale factor | 2025-09-10 | kernel-fusion | kernel-fusion, moe |
| [#1666](../sources/prs/flashinfer/PR-1666.md) | [Hotfix] `test_fp4_quantize.py` failure on sm103 | 2025-09-10 |  | fp4, quantization |
| [#1667](../sources/prs/flashinfer/PR-1667.md) | Refactor Blackwell unit test scripts | 2025-09-10 |  | attention, gemm, moe |
| [#1656](../sources/prs/flashinfer/PR-1656.md) | Add benchmark for MLARopeQuantize | 2025-09-09 |  | fp8, mla, quantization |
| [#1661](../sources/prs/flashinfer/PR-1661.md) | perf&bugfix: skip kv-tile computation out of sliding window in FA2; fix __syncthreads in mergestate | 2025-09-09 |  | attention, decode, fp8 |
| [#1548](../sources/prs/flashinfer/PR-1548.md) | perf: Enable SplitK and fix tile-scheduling for moe fp4 fused moe | 2025-09-05 | tile-scheduling, fine-grained-quantization | moe, fp4, gemm |
| [#1636](../sources/prs/flashinfer/PR-1636.md) | test: pytest.mark.xfail on deepgemm | 2025-09-05 |  | fp8, gemm |
| [#1640](../sources/prs/flashinfer/PR-1640.md) | bugfix: Fix FLOPS calculation for bench_trtllm_gen_mla.py | 2025-09-05 |  | mla |
| [#1643](../sources/prs/flashinfer/PR-1643.md) | fix: zero-init workspace buffer for trtllm-gen fmha | 2025-09-05 |  | attention, flash-attention, mla |
| [#1644](../sources/prs/flashinfer/PR-1644.md) | Added mx_fp4 support using the cudnn backend | 2025-09-05 |  | fp4, gemm |
| [#1633](../sources/prs/flashinfer/PR-1633.md) | feat: add support of fp4_batched_quantize | 2025-09-04 |  | fp4, quantization |
| [#1635](../sources/prs/flashinfer/PR-1635.md) | fix: pass workspace for trtllm-gen attention | 2025-09-04 |  | attention, decode, prefill |
| [#1631](../sources/prs/flashinfer/PR-1631.md) | bugfix: trtllm-gen fmha sm101 and sm100 compatibility | 2025-09-03 |  | flash-attention |
| [#1622](../sources/prs/flashinfer/PR-1622.md) | bugfix: collect all modules to aot | 2025-09-02 | kernel-fusion | attention, decode, gemm |
| [#1614](../sources/prs/flashinfer/PR-1614.md) | bugfix: fix merge_attention_state in BatchAttention w/ gqa-group-size in Qwen family | 2025-09-01 | persistent-kernel | attention, persistent-kernel |
| [#1615](../sources/prs/flashinfer/PR-1615.md) | perf: Fix the tactic sorting in TrtllmGenBatchedGemmRunner::getValidConfigIndices | 2025-09-01 |  | gemm |
| [#1608](../sources/prs/flashinfer/PR-1608.md) | feat: initial support for SM103, SM110, SM120, SM121 | 2025-08-30 | kernel-fusion | attention, flash-attention, fp4 |
| [#1609](../sources/prs/flashinfer/PR-1609.md) | feat: cutlass fp4 gemm bringup for SM120 & SM121 | 2025-08-30 |  | fp4, gemm, quantization |
| [#1610](../sources/prs/flashinfer/PR-1610.md) | feat: cutlass fp8 gemm bringup for SM120 & SM121 | 2025-08-30 |  | fp8, gemm |
| [#1611](../sources/prs/flashinfer/PR-1611.md) | bugfix: fix fp4 quantization with 8x4 scale factor layout | 2025-08-30 |  | fp4, quantization |
| [#1596](../sources/prs/flashinfer/PR-1596.md) | bugfix: fix fused-temperature softmax IMA issue | 2025-08-28 | kernel-fusion | kernel-fusion, tma |
| [#1597](../sources/prs/flashinfer/PR-1597.md) | bugfix: fix the register overflow issue for topk renorm kernels on blackwell | 2025-08-28 |  | gemm |
| [#1599](../sources/prs/flashinfer/PR-1599.md) | bugfix: fix unittest test_fp8_quantize | 2025-08-28 |  | fp8, quantization |
| [#1601](../sources/prs/flashinfer/PR-1601.md) | feat: Enable MnnvlMemory (for alltoallv) on B200 | 2025-08-28 |  | gemm |
| [#1589](../sources/prs/flashinfer/PR-1589.md) | fix: limit the number of nvcc threads for each kernel | 2025-08-27 |  | gemm |
| [#1590](../sources/prs/flashinfer/PR-1590.md) | fix: Improve TRTLLM attention kernel out_dtype unit test | 2025-08-27 |  | attention, decode, prefill |
| [#1577](../sources/prs/flashinfer/PR-1577.md) | bugfix: update trtllm-gen gemm kernel names | 2025-08-26 |  | gemm |
| [#1578](../sources/prs/flashinfer/PR-1578.md) | feat: Support for inferring out_dtype from out.dtype for TRTLLM attention kernel | 2025-08-26 |  | attention, decode, prefill |
| [#1581](../sources/prs/flashinfer/PR-1581.md) | refactor: Expose calculate_tile_tokens_dim function | 2025-08-26 | kernel-fusion | kernel-fusion, moe |
| [#1582](../sources/prs/flashinfer/PR-1582.md) | bugfix: Fix arg passing to TORCH_CHECK and TORCH_WARN macros | 2025-08-26 | kernel-fusion | gemm, kernel-fusion, moe |
| [#1584](../sources/prs/flashinfer/PR-1584.md) | fix: semaphoress must be at the fixed range in workspace buffer on trtllm_gen attention | 2025-08-26 |  | attention, flash-attention |
| [#1585](../sources/prs/flashinfer/PR-1585.md) | bugfix: Fix test_fp4_quantize test bug | 2025-08-26 |  | fp4, quantization |
| [#1567](../sources/prs/flashinfer/PR-1567.md) | Backend: downgrade trtllm-gen kernel to cuda-12 | 2025-08-25 |  | attention, decode, flash-attention |
| [#1571](../sources/prs/flashinfer/PR-1571.md) | bugfix: fix cuda version guard macros | 2025-08-25 | kernel-fusion | kernel-fusion, moe |
| [#1573](../sources/prs/flashinfer/PR-1573.md) | update trtllm-gen fp4 autotuner and routing | 2025-08-25 | kernel-fusion | fp4, kernel-fusion, moe |
| [#1559](../sources/prs/flashinfer/PR-1559.md) | bugfix: fix persistent attention kernel correctness on blackwell | 2025-08-24 | persistent-kernel | attention, persistent-kernel |
| [#1565](../sources/prs/flashinfer/PR-1565.md) | fix: separate out fp4 lib into sm90 and sm100 versions, add oob checking in fused moe | 2025-08-24 | kernel-fusion | fp4, kernel-fusion, moe |
| [#1540](../sources/prs/flashinfer/PR-1540.md) | feat: Add fp8-qkv, fp16/bf16 output MHA | 2025-08-22 | kernel-fusion | attention, decode, fp8 |
| [#1545](../sources/prs/flashinfer/PR-1545.md) | fix trtllm_allreduce_fusion twoshot register problem. | 2025-08-22 | kernel-fusion | kernel-fusion |
| [#1547](../sources/prs/flashinfer/PR-1547.md) | perf: replace cudaGetDeviceProperties with cudaDeviceGetAttribute | 2025-08-22 | kernel-fusion | kernel-fusion, moe |
| [#1550](../sources/prs/flashinfer/PR-1550.md) | Add mnnvl_moe_alltoallv_prepare_without_allgather | 2025-08-22 |  | moe |
| [#1530](../sources/prs/flashinfer/PR-1530.md) | bugfix: Fix compile error for undefined swizzle enum. | 2025-08-21 | kernel-fusion, swizzling | fp8, kernel-fusion, moe |
| [#1533](../sources/prs/flashinfer/PR-1533.md) | bugfix: Fix Persistent kernel precision for masked output  | 2025-08-21 | persistent-kernel | attention, persistent-kernel, prefill |
| [#1534](../sources/prs/flashinfer/PR-1534.md) | Remove cuda-python from dependency and check at runtime | 2025-08-21 |  | gemm |
| [#1535](../sources/prs/flashinfer/PR-1535.md) | Add sm check for sm100 only cutlass/trtllm kernel | 2025-08-21 |  | gemm |
| [#1537](../sources/prs/flashinfer/PR-1537.md) | feat: Integrate TRTLLM varlen kernel for deepseek R1 prefill  | 2025-08-21 |  | attention, flash-attention, prefill |
| [#1518](../sources/prs/flashinfer/PR-1518.md) | backend: Refactor trtllm-gen fmha metainfo loading | 2025-08-20 |  | attention, decode, flash-attention |
| [#1521](../sources/prs/flashinfer/PR-1521.md) | refactor fp4 masked gemm cute-dsl implementation and add manual cache | 2025-08-20 |  | fp4, gemm |
| [#1523](../sources/prs/flashinfer/PR-1523.md) | Fix linking errors with CUDA 13 | 2025-08-20 |  | gemm |
| [#1525](../sources/prs/flashinfer/PR-1525.md) | Add GeGLU support to trtllm-gen NVFP4 Fused MoE Kernel | 2025-08-20 | kernel-fusion | fp4, gemm, kernel-fusion |
| [#1502](../sources/prs/flashinfer/PR-1502.md) | Add benchmark for cutedsl gemm | 2025-08-18 |  | gemm |
| [#1503](../sources/prs/flashinfer/PR-1503.md) | feat: integrate xqa attention backend | 2025-08-18 |  | attention |
| [#1507](../sources/prs/flashinfer/PR-1507.md) | update allreduce to match trtllm | 2025-08-18 | kernel-fusion | kernel-fusion |
| [#1508](../sources/prs/flashinfer/PR-1508.md) | Support cuda<12.8 built for trtllm_allreduce_fusion. | 2025-08-18 | kernel-fusion | kernel-fusion, moe |
| [#1509](../sources/prs/flashinfer/PR-1509.md) | bugfix: Fix stream handling in cutedsl gemm | 2025-08-18 |  | gemm |
| [#1512](../sources/prs/flashinfer/PR-1512.md) | flashinfer_benchmark QoL Improvements and Attention FP8 Support | 2025-08-18 |  | attention, fp8, gemm |
| [#1500](../sources/prs/flashinfer/PR-1500.md) | fix: Replace cub Max/Min with cuda::maximum/minimum for cuda 13 compatibility | 2025-08-16 |  | gemm |
| [#1495](../sources/prs/flashinfer/PR-1495.md) | fix: update masked moe gemm fp4 tensor reshape | 2025-08-15 |  | fp4, gemm, moe |
| [#1498](../sources/prs/flashinfer/PR-1498.md) | feat: scaling at fp4 gemm epilogue | 2025-08-15 | epilogue-fusion | epilogue-fusion, fp4, gemm |
| [#1483](../sources/prs/flashinfer/PR-1483.md) | perf: add fast path to TopPRenormProbKernel for top_p >= 1.0, significantly boosting SGLang workloads | 2025-08-14 |  | gemm |
| [#1484](../sources/prs/flashinfer/PR-1484.md) | feat: add pdl for trtllm-gen attn | 2025-08-14 |  | attention, decode, flash-attention |
| [#1488](../sources/prs/flashinfer/PR-1488.md) | fix: update cutedsl masked moe gemm | 2025-08-14 |  | gemm, moe |
| [#1490](../sources/prs/flashinfer/PR-1490.md) | feat: Support fp8 qkv, fp16/bf16 out MHA for trtllm-gen. | 2025-08-14 | kernel-fusion | attention, decode, flash-attention |
| [#1491](../sources/prs/flashinfer/PR-1491.md) | Perf: support scale_a/scale_b instead of combined scale in cutlass bmm_fp8 | 2025-08-14 |  | fp8, gemm |
| [#1479](../sources/prs/flashinfer/PR-1479.md) | refactor: unify autotuner for bmm_fp8 | 2025-08-13 |  | fp8, gemm |
| [#1480](../sources/prs/flashinfer/PR-1480.md) | fix missing enable_pdl argument in trtllm-gen fp4 moe | 2025-08-13 | kernel-fusion | fp4, kernel-fusion, moe |
| [#1481](../sources/prs/flashinfer/PR-1481.md) | Add python API for masked grouped gemm | 2025-08-13 |  | fp4, gemm |
| [#1469](../sources/prs/flashinfer/PR-1469.md) | bugfix: Verify num_experts greater or equal to local_experts + offset | 2025-08-12 | kernel-fusion | kernel-fusion, moe |
| [#1472](../sources/prs/flashinfer/PR-1472.md) | feat: Enable multiple fused-moe backends | 2025-08-12 | kernel-fusion | kernel-fusion, moe |
| [#1473](../sources/prs/flashinfer/PR-1473.md) | perf: add 1x4x1 cluster shape for fp8 bmm M<16 cases | 2025-08-12 |  | fp8, gemm |
| [#1475](../sources/prs/flashinfer/PR-1475.md) | tuner: Trtllm-gen Fp4 MoE Autotunner | 2025-08-12 | kernel-fusion | fp4, gemm, kernel-fusion |
| [#1453](../sources/prs/flashinfer/PR-1453.md) | feat: enable trtllm-gen attn speculative decoding verify by decode | 2025-08-11 |  | attention, decode, mla |
| [#1460](../sources/prs/flashinfer/PR-1460.md) | Fix TRTLLM NVFP4-out attention kernel scale factor dim issue | 2025-08-11 |  | attention, decode, fp4 |
| [#1444](../sources/prs/flashinfer/PR-1444.md) | fix: remote redundant zero_init from trtllm-gen attn | 2025-08-10 |  | decode, flash-attention |
| [#1445](../sources/prs/flashinfer/PR-1445.md) | Add alignment in MxFP8Quantization | 2025-08-10 |  | fp8, quantization |
| [#1446](../sources/prs/flashinfer/PR-1446.md) | Remove getEnvEnablePDL in favor of enable_pdl parameter | 2025-08-10 | kernel-fusion | flash-attention, fp4, fp8 |
| [#1415](../sources/prs/flashinfer/PR-1415.md) | benchmark: trtllm-gen mha with sink, add benchmark args | 2025-08-08 |  | attention, decode, flash-attention |
| [#1427](../sources/prs/flashinfer/PR-1427.md) | refactor: Sink attention AoT | 2025-08-08 |  | attention |
| [#1428](../sources/prs/flashinfer/PR-1428.md) | Fix redundant kernels in moe | 2025-08-08 | kernel-fusion | kernel-fusion, moe |
| [#1434](../sources/prs/flashinfer/PR-1434.md) | Fixes for Blackwell Tests | 2025-08-08 |  | gemm |
| [#1435](../sources/prs/flashinfer/PR-1435.md) | bugfix: fix perf issue by using fp8 graph that can use cublaslt | 2025-08-08 |  | fp8, gemm |
| [#1405](../sources/prs/flashinfer/PR-1405.md) | feature: enable cublas for fp4 gemm when cudnn == 9.11.1 or >= 9.13 | 2025-08-07 |  | fp4, gemm |
| [#1410](../sources/prs/flashinfer/PR-1410.md) | [bugfix] Fix compilation failure when compiling csrc/trtllm_moe_allreduce_fusion.cu | 2025-08-07 | kernel-fusion | kernel-fusion, moe |
| [#1412](../sources/prs/flashinfer/PR-1412.md) | Faster weight processing (moe nvfp4) | 2025-08-07 | kernel-fusion | fp4, kernel-fusion, moe |
| [#1396](../sources/prs/flashinfer/PR-1396.md) | gpt-oss: Add MXFP8 x MXFP4 CUTLASS MOE for SM100 and BF16 x MXFP4 CUTLASS for SM90 + SwigluBias Activation | 2025-08-06 | kernel-fusion, warp-specialization | fp4, fp8, gemm |
| [#1397](../sources/prs/flashinfer/PR-1397.md) | feature: add cutlass as bmm_fp8 backend. | 2025-08-06 |  | fp8, gemm |
| [#1398](../sources/prs/flashinfer/PR-1398.md) | Fix trtllm moe launcher local_num_experts | 2025-08-06 | kernel-fusion | kernel-fusion, moe |
| [#1399](../sources/prs/flashinfer/PR-1399.md) | Add Mxfp4 trtllm-gen moe unit tests | 2025-08-06 | kernel-fusion | decode, fp4, kernel-fusion |
| [#1402](../sources/prs/flashinfer/PR-1402.md) | fix shared memory alignment conflict in sampling.cuh | 2025-08-06 |  | gemm |
| [#1384](../sources/prs/flashinfer/PR-1384.md) | Allow BatchPrefillPagedWrapper to call cudnn API | 2025-08-05 |  | attention, prefill |
| [#1389](../sources/prs/flashinfer/PR-1389.md) | GPT-OSS Support: Add Blackwell MoE mxfp4 implementation from TRTLLM and Attention Sink | 2025-08-05 | kernel-fusion | attention, decode, flash-attention |
| [#1390](../sources/prs/flashinfer/PR-1390.md) | Adding FP8 benchmark on attention and matmul testing | 2025-08-05 |  | attention, fp8, gemm |
| [#1376](../sources/prs/flashinfer/PR-1376.md) | bugfix: Add guard for fp4/fp8 related include headers | 2025-08-04 |  | fp4, fp8 |
| [#1378](../sources/prs/flashinfer/PR-1378.md) | refactor: download trtllm gemm metadata from server | 2025-08-04 |  | gemm |
| [#1371](../sources/prs/flashinfer/PR-1371.md) | bugfix: fixed cutlass fused moe usage of FP4QuantizationSFLayout::SWIZZLED | 2025-08-03 | kernel-fusion, swizzling | fp4, kernel-fusion, moe |
| [#1363](../sources/prs/flashinfer/PR-1363.md) | Support scale factor start index for fp4 mha prefill/decode | 2025-08-01 |  | decode, flash-attention, fp4 |
| [#1358](../sources/prs/flashinfer/PR-1358.md) | [fix] remove (view) transpose to keep consistent with majorness MN requirement. | 2025-07-31 |  | fp8, gemm |
| [#1359](../sources/prs/flashinfer/PR-1359.md) | hotfix: update mxfp4 groupwise-scaled gemm unittests | 2025-07-31 |  | fp4, gemm |
| [#1360](../sources/prs/flashinfer/PR-1360.md) | support trtllm-gen prefill fp4 output | 2025-07-31 |  | decode, flash-attention, fp4 |
| [#1361](../sources/prs/flashinfer/PR-1361.md) | Update autotune results for the nvfp4 cutlass moe backends for v0.2.9 | 2025-07-31 | kernel-fusion | fp4, kernel-fusion, moe |
| [#1355](../sources/prs/flashinfer/PR-1355.md) | feature: add fp4 mm using trtllm backend | 2025-07-30 | kernel-fusion | fp4, gemm, kernel-fusion |
| [#1344](../sources/prs/flashinfer/PR-1344.md) | Fix bench deepgemm setting | 2025-07-29 |  | gemm |
| [#1348](../sources/prs/flashinfer/PR-1348.md) | fix: fix trtllm-gen mla error on new interface | 2025-07-29 |  | decode, flash-attention, mla |
| [#1350](../sources/prs/flashinfer/PR-1350.md) | Support passing kv_data_type to MultiLevelCascadeAttentionWrapper.plan() | 2025-07-29 |  | attention |
| [#1339](../sources/prs/flashinfer/PR-1339.md) | feat: Fused rope fp8 quantize kernel for MLA | 2025-07-28 | kernel-fusion | fp8, kernel-fusion, mla |
| [#1324](../sources/prs/flashinfer/PR-1324.md) | feat: Support logits_soft_cap for Persistent attn; fix kv split limit | 2025-07-25 | persistent-kernel | attention, persistent-kernel, prefill |
| [#1328](../sources/prs/flashinfer/PR-1328.md) | refactor: Improved metainfo for trtllm-gen kernels | 2025-07-25 | kernel-fusion | gemm, kernel-fusion, moe |
| [#1331](../sources/prs/flashinfer/PR-1331.md) | feat: masked layout fp4 gemm using cute-dsl | 2025-07-25 |  | fp4, gemm |
| [#1333](../sources/prs/flashinfer/PR-1333.md) | add torch float4_e2m1fn_x2 check for cudnn fp4 backend | 2025-07-25 |  | fp4, gemm |
| [#1334](../sources/prs/flashinfer/PR-1334.md) | [Fix] remove torch 2.8 requirement for FP4 GEMM | 2025-07-25 |  | fp4, gemm |
| [#1314](../sources/prs/flashinfer/PR-1314.md) | test qkvo quantization not equal to 1. | 2025-07-24 |  | decode, quantization |
| [#1316](../sources/prs/flashinfer/PR-1316.md) | minor: add trtllm_gen_mla benchmark | 2025-07-24 |  | mla |
| [#1317](../sources/prs/flashinfer/PR-1317.md) | Allow cudnn prefill kernels to be called natively | 2025-07-24 |  | prefill |
| [#1318](../sources/prs/flashinfer/PR-1318.md) | feat: support output nvfp4 in trtllm-gen function call. | 2025-07-24 |  | decode, flash-attention, fp4 |
| [#1319](../sources/prs/flashinfer/PR-1319.md) | Make Fp8 MoE routing_bias optional | 2025-07-24 | kernel-fusion | fp8, kernel-fusion, moe |
| [#1320](../sources/prs/flashinfer/PR-1320.md) | Add blockwise-scaled FP8 GEMM via TRTLLM-Gen. | 2025-07-24 |  | fp8, gemm, tma |
| [#1321](../sources/prs/flashinfer/PR-1321.md) | Optimizations for TRTLLM MNNVL Allreduce | 2025-07-24 |  | gemm |
| [#1322](../sources/prs/flashinfer/PR-1322.md) | feat: Add k_scale and v_scale to persistent attention  | 2025-07-24 | persistent-kernel | attention, persistent-kernel |
| [#1307](../sources/prs/flashinfer/PR-1307.md) | Fix the bug of the kernel-selection heuristic in trtllm-gen | 2025-07-23 |  | flash-attention |
| [#1309](../sources/prs/flashinfer/PR-1309.md) | Refactor Fused Moe Module | 2025-07-23 | kernel-fusion | fp4, gemm, kernel-fusion |
| [#1310](../sources/prs/flashinfer/PR-1310.md) | Support loading autotuned results from json for cutlass fp4 moe backends | 2025-07-23 | kernel-fusion | fp4, kernel-fusion, moe |
| [#1298](../sources/prs/flashinfer/PR-1298.md) | perfix: use lightweight API to query device property | 2025-07-22 | kernel-fusion | kernel-fusion, moe |
| [#1305](../sources/prs/flashinfer/PR-1305.md) | [Feature] SM level profiler  | 2025-07-22 |  | gemm |
| [#1294](../sources/prs/flashinfer/PR-1294.md) | Update cutlass fp4 moe kernels | 2025-07-21 | kernel-fusion, warp-specialization | fp4, fp8, gemm |
| [#1296](../sources/prs/flashinfer/PR-1296.md) | add cutlass backend for mm_fp4 | 2025-07-21 |  | fp4, gemm |
| [#1297](../sources/prs/flashinfer/PR-1297.md) | feat: Add weight layout option for trtllm-gen fused moe | 2025-07-21 | kernel-fusion | gemm, kernel-fusion, moe |
| [#1290](../sources/prs/flashinfer/PR-1290.md) | [fix] fix integer overflow in FA2 customized_mask & add buffer overflow warning. | 2025-07-19 |  | attention, quantization |
| [#1291](../sources/prs/flashinfer/PR-1291.md) | Remove FAST_BUILD FLAG for MOE | 2025-07-19 | kernel-fusion | gemm, kernel-fusion, moe |
| [#1292](../sources/prs/flashinfer/PR-1292.md) | refactor: Improved metainfo for trtllm-gen fmha | 2025-07-19 |  | decode, flash-attention |
| [#1281](../sources/prs/flashinfer/PR-1281.md) | Unify groupwise fp8 GEMM test | 2025-07-18 |  | fp8, gemm |
| [#1284](../sources/prs/flashinfer/PR-1284.md) | Convert scale_factor from scalar to Tensor in trt_allreduce_fusion | 2025-07-18 | kernel-fusion | kernel-fusion |
| [#1286](../sources/prs/flashinfer/PR-1286.md) | fix multiCtasKvScratchPtr misalignment issue (new one) | 2025-07-18 |  | flash-attention |
| [#1287](../sources/prs/flashinfer/PR-1287.md) | Bug fix: guard fp8 e8m0 and e2m1 compile  | 2025-07-18 | kernel-fusion | fp8, kernel-fusion, moe |
| [#1288](../sources/prs/flashinfer/PR-1288.md) | add mm_fp4 use cudnn backend | 2025-07-18 |  | fp4, gemm, quantization |
| [#1289](../sources/prs/flashinfer/PR-1289.md) | refactor: refactor trtllm-gen attention kernel integration code | 2025-07-18 |  | attention, decode, flash-attention |
| [#1278](../sources/prs/flashinfer/PR-1278.md) | hotfix: fix deepgemm artifactory hash | 2025-07-17 |  | gemm |
| [#1280](../sources/prs/flashinfer/PR-1280.md) | fix: update trtllm-gen fmha benchmark | 2025-07-17 |  | flash-attention |
| [#1266](../sources/prs/flashinfer/PR-1266.md) | feat: add masked deepgemm support and benchmarking | 2025-07-16 |  | fp8, gemm |
| [#1267](../sources/prs/flashinfer/PR-1267.md) | Bug fix: fix duplicate launch in POD | 2025-07-16 |  | attention |
| [#1272](../sources/prs/flashinfer/PR-1272.md) | Add shuffle matrix flag | 2025-07-16 | kernel-fusion | gemm, kernel-fusion, moe |
| [#1255](../sources/prs/flashinfer/PR-1255.md) | TRT-LLM's Multi-Node NVLink AR + fused RMSNorm kernel | 2025-07-15 | kernel-fusion | kernel-fusion |
| [#1258](../sources/prs/flashinfer/PR-1258.md) | feat: enable trtllm-gen mla MTP | 2025-07-15 |  | decode, flash-attention, mla |
| [#1264](../sources/prs/flashinfer/PR-1264.md) | init add gemm fp8 using cudnn backend | 2025-07-15 |  | fp8, gemm |
| [#1265](../sources/prs/flashinfer/PR-1265.md) | Made AR output optional + esthetic changes | 2025-07-15 |  | gemm |
| [#1249](../sources/prs/flashinfer/PR-1249.md) | Remove sm100+ requirment for trtllm allreduce kernels | 2025-07-14 | kernel-fusion | kernel-fusion, moe |
| [#1251](../sources/prs/flashinfer/PR-1251.md) | Reduce the JIT compilation time of gen_gemm_sm100_module | 2025-07-14 |  | fp4, fp8, gemm |
| [#1240](../sources/prs/flashinfer/PR-1240.md) | Patch fp8 cubin availability | 2025-07-11 | kernel-fusion | fp8, gemm, kernel-fusion |
| [#1241](../sources/prs/flashinfer/PR-1241.md) | feat: Support MXFP8 x MXFP4 CUTLASS grouped GEMM | 2025-07-11 |  | fp4, fp8, gemm |
| [#1242](../sources/prs/flashinfer/PR-1242.md) | Add trtllm-gen attention mha kernel with FP8 Q/K/V and FP8 output | 2025-07-11 |  | attention, decode, flash-attention |
| [#1239](../sources/prs/flashinfer/PR-1239.md) | add trtllm-gen context attention | 2025-07-10 |  | attention, decode, flash-attention |
| [#1230](../sources/prs/flashinfer/PR-1230.md) | feat: Add non-causal cudnn prefill kernels | 2025-07-08 |  | prefill |
| [#1234](../sources/prs/flashinfer/PR-1234.md) | bugfix: support uint8_t for vec_t class template | 2025-07-08 |  | gemm |
| [#1221](../sources/prs/flashinfer/PR-1221.md) | Enable cudnn decode and add tests for the cudnn decode kernel | 2025-07-07 |  | decode |
| [#1222](../sources/prs/flashinfer/PR-1222.md) | feat: add trtllm-gen mla cubin | 2025-07-07 |  | attention, decode, flash-attention |
| [#1227](../sources/prs/flashinfer/PR-1227.md) | Fix missing hash in the cudnn cubin path | 2025-07-07 |  | gemm |
| [#1213](../sources/prs/flashinfer/PR-1213.md) | [comm] TRT-LLM's Multi-Node NVLink All-Reduce Kernel | 2025-07-04 |  | gemm |
| [#1214](../sources/prs/flashinfer/PR-1214.md) | Feature/sm100 low latency nvfp4 kernels | 2025-07-04 | kernel-fusion | fp4, gemm, kernel-fusion |
| [#1211](../sources/prs/flashinfer/PR-1211.md) | Fix test_groupwise_scaled_gemm_fp8.py | 2025-07-03 |  | fp8, gemm |
| [#1212](../sources/prs/flashinfer/PR-1212.md) | feat: trtllm-gen fp8 moe kernels | 2025-07-03 | kernel-fusion | fp8, gemm, kernel-fusion |
| [#1208](../sources/prs/flashinfer/PR-1208.md) | Fix the issue with auxillary kernel launch and grid dim calculation | 2025-07-02 |  | prefill |
| [#1209](../sources/prs/flashinfer/PR-1209.md) | Add DeepGEMM kernels | 2025-07-02 |  | fp8, gemm |
| [#1206](../sources/prs/flashinfer/PR-1206.md) | [fix] fix BatchAttention CTA_TILE_KV mask issue | 2025-07-01 |  | attention |
| [#1198](../sources/prs/flashinfer/PR-1198.md) | bugfix: fix blackwell fmha hanging issue for empty kv_len | 2025-06-30 | epilogue-fusion | attention, epilogue-fusion, flash-attention |
| [#1200](../sources/prs/flashinfer/PR-1200.md) | [feat] optimize persistent batch attention perf. | 2025-06-30 | persistent-kernel | attention, persistent-kernel |
| [#1189](../sources/prs/flashinfer/PR-1189.md) | update trtllm-gen decode attention kernel launcher | 2025-06-28 |  | attention, decode, flash-attention |
| [#1181](../sources/prs/flashinfer/PR-1181.md) | bugfix: fix invalid blackwell fmha unittests | 2025-06-26 |  | flash-attention |
| [#1176](../sources/prs/flashinfer/PR-1176.md) | Expose fp4 blockscale swizzling kernel | 2025-06-25 | swizzling | fp4, quantization, swizzling |
| [#1177](../sources/prs/flashinfer/PR-1177.md) | [feat] support block sparse attention w/ variable block sizes and head-wise sparse patterns | 2025-06-25 |  | attention, sparse-attention |
| [#1178](../sources/prs/flashinfer/PR-1178.md) | bugfix: softmax NaN results caused by large -inf masks | 2025-06-25 |  | tma |
| [#1170](../sources/prs/flashinfer/PR-1170.md) | feat: logits processor fustion rule for temperature softmax | 2025-06-24 | kernel-fusion | kernel-fusion, tma |
| [#1164](../sources/prs/flashinfer/PR-1164.md) | feat: enable and update all-reduce fused quantization | 2025-06-22 | kernel-fusion | kernel-fusion, moe, quantization |
| [#1157](../sources/prs/flashinfer/PR-1157.md) | Add fp4 quantization swizzling tests | 2025-06-19 | swizzling | fp4, quantization, swizzling |
| [#1158](../sources/prs/flashinfer/PR-1158.md) | Add more logging to TRTLLM-GEN debug trace (NFC) | 2025-06-19 |  | flash-attention |
| [#1159](../sources/prs/flashinfer/PR-1159.md) | feat: add finalize_moe_allreduce from trtllm | 2025-06-19 | kernel-fusion | kernel-fusion, moe |
| [#1160](../sources/prs/flashinfer/PR-1160.md) | feat: nvshmem python bindings | 2025-06-19 |  | gemm |
| [#1161](../sources/prs/flashinfer/PR-1161.md) | feat: update non-fused moe | 2025-06-19 | kernel-fusion | kernel-fusion, moe |
| [#1153](../sources/prs/flashinfer/PR-1153.md) | feat: Fused temperature online softmax kernel | 2025-06-18 | kernel-fusion | kernel-fusion, tma |
| [#1140](../sources/prs/flashinfer/PR-1140.md) | Fix FA2 and FA3 multi-item scoring and cuda illegal memory access error | 2025-06-12 |  | attention, prefill |
| [#1136](../sources/prs/flashinfer/PR-1136.md) | fix: negative zero by type trait --> binary value | 2025-06-11 |  | gemm |
| [#1137](../sources/prs/flashinfer/PR-1137.md) | [feat] add unified batch attention w/ correctness tests. | 2025-06-11 | persistent-kernel | attention, persistent-kernel, prefill |
| [#1134](../sources/prs/flashinfer/PR-1134.md) | MNNVL MoE All-to-All Support | 2025-06-10 |  | moe |
| [#1131](../sources/prs/flashinfer/PR-1131.md) | feat: add trtllm all-reduce fusion | 2025-06-09 | kernel-fusion | kernel-fusion, moe |
| [#1129](../sources/prs/flashinfer/PR-1129.md) | Fix pointer dtype bug in rope | 2025-06-08 |  | gemm |
| [#1116](../sources/prs/flashinfer/PR-1116.md) | hotfix: fix the blackwell fmha stream | 2025-06-06 |  | attention, flash-attention |
| [#1117](../sources/prs/flashinfer/PR-1117.md) | [Feature] Support PDL for batch Prefill and Decode | 2025-06-06 |  | attention, decode, fp8 |
| [#1114](../sources/prs/flashinfer/PR-1114.md) | bugfix: Fix test and output shape of fp4 quantize | 2025-06-05 | epilogue-fusion, kernel-fusion | epilogue-fusion, fp4, gemm |
| [#1113](../sources/prs/flashinfer/PR-1113.md) | Add CUTLASS fused moe kernels from TensorRT-LLM. | 2025-06-04 | epilogue-fusion, kernel-fusion, pipeline-stages | epilogue-fusion, fp8, gemm |
| [#1108](../sources/prs/flashinfer/PR-1108.md) | feat: add trtllm moe_allreduce_fusion | 2025-06-02 | kernel-fusion | kernel-fusion, moe |
| [#1106](../sources/prs/flashinfer/PR-1106.md) | bugfix: host-precomuted plan function for blackwell fmha | 2025-05-31 | epilogue-fusion, kernel-fusion, tile-scheduling | attention, epilogue-fusion, flash-attention |
| [#1096](../sources/prs/flashinfer/PR-1096.md) | feat: add trtllm all-reduce (non-MoE) | 2025-05-28 |  | moe |
| [#1086](../sources/prs/flashinfer/PR-1086.md) | perf: accelerate blackwell grouped gemm | 2025-05-23 |  | fp8, gemm, grouped-gemm |
| [#1087](../sources/prs/flashinfer/PR-1087.md) | bugfix: fix fp8 attention kernels aot compilation issue | 2025-05-23 |  | attention, fp8, prefill |
| [#1089](../sources/prs/flashinfer/PR-1089.md) | comm: refactor and initialize `flashinfer.comm` module | 2025-05-23 |  | gemm |
| [#1071](../sources/prs/flashinfer/PR-1071.md) | bugfix: adding lse output to blackwell fmha kernels | 2025-05-20 | epilogue-fusion | attention, epilogue-fusion, flash-attention |
| [#1072](../sources/prs/flashinfer/PR-1072.md) | bugfix: follow user-specified sm_scale for blackwell cutlass fmha | 2025-05-20 |  | attention, flash-attention, prefill |
| [#1059](../sources/prs/flashinfer/PR-1059.md) | Parameterize prefix mask call (needed by POD-Attention) | 2025-05-14 |  | attention, prefill |
| [#1054](../sources/prs/flashinfer/PR-1054.md) | Fix KV chunking for POD.  | 2025-05-13 |  | attention |
| [#1055](../sources/prs/flashinfer/PR-1055.md) | bugfix: temporally disable split-kv in blackwell mla | 2025-05-13 |  | attention, mla |
| [#1050](../sources/prs/flashinfer/PR-1050.md) | fix: top_k_mask_logits hangs on -inf inputs | 2025-05-09 |  | gemm |
| [#1051](../sources/prs/flashinfer/PR-1051.md) | [nvidia] Add Blackwell FMHA decode kernel from TRT-LLM | 2025-05-09 |  | attention, decode, flash-attention |
| [#1039](../sources/prs/flashinfer/PR-1039.md) | [nvidia] initial support for blackwell kernels | 2025-04-24 | epilogue-fusion, kernel-fusion, tile-scheduling | attention, epilogue-fusion, flash-attention |
| [#1033](../sources/prs/flashinfer/PR-1033.md) | feat: add functional per-head FP8 quantization for FA3 | 2025-04-23 | epilogue-fusion | attention, decode, epilogue-fusion |
| [#1035](../sources/prs/flashinfer/PR-1035.md) | feat: Softmax free sampling | 2025-04-23 |  | tma |
| [#1029](../sources/prs/flashinfer/PR-1029.md) | fix: add zero init for KV tiled copy | 2025-04-21 |  | attention |
| [#1025](../sources/prs/flashinfer/PR-1025.md) | feat: ragged tensor padding kernel for blackwell kernel alignment | 2025-04-20 |  | gemm |
| [#1015](../sources/prs/flashinfer/PR-1015.md) | add multi-item scoring | 2025-04-11 | epilogue-fusion, tile-scheduling | attention, decode, epilogue-fusion |
| [#1013](../sources/prs/flashinfer/PR-1013.md) | bugfix: import wrapper of mla decode | 2025-04-10 |  | decode, mla |
| [#1014](../sources/prs/flashinfer/PR-1014.md) | misc: fix instrument code for mla profiler | 2025-04-10 |  | attention, mla |
| [#1007](../sources/prs/flashinfer/PR-1007.md) | feat: update decode attention APIs | 2025-04-07 |  | attention, decode, prefill |
| [#997](../sources/prs/flashinfer/PR-997.md) | 3rdparty: upgrade cutlass to 3.9 | 2025-04-03 |  | attention, mla |
| [#994](../sources/prs/flashinfer/PR-994.md) | feat: SM-constraint Communication Kernels | 2025-04-01 |  | gemm |
| [#991](../sources/prs/flashinfer/PR-991.md) | perf: prefetch page indices for mla kernel | 2025-03-31 |  | attention, mla |
| [#982](../sources/prs/flashinfer/PR-982.md) | SM-constraint-GEMM by triton persistent kernel | 2025-03-29 | persistent-kernel | gemm, persistent-kernel |
| [#983](../sources/prs/flashinfer/PR-983.md) | Triton `rms_norm` kernels | 2025-03-29 |  | gemm |
| [#974](../sources/prs/flashinfer/PR-974.md) | perf: dual pivot top-p/top-k renorm | 2025-03-26 |  | gemm |
| [#969](../sources/prs/flashinfer/PR-969.md) | perf: Fix python API overhead when CUDAGraph is not enabled | 2025-03-23 |  | decode, fp8, gemm |
| [#968](../sources/prs/flashinfer/PR-968.md) | perf: reduce torch.library dispatch overhead | 2025-03-22 |  | decode, gemm, mla |
| [#958](../sources/prs/flashinfer/PR-958.md) | [TVM] Added tvm binding for sampling kernel | 2025-03-18 |  | attention |
| [#952](../sources/prs/flashinfer/PR-952.md) | perf: Use 2WG pipeline design for MLA implementation on Hopper | 2025-03-17 | pipeline-stages | attention, mla, pipeline-stages |
| [#945](../sources/prs/flashinfer/PR-945.md) | bugfix: fix potential issues of FA3 template loading nans for PageAttention | 2025-03-14 |  | attention, mla |
| [#930](../sources/prs/flashinfer/PR-930.md) | feat: experimenta support of PDL | 2025-03-11 |  | gemm |
| [#913](../sources/prs/flashinfer/PR-913.md) | feat: flashinfer intra-kernel profiler | 2025-03-05 |  | attention, mla |
| [#901](../sources/prs/flashinfer/PR-901.md) | perf: tweak the pipeline design of mla kernel | 2025-02-27 | pipeline-stages | attention, mla, pipeline-stages |
| [#898](../sources/prs/flashinfer/PR-898.md) | perf: fix MLA split-k performance bug | 2025-02-25 |  | attention, mla |
| [#887](../sources/prs/flashinfer/PR-887.md) | perf: FlashAttention-3 style MLA PageAttention | 2025-02-23 | epilogue-fusion | attention, epilogue-fusion, mla |
| [#888](../sources/prs/flashinfer/PR-888.md) | feat - support mla kvcache store | 2025-02-23 |  | mla |
| [#869](../sources/prs/flashinfer/PR-869.md) | Naive Support for Hopper FP8 Prefill Kernel with Per-Head Quantization | 2025-02-18 | epilogue-fusion | attention, epilogue-fusion, fp8 |
| [#858](../sources/prs/flashinfer/PR-858.md) | Add POD-Attention to FlashInfer | 2025-02-17 |  | attention, decode, prefill |
| [#861](../sources/prs/flashinfer/PR-861.md) | unittest: add MLA test cases where kv_len is evenly divided by page_size. | 2025-02-17 |  | mla |
| [#863](../sources/prs/flashinfer/PR-863.md) | perf: dynamic split-k for MLA | 2025-02-17 |  | attention, mla |
| [#868](../sources/prs/flashinfer/PR-868.md) | bugfix: fix the behavior of MLA kernel when kv-length is 0 | 2025-02-17 |  | attention, mla |
| [#850](../sources/prs/flashinfer/PR-850.md) | misc: Remove duplicate param set in MLA kernel | 2025-02-16 |  | mla |
| [#844](../sources/prs/flashinfer/PR-844.md) | perf: MLA decode kernel implemented by CuTe targeted to SM80 | 2025-02-14 |  | attention, decode, mla |
| [#816](../sources/prs/flashinfer/PR-816.md) | bugfix: fix the behavior of mla plan function when provided with host tensors | 2025-02-13 |  | mla |
| [#821](../sources/prs/flashinfer/PR-821.md) | bugfix: bugfix on sm89 MLA | 2025-02-13 |  | attention, mla |
| [#827](../sources/prs/flashinfer/PR-827.md) | bugfix: fix the signature of `CutlassSegmentGEMMSM90` | 2025-02-13 |  | gemm |
| [#810](../sources/prs/flashinfer/PR-810.md) | bugfix: mla page-attention kernel for different page sizes | 2025-02-12 |  | attention, mla |
| [#812](../sources/prs/flashinfer/PR-812.md) | feat: unlocking MLA for A100 | 2025-02-12 |  | attention, mla |
| [#814](../sources/prs/flashinfer/PR-814.md) | feat: unlock MLA attention for sm89 (L40/L40s/4090) | 2025-02-12 |  | attention, mla |
| [#804](../sources/prs/flashinfer/PR-804.md) | perf: memory efficient deepseek mla fused page-attention kernel | 2025-02-10 | kernel-fusion | attention, kernel-fusion, mla |
| [#801](../sources/prs/flashinfer/PR-801.md) | feat: apply sm_scale at logits instead of q in FA2 template | 2025-02-09 |  | attention, decode, prefill |
| [#799](../sources/prs/flashinfer/PR-799.md) | feat: support f32 attention output in FA2 template | 2025-02-08 |  | attention, prefill |
| [#793](../sources/prs/flashinfer/PR-793.md) | fix rope logic in mla decoding | 2025-02-07 |  | attention, decode, mla |
| [#787](../sources/prs/flashinfer/PR-787.md) | bugfix: MLA decode should multiply sm_scale by math::log2e | 2025-02-05 |  | attention, decode, mla |
| [#781](../sources/prs/flashinfer/PR-781.md) | bugfix: fix batch prefill attention kernel unittests | 2025-02-04 |  | attention, prefill |
| [#785](../sources/prs/flashinfer/PR-785.md) | bugfix: drop CTA_TILE_Q=32 | 2025-02-04 |  | prefill |
| [#776](../sources/prs/flashinfer/PR-776.md) | perf: refactor fa2 prefill template | 2025-02-03 |  | attention, prefill |
| [#778](../sources/prs/flashinfer/PR-778.md) | feat: Separate QK/VO head dim dispatch for sm90 AOT | 2025-02-03 |  | prefill |
| [#774](../sources/prs/flashinfer/PR-774.md) | bugfix: Ensure Loop Termination by Enforcing IEEE-754 Compliance in Sampling Kernels | 2025-02-01 |  | gemm |
| [#765](../sources/prs/flashinfer/PR-765.md) | feat: support deepseek prefill attention shape | 2025-01-30 | epilogue-fusion | attention, decode, epilogue-fusion |
| [#754](../sources/prs/flashinfer/PR-754.md) | Change `apply_rope_with_cos_sin_cache` to accept `cos_sin_cache` | 2025-01-27 |  | gemm |
| [#728](../sources/prs/flashinfer/PR-728.md) | Align KV chunk size binary search with actual KV chunk splitting. | 2025-01-09 |  | attention |
| [#718](../sources/prs/flashinfer/PR-718.md) | bugfix: FusedAddRMSNorm kernels might require more than 48KB shared memory when d is large. | 2025-01-06 | kernel-fusion | kernel-fusion |
| [#714](../sources/prs/flashinfer/PR-714.md) | perf: fix the iteration bound of SWA in FA2 prefill template | 2025-01-03 |  | attention, prefill |

<a id="pytorchpytorch"></a>
## pytorch/pytorch
85 PRs

| PR | Title | Date | Techniques | Tags |
|-----|-------|------|------------|------|
| [#180470](../sources/prs/pytorch/PR-180470.md) | [release 2.12] Apply Release only changes to 2.12 branch | 2026-04-15 |  | attention, flash-attention, quantization |
| [#178009](../sources/prs/pytorch/PR-178009.md) | [MPS] fix compiling of SDPA producing nan results | 2026-03-20 |  | gemm |
| [#177144](../sources/prs/pytorch/PR-177144.md) | [Inductor] Don't unfuse addmm for bf16/fp16 to avoid precision loss | 2026-03-11 |  | gemm |
| [#177193](../sources/prs/pytorch/PR-177193.md) | [Inductor][MPS] Fix half-precision type mismatches in Metal shader codegen (#176436) | 2026-03-11 |  | gemm |
| [#176783](../sources/prs/pytorch/PR-176783.md) | [inductor] Fix Identity comparability and evalf recursion | 2026-03-07 |  | gemm |
| [#176410](../sources/prs/pytorch/PR-176410.md) | [Inductor] Reject non-contiguous subnode fusion in mix-order reduction. | 2026-03-04 | kernel-fusion | kernel-fusion |
| [#176495](../sources/prs/pytorch/PR-176495.md) | [inductor] avoid multi-stage for mix-order-red by default (#176228) | 2026-03-04 |  | gemm |
| [#175826](../sources/prs/pytorch/PR-175826.md) | [CI] Update inductor CI jobs to CUDA 13.0 | 2026-02-26 |  | python |
| [#175567](../sources/prs/pytorch/PR-175567.md) | [release-only] Remove +ptx from cuda 13.0 builds | 2026-02-23 |  | gemm |
| [#175580](../sources/prs/pytorch/PR-175580.md) | [MPS] Fix 2-pass SDPA memory corruption by forcing float accumulators | 2026-02-23 |  | attention |
| [#175299](../sources/prs/pytorch/PR-175299.md) | [benchmark] Skip pytorch_CycleGAN_and_pix2pix from inductor benchmarks | 2026-02-19 |  | gemm |
| [#175091](../sources/prs/pytorch/PR-175091.md) | [RELEASE 2.11] Release only changes | 2026-02-16 |  | attention, flash-attention, quantization |
| [#175096](../sources/prs/pytorch/PR-175096.md) | Update inductor expected accuracy files | 2026-02-16 |  | gemm |
| [#172577](../sources/prs/pytorch/PR-172577.md) | [Graph Partition] Improve support for mutation ops | 2026-01-15 |  | gemm |
| [#172141](../sources/prs/pytorch/PR-172141.md) | Skip modded_nanogpt model in TorchInductor benchmark | 2026-01-09 |  | gemm |
| [#171895](../sources/prs/pytorch/PR-171895.md) | [cherry-pick][cuDNN][SDPA] cuDNN SDPA off-by-default for cuDNN versions < 12.9 (#171627) | 2026-01-07 |  | gemm |
| [#171247](../sources/prs/pytorch/PR-171247.md) | [xpu][fix][inductor] fallback bfloat16 atomics to eager | 2025-12-24 |  | gemm |
| [#171150](../sources/prs/pytorch/PR-171150.md) | Avoid closing random file handles in Inductor | 2025-12-23 |  | gemm |
| [#171189](../sources/prs/pytorch/PR-171189.md) | [cherry-pick][CUDA] Upgrade cuDNN to 9.15.1 for CUDA 13 builds  | 2025-12-23 |  | gemm |
| [#171129](../sources/prs/pytorch/PR-171129.md) | [Inductor] Fix constants handling for Triton constexpr (triton#8248) | 2025-12-22 |  | gemm |
| [#171140](../sources/prs/pytorch/PR-171140.md) | [ROCm] Make grouped GEMM CK opt‑in via env and default to fallback path | 2025-12-22 |  | gemm |
| [#170884](../sources/prs/pytorch/PR-170884.md) | [inductor] Fix cudagraph skip for index_put_ with boolean indices, gr… | 2025-12-19 |  | gemm |
| [#170486](../sources/prs/pytorch/PR-170486.md) | [flex_attention] adds support for low precision K/V inputs in compiled mode with GPU | 2025-12-16 |  | attention, flash-attention |
| [#170555](../sources/prs/pytorch/PR-170555.md) | [cherry-pick] Fix vllm issue for flex (#170499) | 2025-12-16 |  | attention |
| [#170190](../sources/prs/pytorch/PR-170190.md) | [ROCm] Enable shared memory based pruning for Triton configs | 2025-12-11 |  | gemm |
| [#170246](../sources/prs/pytorch/PR-170246.md) | [Inductor] ExternKernelBenchmarkRequest best attempt | 2025-12-11 |  | gemm |
| [#170112](../sources/prs/pytorch/PR-170112.md) | [RELEASE 2.10] Release only changes | 2025-12-10 |  | quantization |
| [#167327](../sources/prs/pytorch/PR-167327.md) | [cuDNN][SDPA][Convolution] Expose cuDNN runtime version in CUDA hooks | 2025-11-07 |  | gemm |
| [#167121](../sources/prs/pytorch/PR-167121.md) | [cuDNN][SDPA] Check-in test for #166211 | 2025-11-05 |  | gemm |
| [#166910](../sources/prs/pytorch/PR-166910.md) | [inductor] don't try to reorder loops for template | 2025-11-04 |  | gemm |
| [#166913](../sources/prs/pytorch/PR-166913.md) | [Dynamo] Don't guard data ptrs by default with mark_static_address | 2025-11-04 |  | gemm |
| [#166922](../sources/prs/pytorch/PR-166922.md) | [Inductor] No longer throw error in bmm out_dtype lowering due to tem… | 2025-11-04 |  | gemm |
| [#166967](../sources/prs/pytorch/PR-166967.md) | [Graph Partition] move custom rules to inductor config (#166458) | 2025-11-04 |  | gemm |
| [#166985](../sources/prs/pytorch/PR-166985.md) | [Graph Partition] fix graph partition input signature for fallback kernels | 2025-11-04 |  | gemm |
| [#166994](../sources/prs/pytorch/PR-166994.md) | [GraphPartition] cache get_free_symbol_uses (#166338) | 2025-11-04 |  | gemm |
| [#167020](../sources/prs/pytorch/PR-167020.md) | [Minor][Inductor] move some combo kernel log from warning to debug | 2025-11-04 |  | gemm |
| [#164893](../sources/prs/pytorch/PR-164893.md) | CUDA 13.0 builds fix on Amazon Linux 2023 | 2025-10-07 |  | gemm |
| [#164364](../sources/prs/pytorch/PR-164364.md) | [SDPA] [MPS] Fixes regression in 2.8.0 for scaled_dot_product_attention using mps | 2025-10-01 |  | attention |
| [#164368](../sources/prs/pytorch/PR-164368.md) | [Flex attention] Fix flex attention head broadcast | 2025-10-01 |  | attention |
| [#164236](../sources/prs/pytorch/PR-164236.md) | [AARCH64][CD][CUDA13][Triton][PTXAS] Turn on BUILD_BUNDLE_PTXAS=1   | 2025-09-30 |  | gemm |
| [#164026](../sources/prs/pytorch/PR-164026.md) | [cuDNN][SDPA] Disable dropout for cuDNN SDPA on 9.11 - 9.13 | 2025-09-27 |  | gemm |
| [#163954](../sources/prs/pytorch/PR-163954.md) | Move inductor jobs 3.9->3.10 | 2025-09-26 |  | gemm |
| [#163861](../sources/prs/pytorch/PR-163861.md) | fix pickling for BitwiseFn | 2025-09-25 |  | gemm |
| [#163764](../sources/prs/pytorch/PR-163764.md) | [Cherry-Pick] [CD] CUDA 13 specific followup changes. Remove sm50-70 From CUDA 12.6 and CUDA 12.8 builds (#162455) | 2025-09-24 |  | gemm |
| [#163766](../sources/prs/pytorch/PR-163766.md) | [CD] CUDA 13.0 fix preload logic to include nvidia/cu13/lib/ | 2025-09-24 |  | gemm |
| [#163633](../sources/prs/pytorch/PR-163633.md) | CUDA 13.0 Warning update for supported architectures | 2025-09-23 |  | gemm |
| [#163583](../sources/prs/pytorch/PR-163583.md) | [2.9 cherry pick][triton] update 3.5 pin to bbb06c0334a6772b92d24bde54956e675c8c6604 (#163382) | 2025-09-22 |  | gemm |
| [#163585](../sources/prs/pytorch/PR-163585.md) | CUDA 13.0 Warning update for supported architectures | 2025-09-22 |  | python |
| [#163388](../sources/prs/pytorch/PR-163388.md) | [Inductor][Intel GPU] Save `threads_per_warp` from tirton compiled kernel for launching kernel correctly in cpp wrapper. | 2025-09-20 |  | gemm |
| [#163395](../sources/prs/pytorch/PR-163395.md) | [graph partition] Add way to register custom rule (#163310) | 2025-09-20 |  | gemm |
| [#163380](../sources/prs/pytorch/PR-163380.md) | [Graph Partition] improve custom op output alias | 2025-09-19 |  | gemm |
| [#163265](../sources/prs/pytorch/PR-163265.md) | [Release 2.9] [cuDNN][SDPA][submodule] Roll-back cuDNN frontend upgrade, update Met… | 2025-09-18 |  | gemm |
| [#163097](../sources/prs/pytorch/PR-163097.md) | [Cherry Pick][Graph Partition] allow sharing default device context | 2025-09-16 |  | gemm |
| [#162764](../sources/prs/pytorch/PR-162764.md) | fix cpp extension distributed warning spew | 2025-09-11 |  | python, cuda-cpp |
| [#162455](../sources/prs/pytorch/PR-162455.md) | [CD] CUDA 13 specific followup changes. Remove sm50-70 From CUDA 12.6 and CUDA 12.8 builds | 2025-09-09 |  | python |
| [#162501](../sources/prs/pytorch/PR-162501.md) | CUDA 13.0 Windows Nvidia Driver Update to 580.88 | 2025-09-09 |  | gemm |
| [#158646](../sources/prs/pytorch/PR-158646.md) | [cherry-pick][inductor][triton] Update HAS_WARP_SPEC to check triton.Config params. Update Triton Hash to top of release/3.4.x stack | 2025-07-18 |  | gemm |
| [#158301](../sources/prs/pytorch/PR-158301.md) | Add warning about removed sm50 and sm60 arches | 2025-07-15 |  | python |
| [#158237](../sources/prs/pytorch/PR-158237.md) | [MPS] Switch Cholesky  decomp to column wise | 2025-07-14 |  | gemm |
| [#157752](../sources/prs/pytorch/PR-157752.md) | [release] Triton pin update to 3.4 | 2025-07-08 |  | gemm |
| [#157422](../sources/prs/pytorch/PR-157422.md) | [PowerPC] Fixed build issue for vsx vec256 complexfloat and scaled_mm_out_cpu  | 2025-07-02 |  | gemm |
| [#157241](../sources/prs/pytorch/PR-157241.md) | [user triton] AOT inductor support for device-side TMA | 2025-06-29 |  | tma |
| [#156932](../sources/prs/pytorch/PR-156932.md) | Fix macOS build with `USE_MPS=OFF` | 2025-06-26 |  | gemm |
| [#154121](../sources/prs/pytorch/PR-154121.md) | Fix uint view copy (#151598) | 2025-05-22 |  | gemm |
| [#153641](../sources/prs/pytorch/PR-153641.md) | [FlexAttention] explicilty create grad_q w/ strides | 2025-05-15 |  | attention |
| [#153304](../sources/prs/pytorch/PR-153304.md) | Mark auto_functionalized HOPs as cacheable (#151194) | 2025-05-10 |  | gemm |
| [#153104](../sources/prs/pytorch/PR-153104.md) | [FlexAttention] Remove Old Constraint on lastdim strides | 2025-05-07 |  | attention |
| [#152967](../sources/prs/pytorch/PR-152967.md) | [ATen][CUDA] Optimize 128 bit vectorization | 2025-05-06 |  | gemm |
| [#152774](../sources/prs/pytorch/PR-152774.md) | [dynamo][super variable] Fix bug to use correct source | 2025-05-04 |  | gemm |
| [#150676](../sources/prs/pytorch/PR-150676.md) | [CUDA][avgpool2d] Fix backward launch bounds again for `sm100`, `sm120` | 2025-04-04 |  | gemm |
| [#150705](../sources/prs/pytorch/PR-150705.md) | [CUDA] Only use vec128 if CUDA version is newer than 12.8 | 2025-04-04 | vectorized-loads | cuda-cpp |
| [#150640](../sources/prs/pytorch/PR-150640.md) | [CUDA][avgpool2d] Fix backward launch bounds again for sm100, sm120 | 2025-04-03 |  | cuda-cpp |
| [#150447](../sources/prs/pytorch/PR-150447.md) | [inductor] Fix inductor windows linker error | 2025-04-01 |  | gemm |
| [#150448](../sources/prs/pytorch/PR-150448.md) | [Windows][inductor] fix blank space break windows file path | 2025-04-01 |  | gemm |
| [#150145](../sources/prs/pytorch/PR-150145.md) | Dont exclude constant_pad_nd in prologue fusion | 2025-03-27 | kernel-fusion | kernel-fusion |
| [#149993](../sources/prs/pytorch/PR-149993.md) | [inductor][triton 3.3] Fix cpp_wrapper w/ TMA in triton 3.3 | 2025-03-26 |  | tma |
| [#149871](../sources/prs/pytorch/PR-149871.md) | Add release branch push triggers to inductor-rocm-mi300.yml | 2025-03-24 |  | gemm |
| [#149644](../sources/prs/pytorch/PR-149644.md) | op should NOT be static in aoti_torch_call_dispatcher | 2025-03-20 |  | gemm |
| [#149386](../sources/prs/pytorch/PR-149386.md) | Add AOTI shim for _weight_int4pack_mm_cpu_tensor (#149031) | 2025-03-18 |  | quantization |
| [#149125](../sources/prs/pytorch/PR-149125.md) | Remove runtime dependency on packaging | 2025-03-13 |  | gemm |
| [#149059](../sources/prs/pytorch/PR-149059.md) | [inductor] Fix profiler tests with latest Triton | 2025-03-12 |  | gemm |
| [#144398](../sources/prs/pytorch/PR-144398.md) | ROCm SDPA: Ensure attn_mask has the same dtype with q | 2025-01-08 |  | gemm |
| [#144335](../sources/prs/pytorch/PR-144335.md) | Fix PythonMod printing | 2025-01-07 |  | gemm |
| [#144248](../sources/prs/pytorch/PR-144248.md) | [inductor][cpu] Fix bmm b_index for dynamic expressions in inductor autotuner | 2025-01-06 |  | gemm |
| [#144209](../sources/prs/pytorch/PR-144209.md) | Update torch-xpu-ops commit pin | 2025-01-05 |  | gemm |

<a id="sgl-projectsglang"></a>
## sgl-project/sglang
1277 PRs

| PR | Title | Date | Techniques | Tags |
|-----|-------|------|------------|------|
| [#32126](../sources/prs/sglang/PR-32126.md) | Seed the GDN CuteDSL correctness test inputs to fix flakiness | 2026-07-23 |  | attention, prefill |
| [#32128](../sources/prs/sglang/PR-32128.md) | [Kernel] Reclassify kernel tests by ops group + move helpers out of the package (RFC #29630) | 2026-07-23 | kernel-fusion | attention, decode, flash-attention |
| [#32130](../sources/prs/sglang/PR-32130.md) | [NPU] [FIX] Fix performance degradation of Qwen3.5-397B-A17B | 2026-07-23 |  | attention |
| [#32134](../sources/prs/sglang/PR-32134.md) | [AMD] Fix GPT-OSS-120b break on gfx1250 | 2026-07-23 | kernel-fusion | fp4, kernel-fusion, moe |
| [#32148](../sources/prs/sglang/PR-32148.md) | [Kernel] Classification cleanup: unify _jit_ naming, drop empty/model groups, add elementwise (RFC #29630) | 2026-07-23 | kernel-fusion | attention, fp4, fp8 |
| [#32015](../sources/prs/sglang/PR-32015.md) | [Kernel] Phase 4 batch-2: migrate JIT operator groups into kernels.ops (no shims) (RFC #29630) | 2026-07-22 | kernel-fusion | attention, decode, fp4 |
| [#32040](../sources/prs/sglang/PR-32040.md) | [NPU] ascend fuseep use moe ep group | 2026-07-22 |  | moe |
| [#32043](../sources/prs/sglang/PR-32043.md) | [AMD] GFX1250 ROCm bringup: infra, build, kernels and models (DSV4, DSR1, GPTOSS) | 2026-07-22 | kernel-fusion | attention, decode, fp4 |
| [#32045](../sources/prs/sglang/PR-32045.md) | [Kernel] Phase 4 batch-3: migrate tangled JIT subsystems + new groups into kernels.ops (RFC #29630) | 2026-07-22 | kernel-fusion, pipeline-stages, tile-scheduling | 2sm-cooperative, attention, flash-attention |
| [#32072](../sources/prs/sglang/PR-32072.md) | [Kernel] RFC #29630 finale: retire sglang.jit_kernel into sglang.kernels | 2026-07-22 | kernel-fusion | attention, decode, fp8 |
| [#32076](../sources/prs/sglang/PR-32076.md) | Fix Inkling kernel imports after migration | 2026-07-22 |  | gemm |
| [#32109](../sources/prs/sglang/PR-32109.md) | [Perf] Skip blocks past per-request live length in full-width Triton kernels | 2026-07-22 |  | attention, decode |
| [#32113](../sources/prs/sglang/PR-32113.md) | [Bugfix] [NPU] Fix w4a8 MoE performance degradation | 2026-07-22 |  | moe, quantization |
| [#31867](../sources/prs/sglang/PR-31867.md) | [NPU] Update non-vit vision part for cumulative seqlen | 2026-07-21 |  | attention |
| [#31889](../sources/prs/sglang/PR-31889.md) | [AMD] Cache AITER expert mask across decode | 2026-07-21 |  | decode, moe |
| [#31897](../sources/prs/sglang/PR-31897.md) | [CPU] refactor rope kernels | 2026-07-21 |  | gemm |
| [#31904](../sources/prs/sglang/PR-31904.md) | [KDA] Fix mixed exponent bases in Triton chunk prefill | 2026-07-21 |  | attention, prefill |
| [#31961](../sources/prs/sglang/PR-31961.md) | Change the FP8 per-tensor GEMM backend on SM120 to cuBLAS | 2026-07-21 |  | fp8, gemm, quantization |
| [#31981](../sources/prs/sglang/PR-31981.md) | [Perf] Skip page-table columns past kv length in DSA draft-extend metadata kernel | 2026-07-21 |  | attention |
| [#31986](../sources/prs/sglang/PR-31986.md) | [Perf] Stack dspark dense draft per-layer ctx KV projection into one GEMM | 2026-07-21 |  | gemm |
| [#31748](../sources/prs/sglang/PR-31748.md) | Lower AutoRound quantization MMLU threshold | 2026-07-20 |  | quantization |
| [#31754](../sources/prs/sglang/PR-31754.md) | Fix unnecessary gather/scatter on CPU for non-contiguous Mamba statepool | 2026-07-20 |  | attention |
| [#31762](../sources/prs/sglang/PR-31762.md) | fix(marlin_nvfp4): only apply routed_scaling_factor in moe_sum_reduce | 2026-07-20 | kernel-fusion | fp4, kernel-fusion, moe |
| [#31769](../sources/prs/sglang/PR-31769.md) | [Bugfix] Fix Cohere2MoeConfig import crash from huggingface_hub @strict | 2026-07-20 |  | moe |
| [#31772](../sources/prs/sglang/PR-31772.md) | [NPU] Fix LLaDA2 MoE OOM after the FRACTAL_NZ cast, re-enabling the NZ speedup | 2026-07-20 |  | moe |
| [#31782](../sources/prs/sglang/PR-31782.md) | [bugfix][NPU] Fix startup bug in olmoe 1b 7b | 2026-07-20 |  | moe |
| [#31796](../sources/prs/sglang/PR-31796.md) | [NPU]fix rl update weights 'Parameter' object has no attribute 'weight_LOADER' | 2026-07-20 |  | moe, quantization |
| [#31812](../sources/prs/sglang/PR-31812.md) | config: route runtime config adjustments through the namespace bags | 2026-07-20 | kernel-fusion | attention, kernel-fusion, moe |
| [#31814](../sources/prs/sglang/PR-31814.md) | config: read resolved config via namespace accessors | 2026-07-20 | kernel-fusion | attention, decode, fp4 |
| [#31816](../sources/prs/sglang/PR-31816.md) | config: read parallel config leaves via get_parallel() | 2026-07-20 |  | attention, moe |
| [#31825](../sources/prs/sglang/PR-31825.md) |   [Quant] Support NVFP4_AWQ checkpoints in ModelOpt FP4 path | 2026-07-20 |  | fp4, nvfp4, quantization |
| [#31838](../sources/prs/sglang/PR-31838.md) | Fix pad-row top-k masking with custom_routing_function under DP attention | 2026-07-20 |  | attention, moe, prefill |
| [#31688](../sources/prs/sglang/PR-31688.md) | Fix ROCm fused KV and KDA paths | 2026-07-19 | kernel-fusion | attention, kernel-fusion |
| [#31705](../sources/prs/sglang/PR-31705.md) | [DeepSeek-V4] Fix idle-rank dummy-extend sparse-prefill crash under DP breakable CUDA graph | 2026-07-19 |  | attention, prefill |
| [#31707](../sources/prs/sglang/PR-31707.md) | [NPU] bugfix for W4A8MoE bias 3D dimension mismatch problem | 2026-07-19 |  | moe, quantization |
| [#31732](../sources/prs/sglang/PR-31732.md) | Support GPT-OSS zigzag CP with TRTLLM-MHA | 2026-07-19 |  | attention, fp4 |
| [#31634](../sources/prs/sglang/PR-31634.md) | [MUSA] Fix sglang-kernel build | 2026-07-18 |  | gemm |
| [#31636](../sources/prs/sglang/PR-31636.md) | [NPU] fix: skip Triton embedding kernel on NPU to avoid kernel launch failure | 2026-07-18 |  | gemm |
| [#31649](../sources/prs/sglang/PR-31649.md) | Enable GPT-OSS TinyGEMM on CUDA 13 | 2026-07-18 |  | gemm |
| [#31653](../sources/prs/sglang/PR-31653.md) | Fix SM120 NVFP4 KV cache test OOM | 2026-07-18 |  | fp4, nvfp4 |
| [#31666](../sources/prs/sglang/PR-31666.md) | [Kernel] Phase 3+4: move JIT infra + operator groups into sglang.kernels (RFC #29630) | 2026-07-18 | kernel-fusion | attention, decode, flash-attention |
| [#31667](../sources/prs/sglang/PR-31667.md) | Make Q contiguous before TRT-LLM MHA decode | 2026-07-18 |  | attention, decode, fp4 |
| [#31669](../sources/prs/sglang/PR-31669.md) | Sm120 scatter fallback | 2026-07-18 | kernel-fusion | kernel-fusion, moe |
| [#31672](../sources/prs/sglang/PR-31672.md) | fix(gemma4): prevent attention mask offset overflow | 2026-07-18 |  | attention, gemm |
| [#31675](../sources/prs/sglang/PR-31675.md) | [AMD] Fix DeepSeek MLA prefill shape mismatch on HIP eager fallback (missing mha_companion_layers) | 2026-07-18 |  | mla, prefill |
| [#31681](../sources/prs/sglang/PR-31681.md) | Add Inkling model support | 2026-07-18 | kernel-fusion, pipeline-stages, tile-scheduling | 2sm-cooperative, attention, decode |
| [#31682](../sources/prs/sglang/PR-31682.md) | Turn on breakable prefill cuda graph for dp attention by default | 2026-07-18 |  | attention, decode, prefill |
| [#31514](../sources/prs/sglang/PR-31514.md) | [DCP] Enable decode context parallel for Kimi K2.5 NVFP4 | 2026-07-17 |  | decode, fp4, nvfp4 |
| [#31515](../sources/prs/sglang/PR-31515.md) | [AMD] Fix stale imports in test_fused_fp8_kv_write.py | 2026-07-17 | kernel-fusion | attention, fp8, kernel-fusion |
| [#31527](../sources/prs/sglang/PR-31527.md) | [Spec] Allocate the verify tree-mask scratch on the target backend only | 2026-07-17 |  | attention, mla |
| [#31546](../sources/prs/sglang/PR-31546.md) | [Kernel] Simplify sglang.kernels tests to idiomatic pytest style (RFC #29630) | 2026-07-17 | kernel-fusion | kernel-fusion |
| [#31558](../sources/prs/sglang/PR-31558.md) | [Performance] Avoid FLA L2-norm recompilation by token count | 2026-07-17 |  | attention |
| [#31582](../sources/prs/sglang/PR-31582.md) | [Kernel] Sweep decoupled scattered kernels into sglang.kernels.ops (RFC #29630) | 2026-07-17 | kernel-fusion | attention, kernel-fusion, moe |
| [#31608](../sources/prs/sglang/PR-31608.md) | [LoRA] Guard TMA down path for LoRA hooks | 2026-07-17 | kernel-fusion | kernel-fusion, moe, tma |
| [#31610](../sources/prs/sglang/PR-31610.md) | docs(cookbook): replace pinned nightly/dev images with :latest | 2026-07-17 |  | gemm |
| [#31614](../sources/prs/sglang/PR-31614.md) | [spec decoding] fix multi_layer_eagle rotate_input_ids kernel registration | 2026-07-17 |  | gemm |
| [#31615](../sources/prs/sglang/PR-31615.md) | [AMD] batch 3: register newly-added JIT kernel benchmarks for jit-kernel-benchmark-test-amd | 2026-07-17 |  | gemm |
| [#31619](../sources/prs/sglang/PR-31619.md) | [CP] Migrate MLA prefill CP (DeepSeek V3) to CP-v2 zigzag strategy | 2026-07-17 |  | attention, mla, prefill |
| [#31391](../sources/prs/sglang/PR-31391.md) | Enable Kimi multimodal breakable prefill CUDA graph replay | 2026-07-16 |  | attention, prefill |
| [#31400](../sources/prs/sglang/PR-31400.md) | [JIT] Reduce MoE fused gate CI test sweep | 2026-07-16 | kernel-fusion | kernel-fusion, moe |
| [#31439](../sources/prs/sglang/PR-31439.md) | refactor: wrap split backends once on full-attention backends | 2026-07-16 |  | attention |
| [#31449](../sources/prs/sglang/PR-31449.md) | [Bugfix][NPU] Fix/Refactor routed scaling factor application in MoE routing | 2026-07-16 |  | moe |
| [#31456](../sources/prs/sglang/PR-31456.md) | [NPU] fix modelslim quant tensor name | 2026-07-16 |  | quantization |
| [#31468](../sources/prs/sglang/PR-31468.md) | [Spec] DFlash: remove per-step host syncs so the CPU runs a full step ahead (spec-v2 overlap) | 2026-07-16 |  | attention, mla |
| [#31474](../sources/prs/sglang/PR-31474.md) | Fix KDA prefix caching under mamba extra_buffer and enable it for kimi_linear | 2026-07-16 |  | attention, mla |
| [#31492](../sources/prs/sglang/PR-31492.md) | [AMD] register 8 JIT kernel benchmarks to jit-kernel-benchmark-test-amd | 2026-07-16 | kernel-fusion | fp8, gemm, kernel-fusion |
| [#31498](../sources/prs/sglang/PR-31498.md) | [Fix] Enable chunked input-logprob processing by default to cap peak memory | 2026-07-16 |  | moe |
| [#31501](../sources/prs/sglang/PR-31501.md) | [flashinfer] Pass window_left at plan time for the SWA paged prefill wrapper | 2026-07-16 |  | attention, prefill |
| [#31244](../sources/prs/sglang/PR-31244.md) | [Spec] Converge DP-attention spec width scaling onto `num_tokens_per_req` | 2026-07-15 |  | attention, decode |
| [#31245](../sources/prs/sglang/PR-31245.md) | [Spec] Key DP-attention graph admission on raw sync request counts | 2026-07-15 |  | attention, decode |
| [#31250](../sources/prs/sglang/PR-31250.md) | [XPU][GDN] add XPU path for causal_conv1d_fn and causal_conv1d_update | 2026-07-15 |  | attention |
| [#31251](../sources/prs/sglang/PR-31251.md) | [sglang-miles] lora GLM-5.2 FP8 based model support | 2026-07-15 |  | fp8 |
| [#31254](../sources/prs/sglang/PR-31254.md) | [Spec] Inline DP-attention width scaling; drop the coefficient indirection | 2026-07-15 |  | attention |
| [#31292](../sources/prs/sglang/PR-31292.md) | [Kernel] Decouple KernelBackend from device + device-based CapabilityRequirement (RFC #29630) | 2026-07-15 | kernel-fusion | gemm, kernel-fusion, moe |
| [#31307](../sources/prs/sglang/PR-31307.md) | [Kernel] Fill non-CUDA coverage: HIP (aiter/rocm-triton) + Ascend NPU backends (RFC #29630) | 2026-07-15 | kernel-fusion | kernel-fusion |
| [#31311](../sources/prs/sglang/PR-31311.md) | Fix LongCat-2.0 real EP (deepep): double all-reduce + ScMoE RoPE crash | 2026-07-15 |  | moe |
| [#31334](../sources/prs/sglang/PR-31334.md) | [CPU] Fix mxfp4 padding size | 2026-07-15 |  | fp4, quantization |
| [#31343](../sources/prs/sglang/PR-31343.md) | Fix MiMo-V2 on Blackwell: FA3 fallback and TP-aware audio weight loading | 2026-07-15 |  | gemm |
| [#31364](../sources/prs/sglang/PR-31364.md) | fa3: sync-free eagle spec via fixed-window draft-extend metadata | 2026-07-15 |  | attention, decode |
| [#31367](../sources/prs/sglang/PR-31367.md) | [Bugfix] Stamp capture-time num_tokens_per_req in multi-layer EAGLE; close jit_kernel CI filter gaps | 2026-07-15 | kernel-fusion | attention, decode, kernel-fusion |
| [#31379](../sources/prs/sglang/PR-31379.md) | [AMD] Register 2 CPU/triton unit + kernel tests for AMD 1-GPU PR CI | 2026-07-15 |  | moe |
| [#31088](../sources/prs/sglang/PR-31088.md) | [AMD] Register 3 CPU-bound / triton unit + light-integration tests for AMD 1-GPU PR CI | 2026-07-14 |  | attention, gated-delta-net |
| [#31089](../sources/prs/sglang/PR-31089.md) | [Kernel] Hotfix: update sgl-kernel imports of relocated fp8_kernel (RFC #29630 #30784) | 2026-07-14 |  | fp8, gemm |
| [#31090](../sources/prs/sglang/PR-31090.md) | flashmla: sync-free spec via device-side draft-extend | 2026-07-14 |  | attention, mla |
| [#31106](../sources/prs/sglang/PR-31106.md) | [Cherry-pick to release/v0.5.15] support GLM-5.2 MTP index sharing with prefill CP (#30992) | 2026-07-14 |  | attention, prefill |
| [#31109](../sources/prs/sglang/PR-31109.md) | Remove QServe and FBGEMM FP8 quantization | 2026-07-14 |  | fp8, gemm, quantization |
| [#31110](../sources/prs/sglang/PR-31110.md) | [CPU] bypass scoring_func argument in topk for cpu device | 2026-07-14 |  | moe |
| [#31125](../sources/prs/sglang/PR-31125.md) | Disable flaky DSV4-Flash FP4 BCG determinism test (nondeterminism from #30898 idle-rank dummy extend) | 2026-07-14 |  | fp4 |
| [#31126](../sources/prs/sglang/PR-31126.md) | [Intel XPU] Enable (biased) grouped topk for xpu | 2026-07-14 |  | moe |
| [#31131](../sources/prs/sglang/PR-31131.md) | [AMD] Fix DSV4 JIT build on rocm  | 2026-07-14 |  | gemm |
| [#31143](../sources/prs/sglang/PR-31143.md) | [AMD] jit_kernel: complete utils.cuh HIP-compat (cudaDevAttr / cudaDeviceGetAttribute) | 2026-07-14 |  | gemm |
| [#31159](../sources/prs/sglang/PR-31159.md) | Extract MoE/EP setup into a moe_ep_setup module | 2026-07-14 |  | moe |
| [#31161](../sources/prs/sglang/PR-31161.md) | Introduce ModelRunner.ps ParallelState | 2026-07-14 |  | attention, decode, mla |
| [#31162](../sources/prs/sglang/PR-31162.md) | Introduce KVCacheConfigurator and migrate KV-cache config logic | 2026-07-14 |  | moe |
| [#31165](../sources/prs/sglang/PR-31165.md) | Drop ModelRunner's duplicated parallel-degree fields and read them via self.ps | 2026-07-14 |  | attention, gemm |
| [#31167](../sources/prs/sglang/PR-31167.md) | Extract attention-backend setup into a module | 2026-07-14 |  | attention |
| [#31171](../sources/prs/sglang/PR-31171.md) | [CPU] add fused input proj for qwen3.5 | 2026-07-14 | kernel-fusion | kernel-fusion |
| [#31174](../sources/prs/sglang/PR-31174.md) | fix: Qwen3.5-35B-A3B-AWQ w2_weight KeyError and related params for CPU | 2026-07-14 |  | quantization |
| [#31202](../sources/prs/sglang/PR-31202.md) | Delete sgl-kernel AOT `bmm_fp8`, use `flashinfer.bmm_fp8` | 2026-07-14 | kernel-fusion | attention, fp8, gemm |
| [#31227](../sources/prs/sglang/PR-31227.md) | perf: shard Kimi DP image feature transport | 2026-07-14 |  | attention |
| [#31231](../sources/prs/sglang/PR-31231.md) | Fix gate stride for 4D decode layouts | 2026-07-14 | kernel-fusion | attention, decode, kernel-fusion |
| [#30975](../sources/prs/sglang/PR-30975.md) | Fix --moe-a2a-backend silently ignored for LongCat-2.0 (moe_topk missing from gate) | 2026-07-13 |  | moe |
| [#30976](../sources/prs/sglang/PR-30976.md) | fix: load the right mtp lm head quantization | 2026-07-13 |  | quantization |
| [#30998](../sources/prs/sglang/PR-30998.md) | [Spec] Remove dead `padded_static_len` and stale `SGLANG_ENABLE_SPEC_V2` references | 2026-07-13 |  | attention, fp4 |
| [#31001](../sources/prs/sglang/PR-31001.md) | Fix GLM/DeepSeek NVFP4 + flashinfer_trtllm long-context "!!!!" collapse (NaN routing) | 2026-07-13 |  | fp4, nvfp4 |
| [#31005](../sources/prs/sglang/PR-31005.md) | [NPU] Fix DSA top-k seed buffer shape for MTP IndexShare | 2026-07-13 |  | attention |
| [#31038](../sources/prs/sglang/PR-31038.md) | [XPU] Route topk_sigmoid and topk_softmax to AOT sgl-kernel-xpu symbols | 2026-07-13 |  | moe, tma |
| [#31049](../sources/prs/sglang/PR-31049.md) | [Kernel] Rewrite JIT custom all-reduce (v2) with a decoupled kernel/storage design | 2026-07-13 |  | gemm |
| [#31050](../sources/prs/sglang/PR-31050.md) | [FullCG] Preserve attention LSE through the custom-op boundary | 2026-07-13 |  | attention |
| [#31056](../sources/prs/sglang/PR-31056.md) | Fix MockDSV4ModelRunner missing spec_algorithm | 2026-07-13 |  | attention |
| [#31083](../sources/prs/sglang/PR-31083.md) | [Cherry-pick to release/v0.5.15] Stabilize GLM-5.2 MTP IndexShare across PD and CUDA graph replay (#30839) | 2026-07-13 |  | attention, decode, prefill |
| [#30924](../sources/prs/sglang/PR-30924.md) | [JIT] Trait-driven per_token_group_quant: unify the quant kernel family (flat + masked) | 2026-07-12 |  | fp8, gemm, moe |
| [#30940](../sources/prs/sglang/PR-30940.md) | [AMD] Gate TP4 o_proj/qkv CK block-FP8 GEMM shapes to Triton (ROCm 7.0 Qwen-3.5) | 2026-07-12 |  | fp8, gemm, quantization |
| [#30947](../sources/prs/sglang/PR-30947.md) | [1/3] [EAGLE] perf: Fuse topk=1 draft postprocess | 2026-07-12 |  | gemm |
| [#30948](../sources/prs/sglang/PR-30948.md) | [2/3] [EAGLE] perf: Fuse TP vocab-parallel embedding | 2026-07-12 |  | gemm |
| [#30838](../sources/prs/sglang/PR-30838.md) | [JIT] Refactor dtype traits into DTypeTrait and unify warp reductions | 2026-07-11 | kernel-fusion | kernel-fusion, mla, prefill |
| [#30839](../sources/prs/sglang/PR-30839.md) | [bug-fix] Stabilize GLM-5.2 MTP IndexShare across PD and CUDA graph replay | 2026-07-11 |  | attention, decode, prefill |
| [#30846](../sources/prs/sglang/PR-30846.md) | [Fix] Forward on_after_cuda_graph_warmup through HybridLinearAttnBackend | 2026-07-11 |  | attention |
| [#30847](../sources/prs/sglang/PR-30847.md) | [Fix] Guard kernel OOB accesses and harden runtime edge cases | 2026-07-11 |  | gemm |
| [#30853](../sources/prs/sglang/PR-30853.md) | [Spec] Enable draft extend cuda graph for DeepSeek-V4 attention backend | 2026-07-11 |  | attention |
| [#30866](../sources/prs/sglang/PR-30866.md) | Tune VLM MoE paths | 2026-07-11 | kernel-fusion | kernel-fusion, moe |
| [#30868](../sources/prs/sglang/PR-30868.md) | fix: fix vlm cuda graph shape stability | 2026-07-11 |  | attention, prefill |
| [#30878](../sources/prs/sglang/PR-30878.md) | perf: reuse MoonViT FA3 max-seqlen metadata | 2026-07-11 |  | attention |
| [#30896](../sources/prs/sglang/PR-30896.md) | [Fix] Unify ForwardBatch extend lens cpu fields to their declared list type | 2026-07-11 |  | attention, mla, prefill |
| [#30898](../sources/prs/sglang/PR-30898.md) | Enable breakable prefill CUDA graph for DP attention | 2026-07-11 |  | attention, fp4, prefill |
| [#30721](../sources/prs/sglang/PR-30721.md) | [NPU] use standalone group for moe ep | 2026-07-10 | kernel-fusion | kernel-fusion, moe |
| [#30724](../sources/prs/sglang/PR-30724.md) | [Cherry-pick to release/v0.5.15] Fix gfx95 bpreshuffle FP8 activation scale layout (#29275) | 2026-07-10 |  | attention, fp8, mla |
| [#30754](../sources/prs/sglang/PR-30754.md) | Seed sgl-kernel topk sigmoid tests on all backends | 2026-07-10 |  | moe |
| [#30784](../sources/prs/sglang/PR-30784.md) | [Kernel] Migrate scattered quantization kernels to sglang.kernels (RFC #29630, Phase 2.5, 1/7) | 2026-07-10 | kernel-fusion | fp8, gemm, kernel-fusion |
| [#30786](../sources/prs/sglang/PR-30786.md) | [Kernel] Migrate scattered MoE kernels to sglang.kernels (RFC #29630, Phase 2.5, 2/7) | 2026-07-10 | kernel-fusion | fp8, gemm, kernel-fusion |
| [#30787](../sources/prs/sglang/PR-30787.md) | [Kernel] Migrate top-level srt/layers stray kernels to sglang.kernels (RFC #29630, Phase 2.5, 3/7) | 2026-07-10 | kernel-fusion | attention, fp4, gemm |
| [#30789](../sources/prs/sglang/PR-30789.md) | [Kernel] Migrate generic attention kernels to sglang.kernels (RFC #29630, Phase 2.5, 4/7) | 2026-07-10 | kernel-fusion | attention, decode, kernel-fusion |
| [#30790](../sources/prs/sglang/PR-30790.md) | Fix hybrid attention graph hook test fixture | 2026-07-10 |  | attention |
| [#30792](../sources/prs/sglang/PR-30792.md) | [Kernel] Migrate DSA + DSV4 attention kernels to sglang.kernels (RFC #29630, Phase 2.5, 5/7) | 2026-07-10 | kernel-fusion | attention, decode, fp4 |
| [#30793](../sources/prs/sglang/PR-30793.md) | [Kernel] Migrate linear-attention, MiniMax-sparse and diffusion kernels to sglang.kernels (RFC #29630, Phase 2.5, 6/7) | 2026-07-10 | kernel-fusion | attention, decode, kernel-fusion |
| [#30795](../sources/prs/sglang/PR-30795.md) | [Kernel] Relocate vendored fla and mamba kernel trees to sglang.kernels (RFC #29630, Phase 2.5, 7/7) | 2026-07-10 | kernel-fusion | attention, decode, gated-delta-net |
| [#30802](../sources/prs/sglang/PR-30802.md) | [refactor] Move MLP collective flags onto ForwardFlags | 2026-07-10 |  | attention, moe |
| [#30826](../sources/prs/sglang/PR-30826.md) | Update GLM-5.2 NVFP4 cookbook | 2026-07-10 |  | fp4, nvfp4 |
| [#30828](../sources/prs/sglang/PR-30828.md) | Make the mxfp8 MoE runner backend list extensible | 2026-07-10 |  | fp8, moe |
| [#30580](../sources/prs/sglang/PR-30580.md) | Lazy load TileLang MHC kernels | 2026-07-09 |  | gemm |
| [#30586](../sources/prs/sglang/PR-30586.md) | Move breakable CUDA graph back into model_executor/runner_backend_utils | 2026-07-09 |  | attention |
| [#30589](../sources/prs/sglang/PR-30589.md) | [NPU][bugfix] Fix NPU KernelLaunch Failure in rotate_input_ids_triton with Empty Batch | 2026-07-09 |  | gemm |
| [#30616](../sources/prs/sglang/PR-30616.md) | [mem_cache][7/N] refactor: move  MLATokenToKVPoolHost to pool_host.mla | 2026-07-09 |  | decode, mla |
| [#30622](../sources/prs/sglang/PR-30622.md) | [AMD] Remove ROCm page_first+kernel -> layer_first HiCache fallback (follow-up to #28534) | 2026-07-09 |  | gemm |
| [#30627](../sources/prs/sglang/PR-30627.md) | Fix CuTe DSL DSA paged MQA export | 2026-07-09 |  | attention |
| [#30628](../sources/prs/sglang/PR-30628.md) | [NPU][bugfix] Fix NPU KernelLaunch Failure in rotate_input_ids_triton with Empty Batch | 2026-07-09 |  | gemm |
| [#30641](../sources/prs/sglang/PR-30641.md) | Allow EPLB manual test to use FlashInfer A2A | 2026-07-09 |  | gemm |
| [#30645](../sources/prs/sglang/PR-30645.md) | [DSA] Fix top-k v2 emitting invalid indices under tie overflow / inf scores (IMA in FA3 sparse decode) | 2026-07-09 |  | decode |
| [#30646](../sources/prs/sglang/PR-30646.md) | Improve EPLB dispatch handling and diagnostics | 2026-07-09 |  | moe |
| [#30688](../sources/prs/sglang/PR-30688.md) | Fix MoE functionality on RDNA (gfx1100/gfx1201) via Python-level fallback dispatch | 2026-07-09 |  | moe |
| [#30695](../sources/prs/sglang/PR-30695.md) | [Refactor] Make DeepSeek-V4 attention backend tolerate an absent CPU seq_lens mirror | 2026-07-09 |  | attention |
| [#30703](../sources/prs/sglang/PR-30703.md) | [misc] Remove unit test cases that fail the admission criteria (round 2) | 2026-07-09 |  | gemm |
| [#30706](../sources/prs/sglang/PR-30706.md) | feat(moriep): add fp4 combine dtype (SGLANG_MORI_COMBINE_DTYPE=fp4) | 2026-07-09 |  | fp4, moe |
| [#30708](../sources/prs/sglang/PR-30708.md) | [style] Extract init-static values in forward path | 2026-07-09 |  | attention, decode |
| [#30711](../sources/prs/sglang/PR-30711.md) | [Refactor] Split DeepSeek-V4 MQALayer into a reusable attention base | 2026-07-09 |  | attention |
| [#30443](../sources/prs/sglang/PR-30443.md) | [NVIDIA] Allow modelopt_mixed quantization with flashinfer_cutedsl MoE runner | 2026-07-08 |  | moe, quantization |
| [#30449](../sources/prs/sglang/PR-30449.md) | [Cherry-pick to release/v0.5.15] [DSV4] perf: Make FP8 quant output tensor contiguous (#27926) | 2026-07-08 |  | fp8 |
| [#30450](../sources/prs/sglang/PR-30450.md) | Fix FlashInfer A2A IMA by DP-synchronizing the decode graph bucket (#30242) | 2026-07-08 |  | decode, moe |
| [#30457](../sources/prs/sglang/PR-30457.md) | Support scheduler_recv_interval (recv skipper) under DP-attention | 2026-07-08 |  | attention |
| [#30460](../sources/prs/sglang/PR-30460.md) | [DeepSeek V2] Reorder dual-stream MoE to main-first to avoid CUDA graph stream explosion | 2026-07-08 |  | moe |
| [#30490](../sources/prs/sglang/PR-30490.md) | [refactor] Add the per-forward flags tier: ctx.forward | 2026-07-08 |  | attention, moe |
| [#30491](../sources/prs/sglang/PR-30491.md) | [refactor] Split the DP gathered-buffer state between flags.dp and ctx.forward | 2026-07-08 |  | attention |
| [#30492](../sources/prs/sglang/PR-30492.md) | [refactor] Adopt get_parallel() everywhere and close out the parallel wrapper surface | 2026-07-08 |  | attention, decode, gemm |
| [#30493](../sources/prs/sglang/PR-30493.md) | [refactor] Retire the legacy config accessor and the remaining process singletons | 2026-07-08 | kernel-fusion | attention, fp4, fp8 |
| [#30514](../sources/prs/sglang/PR-30514.md) | [DSA] Integrate Q8KV8 FP8 Sparse MLA Prefill into the DSA Backend (DeepSeek-V3.2) | 2026-07-08 |  | attention, fp8, mla |
| [#30535](../sources/prs/sglang/PR-30535.md) | [hicache]: add  mamba_io_kernel | 2026-07-08 |  | gemm |
| [#30540](../sources/prs/sglang/PR-30540.md) | [Attention Backend] Add HPC-Ops attention backend | 2026-07-08 |  | attention |
| [#30567](../sources/prs/sglang/PR-30567.md) | Support CuteDSL GEMM BF16 on SM100 on by default when allowed by heuristic | 2026-07-08 |  | gemm, quantization |
| [#30346](../sources/prs/sglang/PR-30346.md) | [refactor] Read resolved config from server_args fields; retire the flags mirror tier | 2026-07-07 | kernel-fusion | kernel-fusion, moe |
| [#30347](../sources/prs/sglang/PR-30347.md) | [refactor] Collect MoE and DP-attention runtime state into typed flag groups | 2026-07-07 | kernel-fusion | attention, kernel-fusion, moe |
| [#30348](../sources/prs/sglang/PR-30348.md) | [refactor] ctx.resources: named slots, stream leases, and workspace buffer leases | 2026-07-07 |  | attention, mla, moe |
| [#30355](../sources/prs/sglang/PR-30355.md) | [AMD] [Fix] Fix --attention-backend triton work for DeepSeek MLA on MI355 (null-K + decode dispatch + RoPE) | 2026-07-07 |  | attention, decode, mla |
| [#30356](../sources/prs/sglang/PR-30356.md) | Disable FA3 sparse mask kernels by default | 2026-07-07 |  | gemm |
| [#30365](../sources/prs/sglang/PR-30365.md) | [DSV4] Remove per-step seqlen D2H from speculative to make overlap scheduler work | 2026-07-07 |  | attention |
| [#30378](../sources/prs/sglang/PR-30378.md) | [DSA] Re-enable fused top-k v2 for MTP: clamp padded-row seq_lens to >= 0 | 2026-07-07 | kernel-fusion | attention, kernel-fusion |
| [#30387](../sources/prs/sglang/PR-30387.md) | Fix zero expert routed ids for MoE backends | 2026-07-07 |  | moe |
| [#30397](../sources/prs/sglang/PR-30397.md) | [Cherry pick to release/v0.5.15] Fix NVFP4 online quantization | 2026-07-07 |  | fp4, nvfp4, quantization |
| [#30415](../sources/prs/sglang/PR-30415.md) | Enable RDNA3/4 (gfx1100/gfx1201) for ROCm kernels | 2026-07-07 |  | fp4, fp8, moe |
| [#30424](../sources/prs/sglang/PR-30424.md) | [Cherry-pick to release/v0.5.15] [Cherry pick to release/v0.5.15] Fix NVFP4 online quantization (#30397) | 2026-07-07 |  | fp4, nvfp4, quantization |
| [#30426](../sources/prs/sglang/PR-30426.md) | [Tiny] Fix Import Error for Pure TP config with flashinfer_mxfp4 | 2026-07-07 |  | fp4, moe |
| [#30438](../sources/prs/sglang/PR-30438.md) | Delete CUTLASS FP8 blockwise for SM90 and SM100, move SM120 to JIT and add SwapAB | 2026-07-07 |  | fp8, gemm, quantization |
| [#30207](../sources/prs/sglang/PR-30207.md) | [AMD] Register 2 hardware-agnostic 1-GPU PR tests for AMD CI | 2026-07-06 |  | moe |
| [#30212](../sources/prs/sglang/PR-30212.md) | [AMD] Register 3 ROCm-portable JIT kernel tests for AMD CI | 2026-07-06 | kernel-fusion | kernel-fusion |
| [#30216](../sources/prs/sglang/PR-30216.md) | [CPU] add fused_qk_gemma_norm and refactor norm kernel implementation | 2026-07-06 | kernel-fusion | gemm, kernel-fusion |
| [#30238](../sources/prs/sglang/PR-30238.md) | [AMD] Support two batch overlap with MTP on DeepSeekV4 | 2026-07-06 |  | fp4 |
| [#30247](../sources/prs/sglang/PR-30247.md) | Optimize LongCat-Flash router GEMM with the HPC-Ops bf16xfp32 kernel | 2026-07-06 |  | gemm |
| [#30255](../sources/prs/sglang/PR-30255.md) | Fix DSV4 prefill large Triton recompilation idle across context lengths | 2026-07-06 |  | attention, prefill |
| [#30261](../sources/prs/sglang/PR-30261.md) | [Spec] Add DSpark: confidence-scheduled speculative decoding | 2026-07-06 |  | attention, decode, prefill |
| [#30265](../sources/prs/sglang/PR-30265.md) | [AMD] Fix GLM-5.2 MTP Quark excludes | 2026-07-06 |  | moe |
| [#30272](../sources/prs/sglang/PR-30272.md) | Implement SM120 DeepSeek V4 flashinfer_mxfp4 moe runner backend + TP2 | 2026-07-06 | kernel-fusion | attention, fp4, fp8 |
| [#30274](../sources/prs/sglang/PR-30274.md) | [DSA] Fold page-table into fused top-k v2 (decode): drop page_size=1 expansion | 2026-07-06 | kernel-fusion | attention, decode, kernel-fusion |
| [#30275](../sources/prs/sglang/PR-30275.md) | [Model] Support LongCat 2.0 FP8 | 2026-07-06 |  | attention, decode, fp8 |
| [#30278](../sources/prs/sglang/PR-30278.md) | Fix LTX2 RoPE JIT kernel CI | 2026-07-06 | kernel-fusion | kernel-fusion |
| [#30280](../sources/prs/sglang/PR-30280.md) | Delete sgl-kernel AOT router GEMM and fused A GEMM | 2026-07-06 | kernel-fusion | gemm, kernel-fusion |
| [#30290](../sources/prs/sglang/PR-30290.md) | [AMD] Register 5 CI-verified 1-GPU kernel/attention unit tests for AMD PR CI | 2026-07-06 |  | attention |
| [#30299](../sources/prs/sglang/PR-30299.md) | [refactor] Move model-capability adjustments into the resolution pipeline | 2026-07-06 | pipeline-stages | attention, gemm, pipeline-stages |
| [#30302](../sources/prs/sglang/PR-30302.md) | [AMD] [MORI-EP] Skip LocalExpertCount kernel in decode graph when not recording | 2026-07-06 |  | decode, moe |
| [#30306](../sources/prs/sglang/PR-30306.md) | ci: run jit-kernel tests on scheduled full runs | 2026-07-06 |  | gemm |
| [#30307](../sources/prs/sglang/PR-30307.md) | [AMD] add dedicated jit-kernel-benchmark-test-amd stage + register portable JIT benches | 2026-07-06 | kernel-fusion | decode, fp8, gemm |
| [#30164](../sources/prs/sglang/PR-30164.md) | [1/N] elastic-ep: Add runtime EP scale-up | 2026-07-05 | kernel-fusion | attention, kernel-fusion, moe |
| [#30169](../sources/prs/sglang/PR-30169.md) | [GDN/KDA] Fuse SM100 CuteDSL prefill state I/O into the chunk h kernel | 2026-07-05 |  | attention, prefill |
| [#30079](../sources/prs/sglang/PR-30079.md) | [MoE] Fix moe_fused_gate out-of-range expert id on all-NaN rows (fixes eagle_dp_attention crash) | 2026-07-04 | kernel-fusion | attention, kernel-fusion, moe |
| [#30097](../sources/prs/sglang/PR-30097.md) | [MLX] Size the attention KV pool at the compute dtype for quantized models | 2026-07-04 |  | attention, quantization |
| [#30107](../sources/prs/sglang/PR-30107.md) | [diffusion] perf: add unified SP shard helpers and zero-copy tail-pad attention | 2026-07-04 | kernel-fusion, pipeline-stages | attention, kernel-fusion, pipeline-stages |
| [#30113](../sources/prs/sglang/PR-30113.md) | [KDA] Add FlashInfer SM100 KDA decode + MTP (target_verify) backend | 2026-07-04 |  | attention, decode |
| [#30117](../sources/prs/sglang/PR-30117.md) | Support Cutedsl BF16 GEMM JIT kernel | 2026-07-04 |  | gemm, quantization |
| [#30125](../sources/prs/sglang/PR-30125.md) | [MLX] Fix FakeOverlapScheduler test stub broken by forward_ct accounting | 2026-07-04 |  | attention |
| [#30137](../sources/prs/sglang/PR-30137.md) | [refactor] Config resolution pipeline: full-stack review (10-PR series, review only) | 2026-07-04 | kernel-fusion, pipeline-stages | attention, decode, kernel-fusion |
| [#30140](../sources/prs/sglang/PR-30140.md) | [DeepSeek-V4] Enable non-paged indexer by default for large prefill chunks | 2026-07-04 |  | attention, prefill |
| [#29988](../sources/prs/sglang/PR-29988.md) | [dsv4] Trigger MHC prenorm prewarm at weight-load time with rank sync | 2026-07-03 |  | gemm |
| [#29997](../sources/prs/sglang/PR-29997.md) | [MoE] Retire the AOT moe_fused_gate / kimi_k2_moe_fused_gate gate kernels (#26771) | 2026-07-03 | kernel-fusion | kernel-fusion, moe |
| [#29999](../sources/prs/sglang/PR-29999.md) | [NPU] bugfix for Base class add mamba_track_indices parameter | 2026-07-03 | kernel-fusion | attention, kernel-fusion, moe |
| [#30025](../sources/prs/sglang/PR-30025.md) | fix: reorder DSA indexer dual-stream ops to avoid CUDA graph stream explosion | 2026-07-03 |  | attention |
| [#30040](../sources/prs/sglang/PR-30040.md) | [diffusion] feat: add LingBot realtime prompt, KV window, and lazy VAE controls | 2026-07-03 | kernel-fusion, pipeline-stages | attention, kernel-fusion, pipeline-stages |
| [#30044](../sources/prs/sglang/PR-30044.md) | [Kernel] Introduce sglang.kernels namespace and migrate scattered triton_ops kernels (RFC #29630, Phase 2) | 2026-07-03 | kernel-fusion | attention, decode, fp8 |
| [#30047](../sources/prs/sglang/PR-30047.md) | Bugfix qwen prefix cache circumstances | 2026-07-03 |  | attention |
| [#30073](../sources/prs/sglang/PR-30073.md) | [refactor] Migrate the attention_backend resolution chain (stack 11/15) | 2026-07-03 |  | attention |
| [#30075](../sources/prs/sglang/PR-30075.md) | [refactor] Migrate the moe_runner_backend / quantization resolution chains (stack 13/15) | 2026-07-03 |  | moe, quantization |
| [#29885](../sources/prs/sglang/PR-29885.md) | [DeepSeek V4] Cover both dense and sparse prefill paths in the compress attention unittest | 2026-07-02 |  | attention, prefill |
| [#29903](../sources/prs/sglang/PR-29903.md) | [diffusion] fix: fix z-Image online fp8 quantization crash with dit_cpu_offload | 2026-07-02 | kernel-fusion, pipeline-stages | fp8, kernel-fusion, pipeline-stages |
| [#29905](../sources/prs/sglang/PR-29905.md) | docs: add Qwen3.6-27B-NVFP4 variant to cookbook | 2026-07-02 |  | fp4, nvfp4 |
| [#29909](../sources/prs/sglang/PR-29909.md) | [Bugfix][NPU] Fix Hunyuan3 model where MoE's routing_scaling_ratio is missing on NPU | 2026-07-02 |  | moe |
| [#29911](../sources/prs/sglang/PR-29911.md) | [XPU] Remove redundant xpu graph backend and make xpu graph opt-in by default | 2026-07-02 |  | attention, gemm |
| [#29918](../sources/prs/sglang/PR-29918.md) | [AMD] Gate broken CK block-FP8 GEMM shapes to aiter-triton-GEMM to fix ROCm 7.0 Qwen3.5 accuracy | 2026-07-02 |  | fp8, gemm, quantization |
| [#29921](../sources/prs/sglang/PR-29921.md) | perf(triton): avoid per-step D2H .item() sync in cuda-graph loc translate | 2026-07-02 |  | attention |
| [#29929](../sources/prs/sglang/PR-29929.md) | Fix FlashInfer A2A top-k ID dtype | 2026-07-02 |  | moe |
| [#29945](../sources/prs/sglang/PR-29945.md) | Move deferred mamba cow and clear | 2026-07-02 |  | attention |
| [#29972](../sources/prs/sglang/PR-29972.md) | Support MiMo V2.5 with zigzag context parallelism | 2026-07-02 |  | attention, gemm |
| [#29973](../sources/prs/sglang/PR-29973.md) | [FIX] Prevent Lightning Attention extra-buffer mamba state corruption | 2026-07-02 |  | attention |
| [#29982](../sources/prs/sglang/PR-29982.md) | [AMD][DeepSeek V4] Fix default FlashMLA sparse prefill off on ROCm/HIP | 2026-07-02 |  | mla, prefill |
| [#29793](../sources/prs/sglang/PR-29793.md) | [NPU]Modify --lora-backend & --moe-runner-backend description. | 2026-07-01 |  | moe |
| [#29798](../sources/prs/sglang/PR-29798.md) | fix: avoid DSA indexer CPU seq lens fallback | 2026-07-01 |  | attention |
| [#29822](../sources/prs/sglang/PR-29822.md) | [AMD] Accept ROCm tensors in JIT kernel TensorMatcher + register 4 kernel tests | 2026-07-01 | kernel-fusion | kernel-fusion |
| [#29843](../sources/prs/sglang/PR-29843.md) | [trtllm_mha] Fuse cuda-graph metadata rebuild into one triton kernel | 2026-07-01 |  | attention |
| [#29852](../sources/prs/sglang/PR-29852.md) | [diffusion] refactor: refactor cuda attention backend resolver | 2026-07-01 | kernel-fusion | attention, kernel-fusion |
| [#29867](../sources/prs/sglang/PR-29867.md) | feat(short-conv): shared ShortConvAttnBackend for ZAYA1 CCA + LFM2 short conv | 2026-07-01 |  | attention, moe |
| [#29874](../sources/prs/sglang/PR-29874.md) | [LoRA] DSA indexer targets + MoE-LoRA cuda-graph and RL adapter-reload fixes | 2026-07-01 |  | attention, moe |
| [#29693](../sources/prs/sglang/PR-29693.md) | [AMD] Register ltx2_ada_values JIT kernel test for AMD nightly CI | 2026-06-30 | kernel-fusion | kernel-fusion |
| [#29694](../sources/prs/sglang/PR-29694.md) | [AMD] Fix int8 per-token quant Triton portability + register test for AMD nightly CI | 2026-06-30 |  | quantization |
| [#29699](../sources/prs/sglang/PR-29699.md) | When attention TP for linear and full attention, use Flashinfer allreduce fusion | 2026-06-30 | kernel-fusion | attention, kernel-fusion |
| [#29708](../sources/prs/sglang/PR-29708.md) | [KDA-Pilot] Add LTX2 QKNorm split-RoPE CUDA fast path | 2026-06-30 | kernel-fusion | kernel-fusion |
| [#29734](../sources/prs/sglang/PR-29734.md) | [GDN] Auto-select FlashInfer GDN prefill on validated SM100 configs | 2026-06-30 |  | attention, prefill |
| [#29761](../sources/prs/sglang/PR-29761.md) | [Bugfix] compressed-tensors WNA16 MoE: don't assume a "Linear" config group | 2026-06-30 |  | moe, quantization |
| [#29771](../sources/prs/sglang/PR-29771.md) | [MoE] Consolidate ungrouped + grouped gate/topk onto one Triton router (#26771) — faster than AOT on B200/H100/H200, at parity with flashinfer | 2026-06-30 | kernel-fusion | kernel-fusion, moe |
| [#29775](../sources/prs/sglang/PR-29775.md) | [DeepSeek V4] Enable FlashMLA sparse prefill by default | 2026-06-30 |  | attention, mla, prefill |
| [#29778](../sources/prs/sglang/PR-29778.md) | [Feature] Add DWDP (Distributed Weight Data Parallelism) for MoE prefill | 2026-06-30 | kernel-fusion | kernel-fusion, moe, prefill |
| [#29782](../sources/prs/sglang/PR-29782.md) | [AMD] Register 3 unit mem_cache + utils tests for stage-b-test-1-gpu-small-amd | 2026-06-30 |  | gemm |
| [#29589](../sources/prs/sglang/PR-29589.md) | fa3/fa4: sync-free for all backends and phases | 2026-06-29 |  | attention |
| [#29595](../sources/prs/sglang/PR-29595.md) | [Spec] Enable FlashInfer autotune for spec draft | 2026-06-29 |  | attention, decode, mla |
| [#29619](../sources/prs/sglang/PR-29619.md) | [DeepSeek-V4] Add an opt-in non-paged indexer for long-context prefill | 2026-06-29 |  | attention, prefill |
| [#29636](../sources/prs/sglang/PR-29636.md) | [Kernel] Strengthen kernel shape coverage | 2026-06-29 | kernel-fusion | fp8, gemm, kernel-fusion |
| [#29640](../sources/prs/sglang/PR-29640.md) | Add Qwen3 MoE tests for PP compatibility with CP and DP | 2026-06-29 |  | moe |
| [#29659](../sources/prs/sglang/PR-29659.md) | [LFM2-MoE] Support Transformers v5 packed MoE expert weights | 2026-06-29 |  | moe |
| [#29664](../sources/prs/sglang/PR-29664.md) | [Diffusion] Reuse shared AlignedVector and tidy jit_kernel/diffusion | 2026-06-29 | kernel-fusion | kernel-fusion |
| [#29669](../sources/prs/sglang/PR-29669.md) | Skip MXFP8 autotune on dense GEMM, which causes IMA | 2026-06-29 |  | fp8, gemm |
| [#29671](../sources/prs/sglang/PR-29671.md) | [AMD] Register fused_metadata_copy JIT kernel test for AMD nightly CI | 2026-06-29 | kernel-fusion | kernel-fusion |
| [#29674](../sources/prs/sglang/PR-29674.md) | docs: add B200 NVFP4 recipes + benchmarks to GLM-5.2 cookbook | 2026-06-29 |  | fp4, nvfp4 |
| [#29678](../sources/prs/sglang/PR-29678.md) | feat(mem_cache): unified memory pool for hybrid Mamba / SWA models | 2026-06-29 |  | attention, decode |
| [#29685](../sources/prs/sglang/PR-29685.md) | [sglang-miles] Cherry-pick #26980: Skip routed expert capture for draft model under spec v2 | 2026-06-29 | kernel-fusion | kernel-fusion, moe |
| [#29690](../sources/prs/sglang/PR-29690.md) | Fuse the preprocess kernels of trtllm-gen attention | 2026-06-29 | kernel-fusion | attention, fp8, kernel-fusion |
| [#29533](../sources/prs/sglang/PR-29533.md) | feat(mem_cache): page-major (layer-major within a page) KV/state layout | 2026-06-28 |  | attention, decode |
| [#29543](../sources/prs/sglang/PR-29543.md) | Fix DP-attention SHM feature finalization race | 2026-06-28 |  | attention |
| [#29557](../sources/prs/sglang/PR-29557.md) | [cookbook] GLM-5.2 NVFP4 B300: TP8 recipe + 3 strategies | 2026-06-28 |  | fp4, nvfp4 |
| [#29472](../sources/prs/sglang/PR-29472.md) | [KDA] Add FlashKDA prefill backend for safe-gate KDA linear attention | 2026-06-27 |  | attention, prefill |
| [#29479](../sources/prs/sglang/PR-29479.md) | [AMD] fix dsv4 indexer dtype dispatch on gfx950 | 2026-06-27 |  | attention, decode |
| [#29486](../sources/prs/sglang/PR-29486.md) | [Cookbook] GLM-5.2: tune GB300 NVFP4 recipes + fill benchmarks | 2026-06-27 |  | fp4, nvfp4 |
| [#29498](../sources/prs/sglang/PR-29498.md) | [DSA] Optimize DSA CUDA graph replay metadata generation | 2026-06-27 |  | attention |
| [#29499](../sources/prs/sglang/PR-29499.md) | [DSA] Optimize DSA CUDA graph replay metadata generation | 2026-06-27 |  | attention |
| [#29508](../sources/prs/sglang/PR-29508.md) | [Bugfix] fix quickreduce acc error in cudagraph mode | 2026-06-27 |  | gemm |
| [#29509](../sources/prs/sglang/PR-29509.md) | [NPU]GLM-4.7-Flash optimize with fused kernels | 2026-06-27 | kernel-fusion | attention, kernel-fusion, mla |
| [#29356](../sources/prs/sglang/PR-29356.md) | fix(bench): pass DCP_RANK/DCP_WORLD_SIZE to set_mla_kv_buffer_kernel | 2026-06-26 |  | mla |
| [#29361](../sources/prs/sglang/PR-29361.md) | [KDA-Pilot] Add diffusion residual-gate CUDA fast path for LTX2 | 2026-06-26 | kernel-fusion | kernel-fusion |
| [#29362](../sources/prs/sglang/PR-29362.md) | [AMD ]Feat/dsv4 ep tbo prefill | 2026-06-26 |  | attention, fp4, fp8 |
| [#29365](../sources/prs/sglang/PR-29365.md) | [CP] Consolidate decode-context-parallel (DCP) helpers into layers/dcp/ | 2026-06-26 |  | attention, decode, mla |
| [#29368](../sources/prs/sglang/PR-29368.md) | [Mamba] Fix long-prefill accuracy drop in radix prefix-cache state restore | 2026-06-26 |  | attention, prefill |
| [#29373](../sources/prs/sglang/PR-29373.md) | [AMD] [GLM5] Guard cuda_runtime.h for ROCm in fused_metadata_copy | 2026-06-26 | kernel-fusion | kernel-fusion |
| [#29374](../sources/prs/sglang/PR-29374.md) | [NPU] Fix mllama cross-attention crash in ascend extend SDPA | 2026-06-26 |  | attention |
| [#29378](../sources/prs/sglang/PR-29378.md) | [CPU] enable fused_sigmoid_mul on CPU device | 2026-06-26 | kernel-fusion | kernel-fusion |
| [#29383](../sources/prs/sglang/PR-29383.md) | feat(sgl-kernel): add InfLLM v2 attention kernels | 2026-06-26 |  | attention, tma |
| [#29390](../sources/prs/sglang/PR-29390.md) | [Diffusion] Fuse LTX2 Ada values | 2026-06-26 | kernel-fusion | kernel-fusion |
| [#29409](../sources/prs/sglang/PR-29409.md) | [AMD] Split qwen3.5 triton DCP test into its own nightly job | 2026-06-26 |  | gemm |
| [#29413](../sources/prs/sglang/PR-29413.md) | [DSA] Enable draft-extend CUDA graph for DeepSeek Sparse Attention | 2026-06-26 |  | attention |
| [#29414](../sources/prs/sglang/PR-29414.md) | [DSA] Make seq_lens_cpu optional for DeepSeek Sparse Attention | 2026-06-26 |  | attention |
| [#29415](../sources/prs/sglang/PR-29415.md) | [DSA] Remove host H2D sync in _apply_cuda_graph_metadata | 2026-06-26 |  | attention |
| [#29421](../sources/prs/sglang/PR-29421.md) | [Feat][GLM5.2] Add DSA Cache Layer Split under Prefill CP | 2026-06-26 |  | attention, prefill |
| [#29423](../sources/prs/sglang/PR-29423.md) | Avoid dynamic Q quantization in trtllm_mha | 2026-06-26 |  | attention, quantization |
| [#29432](../sources/prs/sglang/PR-29432.md) | Fix bookkeeping fields not encapsulated with real allocations in normal alloc, PD pre-alloc, DFlash and EAGLE | 2026-06-26 |  | decode |
| [#29440](../sources/prs/sglang/PR-29440.md) | [MLX] Add correctness tests for qwen2_moe and qwen3_moe | 2026-06-26 |  | attention, moe |
| [#29455](../sources/prs/sglang/PR-29455.md) | shm_broadcast: retry bind on EADDRINUSE (fix dp-attention port race) | 2026-06-26 |  | attention |
| [#29458](../sources/prs/sglang/PR-29458.md) | Enable Breakable Cuda Graph as Default | 2026-06-26 |  | fp4, fp8, gemm |
| [#29460](../sources/prs/sglang/PR-29460.md) | Fix SWA cache loc slicing for all attention backends | 2026-06-26 |  | attention |
| [#29461](../sources/prs/sglang/PR-29461.md) | Fix FlashInfer A2A dispatcher during CUDA graph capture | 2026-06-26 |  | moe |
| [#29462](../sources/prs/sglang/PR-29462.md) | Skip FlashInfer FP8 autotune for MXFP8 quantized models | 2026-06-26 |  | fp8, quantization |
| [#29463](../sources/prs/sglang/PR-29463.md) | [DeepSeek V3] Reland: run routed experts on main stream in dual-stream MoE | 2026-06-26 |  | moe |
| [#29228](../sources/prs/sglang/PR-29228.md) | [Spec] Merge dflash triton kernels into a single `dflash.py` | 2026-06-25 |  | gemm |
| [#29245](../sources/prs/sglang/PR-29245.md) | [Cherry-pick to release/v0.5.14] Fix TRTLLM MHA FP8 KV cache scale handling (#28144) | 2026-06-25 |  | attention, fp8 |
| [#29250](../sources/prs/sglang/PR-29250.md) | Fix MiniMax MSA fallback when fmha plan is unavailable | 2026-06-25 |  | attention, flash-attention |
| [#29252](../sources/prs/sglang/PR-29252.md) | Tune Gemma4 26B-A4B B200 memory recipe | 2026-06-25 |  | gemm |
| [#29253](../sources/prs/sglang/PR-29253.md) | Add MiMo V2.5 Blackwell vision FA4 recipe | 2026-06-25 |  | gemm |
| [#29270](../sources/prs/sglang/PR-29270.md) | [sgl-kernel/cpu]: fix arm64 w8a8 moe kernel signature | 2026-06-25 |  | moe |
| [#29271](../sources/prs/sglang/PR-29271.md) | fix: make write_token dynamic | 2026-06-25 |  | attention |
| [#29275](../sources/prs/sglang/PR-29275.md) | Fix gfx95 bpreshuffle FP8 activation scale layout | 2026-06-25 |  | attention, fp8, mla |
| [#29281](../sources/prs/sglang/PR-29281.md) | [KDA-Pilot] Add diffusion causal Conv3D cat-pad CUDA fast path for Cosmos3 | 2026-06-25 | kernel-fusion, pipeline-stages | kernel-fusion, pipeline-stages |
| [#29286](../sources/prs/sglang/PR-29286.md) | [sgl-kernel/cpu]: exclude amx gemm source from arm build | 2026-06-25 |  | gemm |
| [#29290](../sources/prs/sglang/PR-29290.md) | [AMD] Cover DeepSeek-R1 MXFP4 TP4 MTP nightly CI | 2026-06-25 |  | fp4 |
| [#29333](../sources/prs/sglang/PR-29333.md) | [AMD] Register 1 kernel unit test for AMD nightly CI | 2026-06-25 |  | gemm |
| [#29345](../sources/prs/sglang/PR-29345.md) | test(dp-attn): drop --enable-torch-compile from TestDPAttentionDP2TP2 | 2026-06-25 |  | attention |
| [#29131](../sources/prs/sglang/PR-29131.md) | [NPU] Adapt MiMo-V2.5-W8A8 | 2026-06-24 |  | quantization |
| [#29161](../sources/prs/sglang/PR-29161.md) | [Fix]: Defer DSA MLA CP KV gather for fp8 trtllm prefill in PD mode | 2026-06-24 |  | attention, fp8, mla |
| [#29194](../sources/prs/sglang/PR-29194.md) | [AMD] [GLM5] GLM-5.1 MXFP4 (MI355X) + enable EAGLE for gfx950 in cookbook | 2026-06-24 |  | fp4 |
| [#29197](../sources/prs/sglang/PR-29197.md) | [AMD] Register 5 JIT kernel unit tests for AMD nightly CI | 2026-06-24 | kernel-fusion | kernel-fusion |
| [#29201](../sources/prs/sglang/PR-29201.md) | Fix the CuDNN failure on bmm_fp8 when two libcudart.so exists. | 2026-06-24 |  | fp8, quantization |
| [#29212](../sources/prs/sglang/PR-29212.md) | [Cherry-pick to release/v0.5.14] Fix the CuDNN failure on bmm_fp8 when two libcudart.so exists. (#29201) | 2026-06-24 |  | fp8, quantization |
| [#29218](../sources/prs/sglang/PR-29218.md) | [Spec] DFlash: support pure-MLA targets with an fp8 KV cache (Kimi-K2.x-NVFP4) | 2026-06-24 |  | fp4, fp8, mla |
| [#28975](../sources/prs/sglang/PR-28975.md) | [AMD] [GLM5] Add opt-in Triton fp8 sparse-MLA prefill kernel for gfx950 | 2026-06-23 |  | attention, fp8, mla |
| [#28976](../sources/prs/sglang/PR-28976.md) | [server_args] Reland FA4 page_size auto-force for combined --attention-backend fa4 | 2026-06-23 |  | attention |
| [#28983](../sources/prs/sglang/PR-28983.md) | perf(deepseek_v4): enable SGLANG_OPT_FP8_WO_A_GEMM on sm90 (Hopper) | 2026-06-23 |  | fp8, gemm |
| [#29007](../sources/prs/sglang/PR-29007.md) | Fix MoE TP allreduce to use NCCL symmetric memory via in-pool output allocation | 2026-06-23 | kernel-fusion | attention, gemm, kernel-fusion |
| [#29029](../sources/prs/sglang/PR-29029.md) | [NPU][Bugfix] Fix a ModelSlim loading failure | 2026-06-23 |  | quantization |
| [#29030](../sources/prs/sglang/PR-29030.md) | [NPU] use standalone group for moe ep | 2026-06-23 | kernel-fusion | kernel-fusion, moe |
| [#29069](../sources/prs/sglang/PR-29069.md) | fix(runner): autotune flashinfer MoE on a decode-shaped buffer | 2026-06-23 |  | decode, moe |
| [#29080](../sources/prs/sglang/PR-29080.md) | [Cherry-pick to release/v0.5.14] Add GB10 FP8 fused MoE Triton config (#25665) | 2026-06-23 | kernel-fusion | fp8, kernel-fusion, moe |
| [#28928](../sources/prs/sglang/PR-28928.md) | [diffusion] Add Qwen-Image ModelOpt NVFP4 support | 2026-06-22 | kernel-fusion | fp4, kernel-fusion, nvfp4 |
| [#28967](../sources/prs/sglang/PR-28967.md) | [AMD] Register 7 JIT kernel unit tests for AMD nightly CI | 2026-06-22 | kernel-fusion | decode, fp8, kernel-fusion |
| [#28825](../sources/prs/sglang/PR-28825.md) | [server_args] fix FA4 page_size auto-force for combined --attention-backend fa4 | 2026-06-21 |  | attention |
| [#28757](../sources/prs/sglang/PR-28757.md) | [AMD] [GLM5] skip redundant -inf pre-fill of HIP indexer MQA-logits | 2026-06-20 |  | attention |
| [#28760](../sources/prs/sglang/PR-28760.md) | [diffusion] Optimize realtime causal attention fastpath | 2026-06-20 | kernel-fusion | attention, kernel-fusion |
| [#28770](../sources/prs/sglang/PR-28770.md) | [MLX] Fix Apple Silicon server startup; align MLX tests with upstream | 2026-06-20 |  | attention |
| [#28786](../sources/prs/sglang/PR-28786.md) | [B300] Enable FlashInfer allreduce for Qwen3-VL MoE | 2026-06-20 |  | moe |
| [#28788](../sources/prs/sglang/PR-28788.md) | [AMD] Fix int32 offset overflow in Triton decode-attention kernels | 2026-06-20 |  | attention, decode |
| [#28807](../sources/prs/sglang/PR-28807.md) | Clean up startup log noise | 2026-06-20 |  | attention |
| [#28695](../sources/prs/sglang/PR-28695.md) | [GDN] Support ReplaySSM Ring Spec-Verify | 2026-06-19 |  | attention, decode |
| [#28712](../sources/prs/sglang/PR-28712.md) | [minimax-m3] Split 1/4: sparse attention ops + JIT kernels + config foundation | 2026-06-19 | kernel-fusion | attention, decode, fp8 |
| [#28715](../sources/prs/sglang/PR-28715.md) | [minimax-m3] Split 4/4: model + VL + glue + function-call + fp8 quant + generic infra | 2026-06-19 | kernel-fusion | attention, fp8, gemm |
| [#28722](../sources/prs/sglang/PR-28722.md) | [AMD] Optimize o_proj gemm and attn output rope performance | 2026-06-19 |  | fp8, gemm, quantization |
| [#28736](../sources/prs/sglang/PR-28736.md) | [AMD] register 3 tests to stage-b-test-1-gpu-large-amd (batch-6) | 2026-06-19 |  | attention, decode |
| [#28749](../sources/prs/sglang/PR-28749.md) | Fix nightly CI test for GLM-4.6 + B200 | 2026-06-19 |  | gemm |
| [#28612](../sources/prs/sglang/PR-28612.md) | Optimize C128 state pool allocation using request state pool | 2026-06-18 |  | attention, decode, prefill |
| [#28629](../sources/prs/sglang/PR-28629.md) | [Bugfix] Fix Intern-S1 FP8 expert count lookup | 2026-06-18 |  | fp8 |
| [#28644](../sources/prs/sglang/PR-28644.md) | [codex] Fix DSA indexer in prefill piecewise CUDA graph | 2026-06-18 |  | attention, prefill |
| [#28649](../sources/prs/sglang/PR-28649.md) | Pass quant_config to attention gate projection | 2026-06-18 |  | attention |
| [#28658](../sources/prs/sglang/PR-28658.md) | [AMD] Fuse shared-expert sigmoid + bf16->fp32 cast into the MoE append kernel (3 kernels -> 1) | 2026-06-18 | kernel-fusion | kernel-fusion, moe |
| [#28662](../sources/prs/sglang/PR-28662.md) | [Fix] compressed-tensors block FP8: requantize weight scales to UE8M0 for DeepGEMM on Blackwell | 2026-06-18 |  | fp8, gemm, quantization |
| [#28664](../sources/prs/sglang/PR-28664.md) | [Cookbook] Laguna-M.1: enable FP8 on Blackwell + drop provisional AIME numbers | 2026-06-18 |  | fp8 |
| [#28670](../sources/prs/sglang/PR-28670.md) | [JIT] Add kpool_topk_transform JIT kernel | 2026-06-18 |  | gemm |
| [#28676](../sources/prs/sglang/PR-28676.md) | [RL] fix deepseek v4 MXFP8 flashinfer_trtllm_routed MoE weight update | 2026-06-18 | kernel-fusion | fp8, kernel-fusion, moe |
| [#28689](../sources/prs/sglang/PR-28689.md) | [MoE] dedup triton_kernels backend quant-arg asserts and fill weight dtype guard | 2026-06-18 | kernel-fusion | kernel-fusion, moe |
| [#28469](../sources/prs/sglang/PR-28469.md) | fix(moe): MoRI EP init_mori_op missing BF16 dispatch branch | 2026-06-17 |  | moe |
| [#28486](../sources/prs/sglang/PR-28486.md) | [bugfix] guard NVIDIA SM-capability checks with is_cuda() for AMD/ROCm | 2026-06-17 |  | attention, moe |
| [#28495](../sources/prs/sglang/PR-28495.md) | [AMD] Register DP attention test | 2026-06-17 |  | attention |
| [#28505](../sources/prs/sglang/PR-28505.md) | :recycle: [llm][npu][quant] Delegate MXFP8 dense scheme to kernel and use torch.ops.npu | 2026-06-17 |  | fp8, quantization |
| [#28520](../sources/prs/sglang/PR-28520.md) | [AMD] Fix deepseek-v4 mtp accept length issue | 2026-06-17 |  | attention, fp4 |
| [#28546](../sources/prs/sglang/PR-28546.md) | [diffusion] Fix FP8 fused TP scale loading | 2026-06-17 | kernel-fusion | fp8, kernel-fusion |
| [#28555](../sources/prs/sglang/PR-28555.md) | Remove redundant cast and copy in calling `trtllm_fp8_block_scale_moe` | 2026-06-17 |  | block-scale, fp8, moe |
| [#28559](../sources/prs/sglang/PR-28559.md) | fix: speculative draft worker clobbering target attention backend | 2026-06-17 |  | attention |
| [#28567](../sources/prs/sglang/PR-28567.md) | Add get_parallel(): a structured accessor for parallel-topology state | 2026-06-17 | kernel-fusion | attention, fp8, gemm |
| [#28577](../sources/prs/sglang/PR-28577.md) | [Test] Fold EAGLE `return_hidden_states` regression into spec triton suite | 2026-06-17 |  | gemm |
| [#28579](../sources/prs/sglang/PR-28579.md) | [spec decoding] fully overlap spec decoding for hybrid linear attention backend | 2026-06-17 |  | attention, decode |
| [#28367](../sources/prs/sglang/PR-28367.md) | Fix Stage B CUDA CI | 2026-06-16 |  | gemm |
| [#28392](../sources/prs/sglang/PR-28392.md) | [AMD] Annotate ATOM source for imported v4 unified attention kernels | 2026-06-16 |  | attention, decode, prefill |
| [#28404](../sources/prs/sglang/PR-28404.md) | [AMD][Fix] Skip EPLB topk remap when global server args are unset | 2026-06-16 |  | moe |
| [#28416](../sources/prs/sglang/PR-28416.md) | [GLM5][MoE] perf: Write FlashInfer TRT-LLM MoE output directly | 2026-06-16 |  | moe |
| [#28434](../sources/prs/sglang/PR-28434.md) | [HiCache]Support hybrid pool staged H2D kernel | 2026-06-16 |  | gemm |
| [#28436](../sources/prs/sglang/PR-28436.md) | [NPU] Use use_dsa to dispatch Ascend DSA attention | 2026-06-16 |  | attention |
| [#28439](../sources/prs/sglang/PR-28439.md) | [Intel GPU] DeepSeek V4 13/N: use sgl-kernel implementation of kernels in V2 Compressor to run on XPU | 2026-06-16 |  | fp4 |
| [#28450](../sources/prs/sglang/PR-28450.md) | [AMD] Fuse shared-expert append + DeepEP remap into one Triton kernel | 2026-06-16 | kernel-fusion | kernel-fusion, moe |
| [#28451](../sources/prs/sglang/PR-28451.md) | [GDN][KDA] ReplaySSM buffered output-only decode for Linear Attention | 2026-06-16 | kernel-fusion | attention, decode, kernel-fusion |
| [#28455](../sources/prs/sglang/PR-28455.md) | [AMD] Fix DeepSeek-V4 fp8 KV path on gfx942 (e4m3fnuz) | 2026-06-16 | kernel-fusion | attention, decode, fp8 |
| [#28458](../sources/prs/sglang/PR-28458.md) | [AMD] ci: add extra-a 1-gpu-large tier (fp8kv-triton, streaming-session, spec-standalone) | 2026-06-16 |  | fp8 |
| [#28459](../sources/prs/sglang/PR-28459.md) | [RL] Fix FlashInfer TRTLLM MXFP8 dense weight layout | 2026-06-16 |  | fp8, moe, quantization |
| [#28460](../sources/prs/sglang/PR-28460.md) | docs(cookbook): verify GLM-5.2 single-node B300 (FP8 + BF16) | 2026-06-16 |  | fp8 |
| [#28216](../sources/prs/sglang/PR-28216.md) | [AMD] Feat/dp moe reduce scatter | 2026-06-15 |  | attention, moe |
| [#28220](../sources/prs/sglang/PR-28220.md) | [FlashInfer v0.6.13] Use CuTe DSL backend for FlashInfer per-token NVFP4 quantization | 2026-06-15 |  | fp4, moe, nvfp4 |
| [#28221](../sources/prs/sglang/PR-28221.md) | Fix EagleDraftExtendInput missing kv_indptr crash with triton/DP attention | 2026-06-15 |  | attention |
| [#28231](../sources/prs/sglang/PR-28231.md) | Use Marlin for SM120 MXFP4 MoE | 2026-06-15 | kernel-fusion | fp4, kernel-fusion, moe |
| [#28237](../sources/prs/sglang/PR-28237.md) |  [AMD] fix(moe): correct fused shared-expert scaling on aiter/DeepEP path (mori all-to-all) | 2026-06-15 | kernel-fusion | kernel-fusion, moe |
| [#28244](../sources/prs/sglang/PR-28244.md) | [AMD] Fix garbled unquantized Qwen3-30B-A3B output on ROCm/aiter where the aiter CK fused-MoE falls back to Triton with pre-shuffled weights | 2026-06-15 | kernel-fusion | kernel-fusion, moe, quantization |
| [#28265](../sources/prs/sglang/PR-28265.md) | [AMD] refactor sparse MLA decode kernel for Deepseek V4 triton backend | 2026-06-15 | kernel-fusion | attention, decode, kernel-fusion |
| [#28290](../sources/prs/sglang/PR-28290.md) | [AMD] Test DeepSeek V4 FlashMLA backend variants nightly | 2026-06-15 |  | fp4, fp8, mla |
| [#28291](../sources/prs/sglang/PR-28291.md) | [AMD][MXFP4] Reland "Online MXFP4 quantization 2/N - FP8 to MXFP4 requantization on AMD GPUs" | 2026-06-15 |  | fp4, fp8, moe |
| [#28292](../sources/prs/sglang/PR-28292.md) | [Fix] model init / XPU / transformers-v5 / bench-image fixes | 2026-06-15 |  | gemm, moe |
| [#28293](../sources/prs/sglang/PR-28293.md) | [NPU] Add NPU fallback for fused Triton gating kernels | 2026-06-15 | kernel-fusion | kernel-fusion, moe |
| [#28309](../sources/prs/sglang/PR-28309.md) | Support Flashinfer one-sided A2A + CuteDSL MoE for Nemotron Ultra | 2026-06-15 |  | moe |
| [#28320](../sources/prs/sglang/PR-28320.md) | Fused QK GemmaRMSNorm + RoPE + gate kernel for Qwen3.5 | 2026-06-15 | kernel-fusion | attention, gemm, kernel-fusion |
| [#28333](../sources/prs/sglang/PR-28333.md) | Call Flashinfer `mm_fp8` for per-tensor FP8 GEMMs on SM100 | 2026-06-15 |  | fp8, gemm, quantization |
| [#28169](../sources/prs/sglang/PR-28169.md) | [diffusion] UX: reduce attention backend log noise | 2026-06-14 | kernel-fusion | attention, kernel-fusion |
| [#28205](../sources/prs/sglang/PR-28205.md) | [diffusion] Persist torch.compile Inductor/Triton cache across restarts | 2026-06-14 | kernel-fusion | kernel-fusion |
| [#28211](../sources/prs/sglang/PR-28211.md) | [MoE Refactor] Centralize FlashInfer CUTLASS MoE runner | 2026-06-14 |  | fp4, moe, quantization |
| [#28144](../sources/prs/sglang/PR-28144.md) | Fix TRTLLM MHA FP8 KV cache scale handling | 2026-06-13 |  | attention, fp8 |
| [#27986](../sources/prs/sglang/PR-27986.md) | [dsv4] Prewarm MHC prenorm kernel at startup | 2026-06-12 |  | gemm |
| [#27988](../sources/prs/sglang/PR-27988.md) | [Experimental] Full Cuda Graph Support for Prefill | 2026-06-12 |  | attention, prefill |
| [#28013](../sources/prs/sglang/PR-28013.md) | Fix invalid KVFP4QuantizeUtil references | 2026-06-12 |  | fp4, quantization |
| [#28041](../sources/prs/sglang/PR-28041.md) | Fix stale CUDA graph benchmark and docs refs | 2026-06-12 | kernel-fusion | kernel-fusion, moe |
| [#28073](../sources/prs/sglang/PR-28073.md) | fix: Fix DSR1 perf regression due to unnecessarily falling back to triton gemm | 2026-06-12 |  | fp8, gemm, quantization |
| [#28084](../sources/prs/sglang/PR-28084.md) | [AMD] Fuse topk padded-token masking into a single Triton kernel | 2026-06-12 |  | moe |
| [#28091](../sources/prs/sglang/PR-28091.md) | [LoRA] Fix experimental fast-path multi-adapter correctness + flashinfer 0.6.12 compatibility | 2026-06-12 | kernel-fusion | fp8, gemm, kernel-fusion |
| [#28102](../sources/prs/sglang/PR-28102.md) | Fix DP attention + EP mode of Nemotron | 2026-06-12 |  | attention |
| [#28106](../sources/prs/sglang/PR-28106.md) | [attn backend] Make seq_lens_cpu optional in trtllm_mha backend | 2026-06-12 |  | attention |
| [#27855](../sources/prs/sglang/PR-27855.md) | [AMD] fix moriep quant kernel not implemented issue | 2026-06-11 |  | moe |
| [#27858](../sources/prs/sglang/PR-27858.md) | [AMD] Fix the dsv4 performance of MoE issue. | 2026-06-11 |  | fp8, moe, quantization |
| [#27869](../sources/prs/sglang/PR-27869.md) | Fix Qwen3.5 deterministic batch-invariant logprobs | 2026-06-11 | kernel-fusion | attention, kernel-fusion, moe |
| [#27896](../sources/prs/sglang/PR-27896.md) | [Perf] Skip per-call mat_a/scales_a padding in cutlass FP8 blockwise GEMM | 2026-06-11 |  | fp8, gemm, quantization |
| [#27906](../sources/prs/sglang/PR-27906.md) | [Model] Support Qwen3.6 ModelOpt mixed NVFP4 | 2026-06-11 |  | fp4, nvfp4, quantization |
| [#27915](../sources/prs/sglang/PR-27915.md) | [Intel GPU] DeepSeek V4 7/N: Support fused_rope_inplace on XPU using triton | 2026-06-11 | kernel-fusion | kernel-fusion |
| [#27926](../sources/prs/sglang/PR-27926.md) | [DSV4] perf: Make FP8 quant output tensor contiguous | 2026-06-11 |  | fp8 |
| [#27928](../sources/prs/sglang/PR-27928.md) | [AMD] Feat: Add prefill context parallel support for deepseek v4 unified kv attention | 2026-06-11 |  | attention, fp4, prefill |
| [#27935](../sources/prs/sglang/PR-27935.md) | [AMD] Support unified_kv_triton for disaggregation | 2026-06-11 |  | decode, prefill |
| [#27939](../sources/prs/sglang/PR-27939.md) | Support online MXFP8 quantization for ungated MoE | 2026-06-11 |  | fp8, moe, quantization |
| [#27945](../sources/prs/sglang/PR-27945.md) | fix(moe): make FlashInfer A2A robust to collapsed global_num_tokens (moe_dense_tp_size NaN) | 2026-06-11 |  | moe |
| [#27972](../sources/prs/sglang/PR-27972.md) | [AMD] Fix DeepSeek-V4-Flash-FP8 on MI300 | 2026-06-11 |  | attention, fp8, prefill |
| [#27739](../sources/prs/sglang/PR-27739.md) | step3.5 flash revise for graph mode and use triton activation | 2026-06-10 |  | attention, quantization |
| [#27761](../sources/prs/sglang/PR-27761.md) | [Spec] Remove dead `prepare_for_verify` / `prepare_extend_after_decode` + extend-decode kernel | 2026-06-10 |  | decode |
| [#27782](../sources/prs/sglang/PR-27782.md) | Fix gpt-oss-20b with mxfp4 support for Xeon | 2026-06-10 |  | fp4, quantization |
| [#27793](../sources/prs/sglang/PR-27793.md) | [AMD][Perf] Tune extend attention block sizes for gfx950 (head_dim > 128) | 2026-06-10 |  | attention |
| [#27817](../sources/prs/sglang/PR-27817.md) | [AMD] ci: register 8 attention-backend unit tests to run on AMD CI | 2026-06-10 |  | attention |
| [#27835](../sources/prs/sglang/PR-27835.md) | [bugfix][AMD] Disable aiter allreduce+RMSNorm fusion under DP attention / EP | 2026-06-10 | kernel-fusion | attention, kernel-fusion |
| [#27841](../sources/prs/sglang/PR-27841.md) | Remove MoE prefill CUDA graph disable guard | 2026-06-10 |  | moe, prefill |
| [#27617](../sources/prs/sglang/PR-27617.md) | [SWA] Cache full→SWA out_cache_loc per forward across attention backends | 2026-06-09 |  | attention |
| [#27624](../sources/prs/sglang/PR-27624.md) | triton-ascend update | 2026-06-09 |  | gemm |
| [#27630](../sources/prs/sglang/PR-27630.md) | [AMD] Fuse sigmoid + mul attention output gate into single Triton kernel | 2026-06-09 |  | attention |
| [#27636](../sources/prs/sglang/PR-27636.md) | [AMD] Fuse sigmoid + mul into single Triton kernel for shared expert gating | 2026-06-09 |  | moe |
| [#27639](../sources/prs/sglang/PR-27639.md) | [diffusion] model: support LongLive 2.0 T2V and I2V inference | 2026-06-09 | kernel-fusion, pipeline-stages | attention, kernel-fusion, pipeline-stages |
| [#27690](../sources/prs/sglang/PR-27690.md) | Support asymmetric compressed-tensors MoE | 2026-06-09 | kernel-fusion | kernel-fusion, moe, quantization |
| [#27705](../sources/prs/sglang/PR-27705.md) | Fuse the DSA (V3.2, GLM-5.x) indexer Q/K paths into single kernels | 2026-06-09 | kernel-fusion | attention, kernel-fusion |
| [#27553](../sources/prs/sglang/PR-27553.md) | fix: preserve divisible FP8 block K configs on CUDA | 2026-06-08 |  | fp8, quantization |
| [#27583](../sources/prs/sglang/PR-27583.md) | [AMD] Enable fused GDN QKV split Triton kernel on HIP | 2026-06-08 | kernel-fusion | attention, kernel-fusion |
| [#27588](../sources/prs/sglang/PR-27588.md) | [quantization] NVFP4 MoE: split fused w13 gate/up global scales | 2026-06-08 | kernel-fusion | fp4, kernel-fusion, moe |
| [#27590](../sources/prs/sglang/PR-27590.md) | [diffusion] Use fused FP8 GEMM for Ideogram4 weight-only linears | 2026-06-08 | kernel-fusion | fp8, gemm, kernel-fusion |
| [#27469](../sources/prs/sglang/PR-27469.md) | dflash add sliding window attention draft layer support | 2026-06-07 |  | attention |
| [#27488](../sources/prs/sglang/PR-27488.md) | [KDA] Add CuteDSL Prefill Kernel on SM100 | 2026-06-07 |  | attention, prefill |
| [#27510](../sources/prs/sglang/PR-27510.md) | [deepseek] Enable DP attention + TBO + shared experts fusion | 2026-06-07 | kernel-fusion | attention, kernel-fusion |
| [#27429](../sources/prs/sglang/PR-27429.md) | [codex] Centralize more inline Triton kernels | 2026-06-06 |  | attention, mla |
| [#27449](../sources/prs/sglang/PR-27449.md) | [jit-kernel] Support per token group quant 8bit v2 jit kernel | 2026-06-06 |  | fp8, gemm, quantization |
| [#27455](../sources/prs/sglang/PR-27455.md) | [SM120] Add FlashInfer sparse MLA decode for DSv4-Flash | 2026-06-06 |  | attention, decode, mla |
| [#27459](../sources/prs/sglang/PR-27459.md) | [core] Probe `set_kv_buffer` / `set_mla_kv_buffer` slot ids for OOB | 2026-06-06 |  | mla |
| [#27463](../sources/prs/sglang/PR-27463.md) | Support `topk > 1` tree drafting for mamba/hybrid-linear models on spec v2 | 2026-06-06 |  | attention |
| [#27328](../sources/prs/sglang/PR-27328.md) | fix: add missing clamp_limit for CompressedTensorsWNA16MoE | 2026-06-05 |  | moe, quantization |
| [#27329](../sources/prs/sglang/PR-27329.md) | [LoRA] Experimental fast LoRA path with `experimental_sgl_trtllm` MoE backend for FP8 and NVFP4 models | 2026-06-05 | kernel-fusion | attention, fp4, fp8 |
| [#27342](../sources/prs/sglang/PR-27342.md) | test: fix gemma GSM8K thresholds in nightly text eval | 2026-06-05 |  | gemm |
| [#27343](../sources/prs/sglang/PR-27343.md) | fix(fa3): no NaN embeddings with fa_skip_kv_cache under piecewise CUDA graph | 2026-06-05 |  | attention, prefill |
| [#27349](../sources/prs/sglang/PR-27349.md) | Support DSV4 shared expert fusion for DeepEP and MegaMOE | 2026-06-05 | kernel-fusion | fp4, fp8, kernel-fusion |
| [#27350](../sources/prs/sglang/PR-27350.md) | Support Waterfill with MegaMoE backend | 2026-06-05 |  | moe |
| [#27375](../sources/prs/sglang/PR-27375.md) | [Model] Add support for JetBrains' Mellum v2 code generation model | 2026-06-05 | kernel-fusion | kernel-fusion, moe |
| [#27380](../sources/prs/sglang/PR-27380.md) | [AMD] Add unified kv attention support in dpsk-v4 | 2026-06-05 | kernel-fusion | attention, decode, kernel-fusion |
| [#27382](../sources/prs/sglang/PR-27382.md) | [AMD][Perf] Split-KV flash-decode attention for EAGLE target-verify (Triton backend) | 2026-06-05 |  | attention, decode |
| [#27401](../sources/prs/sglang/PR-27401.md) | [Cohere2Moe] Enable flashinfer_trtllm NVFP4 fused-MoE via SigmoidRenorm routing | 2026-06-05 | kernel-fusion | fp4, kernel-fusion, moe |
| [#27204](../sources/prs/sglang/PR-27204.md) | [AMD] Implement QuarkW4A8MXFp4MoE to support amd/gpt-oss-120b-w-mxfp4-a-fp8 | 2026-06-04 |  | fp4, fp8, moe |
| [#27279](../sources/prs/sglang/PR-27279.md) | [Multimodal] Add Ideogram 4 FP8 generation support | 2026-06-04 | pipeline-stages | fp8, pipeline-stages, quantization |
| [#27111](../sources/prs/sglang/PR-27111.md) | [AMD] Minimax M25 : FP8 block-scale GEMM dispatch for ROCm 7.0 on gfx950 | 2026-06-03 |  | block-scale, fp8, gemm |
| [#27127](../sources/prs/sglang/PR-27127.md) | use sgl_kernel_npu rmsrope accelerate llada2 | 2026-06-03 |  | gemm |
| [#27131](../sources/prs/sglang/PR-27131.md) | fp8 kv cache: treat non-positive k/v_scale as uncalibrated (default 1.0) | 2026-06-03 |  | fp8, quantization |
| [#26982](../sources/prs/sglang/PR-26982.md) | [Spec] Add `move_kv_cache` to `MLATokenToKVPool` for EAGLE v2 verify | 2026-06-02 |  | mla |
| [#27057](../sources/prs/sglang/PR-27057.md) | [AMD] move shared expert check function to quark | 2026-06-02 |  | moe, quantization |
| [#27063](../sources/prs/sglang/PR-27063.md) | [AMD] Optimize gpt-oss-120B performance | 2026-06-02 |  | attention, fp4, quantization |
| [#26884](../sources/prs/sglang/PR-26884.md) | [AMD] Fix GPT-OSS MXFP4 accuracy on ROCm AITER path | 2026-06-01 |  | fp4, moe, quantization |
| [#26894](../sources/prs/sglang/PR-26894.md) | [AMD] Fuse compress norm+rope+hadamard into single Triton kernel | 2026-06-01 | kernel-fusion | attention, kernel-fusion |
| [#26929](../sources/prs/sglang/PR-26929.md) | Add stochastic rounding for FP16 Mamba SSM cache | 2026-06-01 |  | attention |
| [#26806](../sources/prs/sglang/PR-26806.md) | Add the KV-canary write JIT kernel and reference implementation | 2026-05-31 |  | gemm |
| [#26845](../sources/prs/sglang/PR-26845.md) | [Qwen3.5][AMD] Fix shared-expert ×ep_size over-count under allreduce-EP | 2026-05-31 |  | moe |
| [#26852](../sources/prs/sglang/PR-26852.md) | [AMD]Reuse fused FP8 KV cache write on standard aiter prefill/decode | 2026-05-31 | kernel-fusion | attention, decode, fp8 |
| [#26766](../sources/prs/sglang/PR-26766.md) | [DeepSeek-V4] Fuse UE8M0 scale rounding into FP8 group quantization | 2026-05-30 |  | fp4, fp8, gemm |
| [#26784](../sources/prs/sglang/PR-26784.md) | update quantization code owner and document quantization contributions | 2026-05-30 |  | quantization |
| [#26786](../sources/prs/sglang/PR-26786.md) | [GPTQ] Refactor CPU quantization schemes | 2026-05-30 |  | moe, quantization |
| [#26791](../sources/prs/sglang/PR-26791.md) | Fix Gemma4 NVFP4 MoE default attention backend | 2026-05-30 |  | attention, fp4, gemm |
| [#26639](../sources/prs/sglang/PR-26639.md) | [AMD] Enable HiSparse on ROCm | 2026-05-29 |  | attention |
| [#26651](../sources/prs/sglang/PR-26651.md) | Fix DRAFT_EXTEND_V2 CG metadata: align test fixture and Triton with production seq_lens convention | 2026-05-29 |  | attention |
| [#26665](../sources/prs/sglang/PR-26665.md) | [refactor] unify cuda-graph capture/replay across attention backends | 2026-05-29 |  | attention, mla |
| [#26704](../sources/prs/sglang/PR-26704.md) | pin kernels<0.15 | 2026-05-29 |  | gemm |
| [#26710](../sources/prs/sglang/PR-26710.md) | Fix MoE LoRA wrapper exposing moe_runner_config | 2026-05-29 |  | moe |
| [#26717](../sources/prs/sglang/PR-26717.md) | [NPU] RL update_weights_from_disk/ tensor /distributed | 2026-05-29 | kernel-fusion | kernel-fusion, moe |
| [#26746](../sources/prs/sglang/PR-26746.md) | Support optional kwargs in AITER fused_moe runner | 2026-05-29 | kernel-fusion | kernel-fusion, moe |
| [#26588](../sources/prs/sglang/PR-26588.md) | Optimize Gemma4 H200 MoE and extend attention | 2026-05-28 | kernel-fusion | attention, gemm, kernel-fusion |
| [#26437](../sources/prs/sglang/PR-26437.md) | [MUSA] Fix startup with patched torchada | 2026-05-27 |  | gemm |
| [#26468](../sources/prs/sglang/PR-26468.md) | [Model] Add Qwen3-MoE MTP | 2026-05-27 |  | moe |
| [#26496](../sources/prs/sglang/PR-26496.md) | Changes for SM120 perf and usability for NVFP4 | 2026-05-27 | kernel-fusion | fp4, gemm, kernel-fusion |
| [#26499](../sources/prs/sglang/PR-26499.md) | [Kernel] Import flash_mla kernels from sglang kernel for deepseek v4 | 2026-05-27 |  | attention, mla |
| [#26514](../sources/prs/sglang/PR-26514.md) | Expose Flex attention causal/decode masks as static methods | 2026-05-27 |  | attention, decode |
| [#26517](../sources/prs/sglang/PR-26517.md) | Add attention-backend unit-test suite under test/registered/attention/unittest | 2026-05-27 |  | attention, decode, mla |
| [#26335](../sources/prs/sglang/PR-26335.md) | [Spec] Async-assert probes across EAGLE/MTP; zero `tgt_cache_loc` | 2026-05-26 |  | attention, fp4 |
| [#26402](../sources/prs/sglang/PR-26402.md) | [5/N] Quantization Refactor: GPTQ schemes and kernel split | 2026-05-26 |  | moe, quantization |
| [#26415](../sources/prs/sglang/PR-26415.md) | [Fix] Fix FP8 Online Quantization | 2026-05-26 |  | fp4, fp8, quantization |
| [#26255](../sources/prs/sglang/PR-26255.md) | [fix] Add support for flashinfer MOE A2A to Qwen3 BF16 model path | 2026-05-25 |  | moe, tma |
| [#26303](../sources/prs/sglang/PR-26303.md) | [MoE] Extend kimi_k2_moe_fused_gate to support 256 experts (MiMo V2 Flash) | 2026-05-25 | kernel-fusion | kernel-fusion, moe |
| [#26309](../sources/prs/sglang/PR-26309.md) | [npu] [bugfix] Add contiguous operation during quantized weight loading. | 2026-05-25 | kernel-fusion | kernel-fusion, moe, quantization |
| [#26204](../sources/prs/sglang/PR-26204.md) | Optimize Qwen3 Next FP8 MoE on H200 | 2026-05-24 |  | fp8, moe |
| [#26147](../sources/prs/sglang/PR-26147.md) | [NPU] Add Gemma4 Sliding Window Attention support on Ascend backend | 2026-05-23 |  | attention, gemm |
| [#26026](../sources/prs/sglang/PR-26026.md) | [bug fix] Fix 3 issues when using Gemma4 MTP | 2026-05-22 |  | gemm |
| [#26083](../sources/prs/sglang/PR-26083.md) | Implement online nvfp4 quantization | 2026-05-22 |  | fp4, moe, nvfp4 |
| [#26123](../sources/prs/sglang/PR-26123.md) | Fix routed-experts device buffer overflow under DP attention | 2026-05-22 |  | attention |
| [#26000](../sources/prs/sglang/PR-26000.md) | [codex] Centralize Triton utility kernels | 2026-05-21 |  | attention, mla |
| [#25835](../sources/prs/sglang/PR-25835.md) | [JIT Kernel] Triton moe fused gate | 2026-05-20 | kernel-fusion | kernel-fusion, moe |
| [#25874](../sources/prs/sglang/PR-25874.md) | [CPU] add faster KV-cache writes | 2026-05-20 |  | gemm |
| [#25904](../sources/prs/sglang/PR-25904.md) | :memo: docs(diffusion): add MXFP4 quantization docs | 2026-05-20 | kernel-fusion | fp4, kernel-fusion, quantization |
| [#25702](../sources/prs/sglang/PR-25702.md) | Use pack topk ids triton kernel for flashinfer_trtllm_routed | 2026-05-19 |  | moe |
| [#25751](../sources/prs/sglang/PR-25751.md) | [Kernel] Add SM90 Q8KV8 FP8 Sparse MLA Prefill JIT Kernel with Tests and Benchmark | 2026-05-19 |  | fp8, mla, prefill |
| [#25763](../sources/prs/sglang/PR-25763.md) | [Feature] Support DeepSeek-V4 Wint4Abf16 and Win4Afp8. | 2026-05-19 |  | fp8, gemm, moe |
| [#25814](../sources/prs/sglang/PR-25814.md) | Update GLM-5 H200 FP8 | 2026-05-19 |  | fp8 |
| [#25820](../sources/prs/sglang/PR-25820.md) | [NVIDIA] Support NVFP4 MoE for DeepSeek-V4 | 2026-05-19 |  | fp4, moe, nvfp4 |
| [#25569](../sources/prs/sglang/PR-25569.md) | Add DeepSeekV4 fused MoE Triton autotune support | 2026-05-18 | kernel-fusion | kernel-fusion, moe |
| [#25611](../sources/prs/sglang/PR-25611.md) | Introduce SchedulerDPAttnAdapter to own DP-attention state | 2026-05-18 |  | attention, decode, prefill |
| [#25612](../sources/prs/sglang/PR-25612.md) | Move DP-attention adapter methods to SchedulerDPAttnAdapter | 2026-05-18 |  | attention, decode, prefill |
| [#25663](../sources/prs/sglang/PR-25663.md) | [MoE Refactor] [NPU] Refactor Ascend MoE implementation to reduce code duplication and align with community design | 2026-05-18 | kernel-fusion | kernel-fusion, moe, quantization |
| [#25665](../sources/prs/sglang/PR-25665.md) | Add GB10 FP8 fused MoE Triton config | 2026-05-18 | kernel-fusion | fp8, kernel-fusion, moe |
| [#25694](../sources/prs/sglang/PR-25694.md) | [Quantization][Bugfix]: Join multi-arg RuntimeError in Quark _check_scheme_supported | 2026-05-18 |  | quantization |
| [#25514](../sources/prs/sglang/PR-25514.md) | [diffusion] Clean up VSA attention hot path | 2026-05-17 | kernel-fusion, pipeline-stages | attention, kernel-fusion, pipeline-stages |
| [#25519](../sources/prs/sglang/PR-25519.md) | [Quantization][bugfix] Correct E8M0 NaN-sentinel detection in e8m0_to_f32 | 2026-05-17 |  | quantization |
| [#25547](../sources/prs/sglang/PR-25547.md) | Respect user override for Gemma4 attention backend | 2026-05-17 |  | attention, gemm |
| [#25467](../sources/prs/sglang/PR-25467.md) | [Quantization] Update error message strings with correct framework name in Quark/compressed-tensors | 2026-05-16 |  | quantization |
| [#25367](../sources/prs/sglang/PR-25367.md) | Fix EPLB redundant experts with shared expert fusion and Waterfill | 2026-05-15 | kernel-fusion | kernel-fusion, moe |
| [#25379](../sources/prs/sglang/PR-25379.md) | feat(moe): reuse prev-layer output as symm_output for FP4 routed MoE | 2026-05-15 | kernel-fusion | fp4, kernel-fusion, moe |
| [#25406](../sources/prs/sglang/PR-25406.md) | [MoE] Decouple Mega MoE from DeepEP backend | 2026-05-15 | kernel-fusion | fp4, fp8, gemm |
| [#25220](../sources/prs/sglang/PR-25220.md) | Cute-DSL FP8 MQA logits  | 2026-05-14 |  | attention, fp8, gemm |
| [#25221](../sources/prs/sglang/PR-25221.md) | [MLX] bench_one_batch: thread --quantization through to MlxModelRunner | 2026-05-14 |  | quantization |
| [#25279](../sources/prs/sglang/PR-25279.md) | DeepseekV2MoE: defer shared experts when routed kernel is non-mutating | 2026-05-14 |  | moe |
| [#25286](../sources/prs/sglang/PR-25286.md) | [Gemma4]: Fix FP8 Triton scale layout | 2026-05-14 |  | fp8, gemm, quantization |
| [#25292](../sources/prs/sglang/PR-25292.md) | [Quant] Support asymmetric weight quant in compressed-tensors WNA16 | 2026-05-14 |  | quantization |
| [#25141](../sources/prs/sglang/PR-25141.md) | support kimi 2.5/6 lora (logprob diff exist) | 2026-05-13 |  | moe |
| [#25180](../sources/prs/sglang/PR-25180.md) | Fix AMX GQA extend attention | 2026-05-13 |  | attention |
| [#25191](../sources/prs/sglang/PR-25191.md) | [Apple Silicon] [MLX] Auto-detect MLX-format quantization_config dict | 2026-05-13 |  | quantization |
| [#25076](../sources/prs/sglang/PR-25076.md) | Fix fused_moe import for non-NPU devices | 2026-05-12 | kernel-fusion | kernel-fusion, moe |
| [#25090](../sources/prs/sglang/PR-25090.md) | [AMD] Support triton backend decode context parallel for Qwen3.5 | 2026-05-12 |  | attention, decode |
| [#24918](../sources/prs/sglang/PR-24918.md) | :memo: docs(diffusion): add MXFP8 quantization docs for Wan2.2 on Ascend NPU | 2026-05-11 | kernel-fusion | fp8, kernel-fusion, quantization |
| [#24925](../sources/prs/sglang/PR-24925.md) | [attn backend] Integrate tokenspeed_mla prefill/decode kernels (fp8 kv cache, blackwell) | 2026-05-11 |  | attention, decode, fp8 |
| [#24947](../sources/prs/sglang/PR-24947.md) | DeepSeek V4: Support context parallelism with fused MoE (non-DeepEP)  | 2026-05-11 | kernel-fusion | fp4, fp8, kernel-fusion |
| [#24955](../sources/prs/sglang/PR-24955.md) | Support Nemotron DP attention and MTP | 2026-05-11 |  | attention |
| [#24971](../sources/prs/sglang/PR-24971.md) | amd/deepseek_v4 integration 17/N flydsl moe 0508 | 2026-05-11 |  | attention, fp8, mla |
| [#24879](../sources/prs/sglang/PR-24879.md) | [AMD] support fp8 blockwise quantization combine for mori ep | 2026-05-10 |  | fp8, moe, quantization |
| [#24884](../sources/prs/sglang/PR-24884.md) | [MoE] Decouple Mega MoE from DeepEP backend | 2026-05-10 | kernel-fusion | fp4, fp8, gemm |
| [#24907](../sources/prs/sglang/PR-24907.md) | [MLX] Add on-the-fly --quantization mlx_q4 / mlx_q8 for Apple Silicon | 2026-05-10 |  | quantization |
| [#24737](../sources/prs/sglang/PR-24737.md) | Support Flashinfer Cute-DSL MLA attention | 2026-05-09 |  | attention, mla |
| [#24816](../sources/prs/sglang/PR-24816.md) | Add FlashInfer SM90 cutlass MXFP4 MoE backend (W4A16) for GPT-OSS + DeepSeek-V4 | 2026-05-09 |  | fp4, fp8, moe |
| [#24651](../sources/prs/sglang/PR-24651.md) | [AMD] Add fused all-reduce RMSNorm per-group quant for Qwen3.5 FP8 | 2026-05-08 | kernel-fusion | fp8, kernel-fusion |
| [#24664](../sources/prs/sglang/PR-24664.md) | Feat: Support SWA (Sliding Window Attention) for EAGLE-3 drafter | 2026-05-08 |  | attention |
| [#24668](../sources/prs/sglang/PR-24668.md) | [NPU]Documentation update for communications quantization feature | 2026-05-08 |  | quantization |
| [#24692](../sources/prs/sglang/PR-24692.md) | feat: SM120 (Blackwell Desktop) support for DeepSeek-V4 inference | 2026-05-08 | kernel-fusion | attention, fp4, gemm |
| [#24582](../sources/prs/sglang/PR-24582.md) | [NPU] Enhance accuracy for model Step3_5 from 0 to 88% | 2026-05-07 |  | attention, quantization |
| [#24587](../sources/prs/sglang/PR-24587.md) | [AMD][aiter] Fix cuda_graph_kv_indices OOB under page_size>1 | 2026-05-07 |  | attention |
| [#24487](../sources/prs/sglang/PR-24487.md) | propagate pytest exit code from test __main__ entries | 2026-05-06 | kernel-fusion | attention, flash-attention, gemm |
| [#24540](../sources/prs/sglang/PR-24540.md) | [NPU] [Bugfix] Wan quantization fix | 2026-05-06 |  | quantization |
| [#24262](../sources/prs/sglang/PR-24262.md) | (3/n - prefill optimize)[LoRA][MoE] Optimize virtual experts: remove CPU-GPU sync & multi-block CUDA JIT histogram | 2026-05-02 |  | moe, prefill |
| [#24129](../sources/prs/sglang/PR-24129.md) | fix(aiter): drop FP8 KV upcast; use native FP8 path in paged_attentio… | 2026-04-30 |  | attention, fp8 |
| [#23975](../sources/prs/sglang/PR-23975.md) | Fix LFM2 ShortConv Mamba State Indexing | 2026-04-29 |  | moe |
| [#24004](../sources/prs/sglang/PR-24004.md) | fix(moe): relocate orphan tuned configs after #23019 | 2026-04-29 | kernel-fusion | fp8, kernel-fusion, moe |
| [#24007](../sources/prs/sglang/PR-24007.md) | (1/n - prefill optimize)feat(lora): enable csgmv backend with virtual experts for MoE LoRA | 2026-04-29 |  | moe, prefill |
| [#24013](../sources/prs/sglang/PR-24013.md) | [VLM] Batch cross-request ViT encoding and reuse attention metadata | 2026-04-29 |  | attention |
| [#24069](../sources/prs/sglang/PR-24069.md) | fix(moe): repair dead import in fused_moe_native after MoE refactor | 2026-04-29 | kernel-fusion | kernel-fusion, moe |
| [#24097](../sources/prs/sglang/PR-24097.md) | Restrict fa_skip_kv_cache to non-MLA backends | 2026-04-29 |  | attention, mla |
| [#23927](../sources/prs/sglang/PR-23927.md) | [AMD] Replace fp8 mla with fp8 mha kernel for diffusion model aiter backend | 2026-04-28 | kernel-fusion | attention, fp8, kernel-fusion |
| [#23795](../sources/prs/sglang/PR-23795.md) | :sparkles: [llm][npu][quant] Add W4A4 MXFP4 quantization support for Qwen3 Dense on Ascend NPU | 2026-04-27 |  | fp4, quantization |
| [#23832](../sources/prs/sglang/PR-23832.md) | amd/deepseek_v4 integration 2/N - cuda graph 0426 | 2026-04-27 |  | attention, mla |
| [#23882](../sources/prs/sglang/PR-23882.md) | Deepseek V4 | 2026-04-27 | kernel-fusion | attention, decode, fp4 |
| [#23745](../sources/prs/sglang/PR-23745.md) | Use Cute-DSL NVFP4 quantization kernels | 2026-04-26 |  | fp4, moe, nvfp4 |
| [#23754](../sources/prs/sglang/PR-23754.md) | [Quantization] add humming quantization kernel | 2026-04-26 | kernel-fusion | fp4, fp8, kernel-fusion |
| [#23760](../sources/prs/sglang/PR-23760.md) | [MoE] Unify DeepEPMoE+MoriEPMoE through AITER MoeRunner pre/post-permute | 2026-04-26 |  | fp4, fp8, moe |
| [#23682](../sources/prs/sglang/PR-23682.md) | Add fused moe triton config for Qwen3.5-397B-A17B-FP8 | 2026-04-25 | kernel-fusion | fp8, kernel-fusion, moe |
| [#23719](../sources/prs/sglang/PR-23719.md) | add H100 configs for GLM-4.7-Flash | 2026-04-25 |  | moe |
| [#23597](../sources/prs/sglang/PR-23597.md) | [MoE] Add Aiter MoE runner backend and purge aiter.fused_moe from quant methods | 2026-04-24 | kernel-fusion | fp4, fp8, kernel-fusion |
| [#23625](../sources/prs/sglang/PR-23625.md) | Flux2 nvfp4 quantization correctness on Blackwell (B200) | 2026-04-24 |  | fp4, nvfp4, quantization |
| [#23650](../sources/prs/sglang/PR-23650.md) | :sparkles: [llm][npu][quant] Add W4A8 MXFP quantization support for Qwen3 Dense on Ascend NPU | 2026-04-24 |  | fp4, quantization |
| [#23545](../sources/prs/sglang/PR-23545.md) | Fix MoE no_combine: skip router weight in down projection | 2026-04-23 | kernel-fusion | kernel-fusion, moe |
| [#23585](../sources/prs/sglang/PR-23585.md) | Move expert_mask_gpu from FusedMoE layer to StandardDispatcher | 2026-04-23 | kernel-fusion | fp4, fp8, kernel-fusion |
| [#23351](../sources/prs/sglang/PR-23351.md) | Support piecewise CUDA graph with NSA | 2026-04-21 |  | attention, fp4 |
| [#23255](../sources/prs/sglang/PR-23255.md) | [MUSA][18/N] Add MUSA-optimized kernel implementations for hot ops | 2026-04-20 | kernel-fusion | gemv, kernel-fusion, moe |
| [#23292](../sources/prs/sglang/PR-23292.md) | [CP] 1/N: Support MLA Prefill Context Parallel | 2026-04-20 |  | attention, fp4, mla |
| [#23019](../sources/prs/sglang/PR-23019.md) | refactor(moe): de-duplicate triton MoE runner path into shared helpers | 2026-04-17 | kernel-fusion | fp8, kernel-fusion, moe |
| [#23065](../sources/prs/sglang/PR-23065.md) | sglang miles lora | 2026-04-17 | kernel-fusion | fp8, gemm, kernel-fusion |
| [#22918](../sources/prs/sglang/PR-22918.md) | [FlashInfer v0.6.11] [RL] Support FlashInfer per-token NVFP4 MoE | 2026-04-16 |  | fp4, moe, nvfp4 |
| [#22952](../sources/prs/sglang/PR-22952.md) | [AMD] Add SGLANG_MORI_MOE_MAX_INPUT_TOKENS to truncate dispatch before MoE. | 2026-04-16 |  | moe |
| [#22844](../sources/prs/sglang/PR-22844.md) | [AMD] Optimize _append_shared_to_topk_output by a single fused Triton kernel for Qwen3.5 | 2026-04-15 | kernel-fusion | kernel-fusion, moe |
| [#22868](../sources/prs/sglang/PR-22868.md) | [Apple Silicon] Add custom Metal RoPE kernel with fused KV cache store | 2026-04-15 | kernel-fusion | attention, kernel-fusion |
| [#22773](../sources/prs/sglang/PR-22773.md) | [Step3p5] Optimize allreduce in MoE layers  | 2026-04-14 |  | moe |
| [#22774](../sources/prs/sglang/PR-22774.md) | [MUSA][16/N] Add MUSA backend support for layers and DeepSeek models (V2/V3/R1) | 2026-04-14 | kernel-fusion | attention, fp8, gemm |
| [#22791](../sources/prs/sglang/PR-22791.md) | [MoE] Add LFM2 MoE tuning support + tuned configs for H100/B200/MI325X | 2026-04-14 | kernel-fusion | kernel-fusion, moe |
| [#22820](../sources/prs/sglang/PR-22820.md) | Cleanup server_args.py and minor code tidying | 2026-04-14 | kernel-fusion | kernel-fusion, moe |
| [#22653](../sources/prs/sglang/PR-22653.md) | [Docker] Remove flashinfer cache copy | 2026-04-13 |  | gemm |
| [#22660](../sources/prs/sglang/PR-22660.md) | Skip redundant moe_sum_reduce for single-expert routing on XPU | 2026-04-13 | kernel-fusion | kernel-fusion, moe |
| [#22672](../sources/prs/sglang/PR-22672.md) | reland [Diffusion] Add FLUX.1-dev ModelOpt NVFP4 support | 2026-04-13 | kernel-fusion, pipeline-stages | fp4, fp8, kernel-fusion |
| [#22681](../sources/prs/sglang/PR-22681.md) | [Diffusion] Add Wan2.2 ModelOpt NVFP4 support | 2026-04-13 | kernel-fusion | fp4, kernel-fusion, nvfp4 |
| [#22685](../sources/prs/sglang/PR-22685.md) | [CPU] [Quantization] Add GPTQ/AWQ 4bits quantization support for CPU  | 2026-04-13 |  | gemm, moe, quantization |
| [#22688](../sources/prs/sglang/PR-22688.md) | Fix trtllm mla chunked-prefill zero-length bug (#22291) | 2026-04-13 |  | attention, mla, prefill |
| [#22723](../sources/prs/sglang/PR-22723.md) | [Fix] Fix accuracy bug in Flashmla sparse MLA kernel | 2026-04-13 |  | mla |
| [#22627](../sources/prs/sglang/PR-22627.md) | [Bugfix] Fix flashinfer_cutlass MoE crash when intermediate_size_per_partition is not 16-aligned | 2026-04-12 |  | moe, quantization |
| [#22574](../sources/prs/sglang/PR-22574.md) | [Diffusion] Add FLUX.1-dev ModelOpt NVFP4 support | 2026-04-11 | kernel-fusion, pipeline-stages | fp4, fp8, kernel-fusion |
| [#22594](../sources/prs/sglang/PR-22594.md) | diffusion: fix layerwise offload for ModelOpt quantized DiTs | 2026-04-11 | kernel-fusion | kernel-fusion, quantization |
| [#22598](../sources/prs/sglang/PR-22598.md) | [AMD] fix tbo runtime error when initializing metadata for cuda graph | 2026-04-11 |  | attention |
| [#22484](../sources/prs/sglang/PR-22484.md) | [RL] Fix weight update for mxfp8 flashinfer_cutlass gemm backend | 2026-04-10 |  | fp8, gemm, quantization |
| [#22491](../sources/prs/sglang/PR-22491.md) | [CI/Docker] Clean up redundant flashinfer cubin downloads | 2026-04-10 |  | gemm |
| [#22543](../sources/prs/sglang/PR-22543.md) | GLM-5/5.1 MXFP4 Checkpoint Inference Compatibility Fix | 2026-04-10 |  | fp4 |
| [#22424](../sources/prs/sglang/PR-22424.md) | [AMD] Use aiter CK layernorm2d for LayerNorm to reduce NSA indexer kernel launches | 2026-04-09 |  | attention |
| [#22430](../sources/prs/sglang/PR-22430.md) | [Fix] Fix several bugs on DSA models | 2026-04-09 |  | attention |
| [#22440](../sources/prs/sglang/PR-22440.md) | Upgrade sglang-torch-profiler-analysis SKILLS | 2026-04-09 |  | gemm |
| [#22306](../sources/prs/sglang/PR-22306.md) | Lazy import flash_attention_v4 to avoid loading flash_attn.cute at startup | 2026-04-08 |  | attention, flash-attention |
| [#22314](../sources/prs/sglang/PR-22314.md) | [AMD] Fix GLM-5 fp8 KV quant path dispatch on MI300 | 2026-04-08 |  | fp8 |
| [#22316](../sources/prs/sglang/PR-22316.md) | [Reland] DeepSeek-R1-0528-w4a8: DeepEP Low Latency Dispatch Adopts FP8 Communication | 2026-04-08 |  | fp8, moe, quantization |
| [#22322](../sources/prs/sglang/PR-22322.md) | [Docker] Fix Trivy CVEs, cubin download 403s, and kernels command order | 2026-04-08 |  | gemm |
| [#22323](../sources/prs/sglang/PR-22323.md) | [Lora] Lora quat info re-factor and support deepseekv3 mla lora | 2026-04-08 |  | fp8, mla, moe |
| [#22336](../sources/prs/sglang/PR-22336.md) | [AMD] Add GLM-5.1-FP8 nightly accuracy and performance benchmarks for MI30x and MI35x | 2026-04-08 |  | fp8 |
| [#22338](../sources/prs/sglang/PR-22338.md) | :sparkles: [diffusion][npu][quant] Add MXFP4 quantization support for Wan2.2 Diffusion on Ascend NPU | 2026-04-08 | kernel-fusion | fp4, kernel-fusion, quantization |
| [#22352](../sources/prs/sglang/PR-22352.md) | :sparkles: [llm][npu][quant] Add W8A8 MXFP8 quantization support for Qwen3 Dense on Ascend NPU | 2026-04-08 |  | fp8, quantization |
| [#22365](../sources/prs/sglang/PR-22365.md) | [Diffusion] modelopt diffusion fp8 support for flux1/flux2 and wan2.2 | 2026-04-08 | kernel-fusion | fp8, kernel-fusion, quantization |
| [#22372](../sources/prs/sglang/PR-22372.md) | [DSA] Hopper FP8 FlashMLA KV padding | 2026-04-08 |  | attention, fp8, mla |
| [#22381](../sources/prs/sglang/PR-22381.md) | [Lora] Lora kimi support | 2026-04-08 |  | moe, quantization |
| [#22232](../sources/prs/sglang/PR-22232.md) | Reduce unnecessary kernels and copies in the NSA indexer | 2026-04-07 |  | attention |
| [#22258](../sources/prs/sglang/PR-22258.md) | [AMD][HIP] NSA: bf16 passthrough from RMSNorm to eliminate FP8 dequantization | 2026-04-07 |  | attention, fp8, quantization |
| [#22187](../sources/prs/sglang/PR-22187.md) | [HiSparse]: Add benchmark for hisparse kernel | 2026-04-06 |  | gemm |
| [#22204](../sources/prs/sglang/PR-22204.md) | [RL] Refactor NVFP4 shuffling/swizzling to in-place replacement | 2026-04-06 | swizzling | fp4, moe, nvfp4 |
| [#22127](../sources/prs/sglang/PR-22127.md) | [Diffusion] Add diffusion NVFP4 scaled-mm correctness test | 2026-04-05 | kernel-fusion | fp4, kernel-fusion, nvfp4 |
| [#22134](../sources/prs/sglang/PR-22134.md) | [Hotfix] Fix router gemm on sm103 | 2026-04-05 |  | gemm |
| [#22145](../sources/prs/sglang/PR-22145.md) | [Disagg][NIXL] Fix heterogeneous TP KV transfer for non-MLA models (same logic with mooncake, Step 1/2 for Qwen3.5 support) | 2026-04-05 |  | mla |
| [#22155](../sources/prs/sglang/PR-22155.md) | [hisparse]: Adding ci for hisparse kvcache-swap-in jit-kernel | 2026-04-05 |  | gemm |
| [#22091](../sources/prs/sglang/PR-22091.md) | [diffusion] Default NVFP4 to CUTLASS and add all-model shape benchmarks | 2026-04-04 | kernel-fusion | fp4, kernel-fusion, nvfp4 |
| [#22122](../sources/prs/sglang/PR-22122.md) | [lora][moe] Virtual experts for LoRA MoE | 2026-04-04 | kernel-fusion | kernel-fusion, moe |
| [#21987](../sources/prs/sglang/PR-21987.md) | [Bugfix] Fix CUDA graph replay issues in trtllm_mla draft_extend | 2026-04-03 |  | attention, mla |
| [#22006](../sources/prs/sglang/PR-22006.md) | Tiny fix trtllm_fp8_per_tensor_scale_moe_wrapper router_logits dtype | 2026-04-03 |  | fp8, moe |
| [#22024](../sources/prs/sglang/PR-22024.md) | [NPU] enable mla prepare fused kernel only when being mla attn | 2026-04-03 | kernel-fusion | attention, kernel-fusion, mla |
| [#22051](../sources/prs/sglang/PR-22051.md) | [MUSA][9/N] Add FA3 attention backend support through MATE (MUSA AI Tensor Engine) | 2026-04-03 |  | attention |
| [#22064](../sources/prs/sglang/PR-22064.md) | [Diffusion] Fix weight scale swizzle and add large-M kernel config for FLUX.2-dev-NVFP4 | 2026-04-03 | kernel-fusion, swizzling | fp4, gemm, kernel-fusion |
| [#22079](../sources/prs/sglang/PR-22079.md) | [nvidia] Gemma4 nvfp4 fix | 2026-04-03 |  | attention, fp4, gemm |
| [#21888](../sources/prs/sglang/PR-21888.md) | fix pcg torch dynamo recompile in mxfp8 Triton path | 2026-04-02 |  | fp8, quantization |
| [#21896](../sources/prs/sglang/PR-21896.md) | fix(ci): update est_time for 57 tests based on runtime analysis | 2026-04-02 | kernel-fusion | attention, decode, fp8 |
| [#21906](../sources/prs/sglang/PR-21906.md) | [Bugfix] Temporarily skip TRTLLM attention on (G)B300 (SM103) to avoid high-concurrency hang | 2026-04-02 |  | attention |
| [#21914](../sources/prs/sglang/PR-21914.md) | [DSA] Set trtllm kernels as default for Blackwell | 2026-04-02 |  | gemm |
| [#21971](../sources/prs/sglang/PR-21971.md) | perf: skip KV cache in FA backend for embedding mode | 2026-04-02 |  | attention |
| [#21834](../sources/prs/sglang/PR-21834.md) | [Feature] JIT rmsnorm update (with claude) | 2026-04-01 | kernel-fusion | kernel-fusion |
| [#21861](../sources/prs/sglang/PR-21861.md) |   [GDN] Remove FlashInfer GDN decode + no_buffer guard and default to FlashInfer on SM100+   | 2026-04-01 |  | decode |
| [#21863](../sources/prs/sglang/PR-21863.md) | [server] Add --quantization unquant to explicitly opt out of quantization | 2026-04-01 |  | quantization |
| [#21881](../sources/prs/sglang/PR-21881.md) | [Misc] [MXFP8] Drop sm100 mxfp8 warning | 2026-04-01 |  | fp8 |
| [#21776](../sources/prs/sglang/PR-21776.md) | Harden FlashInfer FP4 imports in standard dispatcher | 2026-03-31 |  | fp4, moe |
| [#21780](../sources/prs/sglang/PR-21780.md) | [Fix] Fall back to triton MOE for GPT-OSS on Blackwell with driver >= 595 | 2026-03-31 |  | moe |
| [#21783](../sources/prs/sglang/PR-21783.md) | [DSA] Support trtllm sparse mla kernel for prefill batches  | 2026-03-31 |  | attention, mla, prefill |
| [#21787](../sources/prs/sglang/PR-21787.md) | Remove redundant test_moe_eval_accuracy_large | 2026-03-31 |  | moe |
| [#21647](../sources/prs/sglang/PR-21647.md) | [5/n] Lora support cuda graph | 2026-03-30 | kernel-fusion | kernel-fusion, moe |
| [#21649](../sources/prs/sglang/PR-21649.md) | fix: TRT-LLM MHA CUDA illegal address with EAGLE v2 + DP attention | 2026-03-30 |  | attention |
| [#21657](../sources/prs/sglang/PR-21657.md) | [AMD] Use tgemm.mm for MoEGate router gemm in deepseek_v2.py | 2026-03-30 |  | gemm, moe |
| [#21710](../sources/prs/sglang/PR-21710.md) | [AMD] Add GLM-5-FP8 nightly performance benchmarks for MI30x and MI35x | 2026-03-30 |  | fp8 |
| [#21711](../sources/prs/sglang/PR-21711.md) | Remove flashinfer wheel cache cleanup that deletes other versions | 2026-03-30 |  | gemm |
| [#21576](../sources/prs/sglang/PR-21576.md) | [FlashInver v0.6.7] Integrate flashinfer_trtllm mxfp8 gemm | 2026-03-28 |  | fp8, gemm, quantization |
| [#21595](../sources/prs/sglang/PR-21595.md) | Change default mm-attention backend from triton_attn to fa4 | 2026-03-28 |  | attention |
| [#21601](../sources/prs/sglang/PR-21601.md) | [Feature] Add FP4 KV Cache Design and support SM120 GPUs | 2026-03-28 |  | attention, fp4, nvfp4 |
| [#21511](../sources/prs/sglang/PR-21511.md) | [AMD] Enable FP8 KV cache and FP8 attention kernel for NSA on MI300/MI355 with TileLang backend | 2026-03-27 |  | attention, fp8, mla |
| [#21531](../sources/prs/sglang/PR-21531.md) | [JIT Kernel] Migrate dsv3_router_gemm from AOT sgl-kernel to JIT kernel | 2026-03-27 |  | gemm |
| [#21561](../sources/prs/sglang/PR-21561.md) | test: point DSV3 int8 MLA CI models to lmsys Hugging Face org | 2026-03-27 |  | mla |
| [#21436](../sources/prs/sglang/PR-21436.md) | fix nemotron capture for non attention layers | 2026-03-26 |  | attention |
| [#21446](../sources/prs/sglang/PR-21446.md) | Add explicit disable flag for FlashInfer allreduce fusion | 2026-03-26 | kernel-fusion | kernel-fusion |
| [#21452](../sources/prs/sglang/PR-21452.md) | fix: piecewise_cuda_graph get correct qo_indptr | 2026-03-26 |  | attention |
| [#21463](../sources/prs/sglang/PR-21463.md) | Migrate all callers from /get_server_info to /server_info | 2026-03-26 |  | attention, fp4, fp8 |
| [#21411](../sources/prs/sglang/PR-21411.md) | [GDN] Fuse GDN kkt + solve_tril into one kernel | 2026-03-25 | kernel-fusion | attention, kernel-fusion |
| [#21421](../sources/prs/sglang/PR-21421.md) | [AMD]Integrate aiter's fused_topk for softmax scoring in topk function | 2026-03-25 | kernel-fusion | kernel-fusion, moe, tma |
| [#21428](../sources/prs/sglang/PR-21428.md) | [Bugfix] Lazy-import CuteDSL KDA kernel to fix AMD/ROCm startup crash | 2026-03-25 |  | attention |
| [#21431](../sources/prs/sglang/PR-21431.md) | [Diffusion] [AMD] Online MXFP4 and FP8 Quantization for Multimodal Generation | 2026-03-25 | kernel-fusion | attention, flash-attention, fp4 |
| [#21278](../sources/prs/sglang/PR-21278.md) | P2P Weight Update features for miles  | 2026-03-24 |  | moe |
| [#21280](../sources/prs/sglang/PR-21280.md) | [RL] Support mxfp8 DeepSeek V3 | 2026-03-24 | kernel-fusion | fp8, kernel-fusion, moe |
| [#21296](../sources/prs/sglang/PR-21296.md) | [MUSA] apply_vocab_mask support musa device | 2026-03-24 |  | gemm |
| [#21314](../sources/prs/sglang/PR-21314.md) | CUTLASS NVFP4 GEMM improvement of SM120 | 2026-03-24 |  | fp4, gemm, nvfp4 |
| [#21321](../sources/prs/sglang/PR-21321.md) | [Kernel] Support FlashInfer TRTLLM-Gen fused MoE for non-gated FP4 & FP8 (Nemotron) | 2026-03-24 | kernel-fusion | fp4, fp8, kernel-fusion |
| [#21325](../sources/prs/sglang/PR-21325.md) | [misc] clean up kernel API | 2026-03-24 | kernel-fusion | attention, flash-attention, fp4 |
| [#21339](../sources/prs/sglang/PR-21339.md) | Add dedicated FlashInferCuteDslMoE layer for standard-path FP4 MoE | 2026-03-24 |  | fp4, moe, quantization |
| [#21166](../sources/prs/sglang/PR-21166.md) | [Not-Merge][AMD] GLM-5 performance optimization | 2026-03-23 |  | attention |
| [#21190](../sources/prs/sglang/PR-21190.md) | [Whisper] Enable CUDA graph support and timestamp for whisper model | 2026-03-23 |  | attention |
| [#21200](../sources/prs/sglang/PR-21200.md) | [NPU] bugfix for import sgl-kernel error | 2026-03-23 |  | gemm |
| [#21203](../sources/prs/sglang/PR-21203.md) | [KDA] Support CuTeDSL KDA decode kernel | 2026-03-23 |  | attention, decode |
| [#21213](../sources/prs/sglang/PR-21213.md) | [AMD]: Support MLA with nhead<16 and FP8 KV cache for TP=8 (Kimi K2.5… | 2026-03-23 |  | attention, fp4, fp8 |
| [#21219](../sources/prs/sglang/PR-21219.md) | Split pr-test.yml: extract sgl-kernel, jit-kernel, and multimodal-gen tests into separate workflow files | 2026-03-23 |  | gemm |
| [#21233](../sources/prs/sglang/PR-21233.md) | [refactor] Clean up duplicate flashinfer trtllm moe code | 2026-03-23 | kernel-fusion | kernel-fusion, moe, quantization |
| [#21239](../sources/prs/sglang/PR-21239.md) | Refactor JIT kernel CI to use run_suite.py registration system | 2026-03-23 | kernel-fusion | attention, flash-attention, fp4 |
| [#21240](../sources/prs/sglang/PR-21240.md) | [NVIDIA] Enable FP4 flashinfer trtllm routed moe | 2026-03-23 |  | fp4, moe, quantization |
| [#21118](../sources/prs/sglang/PR-21118.md) | ci: remove IS_BLACKWELL env var; auto-detect Blackwell | 2026-03-22 |  | gemm |
| [#21097](../sources/prs/sglang/PR-21097.md) | [AMD] Add MoE weights and scales padding | 2026-03-21 | kernel-fusion | fp4, fp8, kernel-fusion |
| [#20957](../sources/prs/sglang/PR-20957.md) | [Tiny Fix] Fix IS_BLACKWELL env var empty string warning in rerun-ut workflow | 2026-03-20 |  | gemm |
| [#20988](../sources/prs/sglang/PR-20988.md) | ci: run Stage A CUDA tests as stage-a-test-small-1-gpu on 5090 | 2026-03-20 | kernel-fusion | attention, fp8, kernel-fusion |
| [#21019](../sources/prs/sglang/PR-21019.md) | [Qwen3.5] Fuse split/reshape/cat ops in GDN projection with Triton kernel | 2026-03-20 | kernel-fusion | kernel-fusion |
| [#21022](../sources/prs/sglang/PR-21022.md) | [Chore] Clean up JIT compilation flags | 2026-03-20 |  | fp4, nvfp4 |
| [#21035](../sources/prs/sglang/PR-21035.md) | fix: wrap _import_static_state in inference_mode to fix resume on Blackwell | 2026-03-20 |  | gemm |
| [#20910](../sources/prs/sglang/PR-20910.md) | Add SGLang CUDA crash API logging inspired by FlashInfer | 2026-03-19 | kernel-fusion | attention, flash-attention, fp4 |
| [#20922](../sources/prs/sglang/PR-20922.md) | :sparkles: [diffusion][npu][quant] Add MXFP8 quantization support for Wan2.2 Diffusion on Ascend NPU | 2026-03-19 | kernel-fusion | fp8, kernel-fusion, quantization |
| [#20868](../sources/prs/sglang/PR-20868.md) | fix: guard configure_deep_gemm_num_sms when JIT disabled | 2026-03-18 |  | gemm |
| [#20874](../sources/prs/sglang/PR-20874.md) | [JIT Kernel] Fix NVFP4 multi-arch compilation failure | 2026-03-18 |  | fp4, nvfp4 |
| [#20887](../sources/prs/sglang/PR-20887.md) | CUTLASS FP8 Blockwise GEMM improvement of SM120 | 2026-03-18 |  | fp8, gemm |
| [#20755](../sources/prs/sglang/PR-20755.md) | Use FlashInfer tinygemm for GPT-OSS MoE router on SM90+ | 2026-03-17 |  | gemm, moe |
| [#20699](../sources/prs/sglang/PR-20699.md) | [Diffusion] Fix compile graph broken by flashinfer rope | 2026-03-16 | kernel-fusion | kernel-fusion |
| [#20708](../sources/prs/sglang/PR-20708.md) | Add Mistral Small 4 (Pixtral) support | 2026-03-16 |  | moe |
| [#20606](../sources/prs/sglang/PR-20606.md) | FIX: (NSA) Compute topk_indices_offset when NSA prefill flashmla_sparse is used with FP8 KV cache | 2026-03-15 |  | attention, fp8, mla |
| [#20619](../sources/prs/sglang/PR-20619.md) | fix(docs): correct quantization documentation (#20301) | 2026-03-15 |  | quantization |
| [#20632](../sources/prs/sglang/PR-20632.md) | [Diffusion] Add a benchmark for rmsnorm/fuse_add_rmsnorm | 2026-03-15 | kernel-fusion | kernel-fusion |
| [#20604](../sources/prs/sglang/PR-20604.md) | Use Flashinfer for target_verify in GDN model for SM120 | 2026-03-14 |  | attention |
| [#20479](../sources/prs/sglang/PR-20479.md) | Support Triton MLA FP8 KV cache | 2026-03-13 |  | attention, decode, fp8 |
| [#20501](../sources/prs/sglang/PR-20501.md) | [Kernel] Fuse temperature + softmax in sampling for decode speedup | 2026-03-13 | kernel-fusion | decode, kernel-fusion, tma |
| [#20394](../sources/prs/sglang/PR-20394.md) | [NVIDIA] Enable fp8 flashinfer_trtllm_routed MoE for MiniMax-M2.5 | 2026-03-12 |  | fp8, moe, quantization |
| [#20399](../sources/prs/sglang/PR-20399.md) | [AMD][Bug-fix] Fix gpu fault when run the test with dp-attention-enabled and max-concurrency is over 256 | 2026-03-12 |  | attention |
| [#20407](../sources/prs/sglang/PR-20407.md) | [Model] Support Nemotron 3 Super NVFP4 | 2026-03-12 |  | fp4, nvfp4, quantization |
| [#20409](../sources/prs/sglang/PR-20409.md) | [AMD][AITER] Guard _use_mla_ps_kernel with self.use_mla in draft_extend_v2 paths | 2026-03-12 |  | attention, mla |
| [#20412](../sources/prs/sglang/PR-20412.md) | Fix global server args not set error in test_triton_moe_wna16.py | 2026-03-12 |  | moe |
| [#20428](../sources/prs/sglang/PR-20428.md) | [GDN] Add benchmark for sglang gdn prefill | 2026-03-12 |  | attention, prefill |
| [#20453](../sources/prs/sglang/PR-20453.md) | [AMD][MORI] Fix MTP crash with FP4/FP8 dispatch and add NEXTN dispatch env vars. | 2026-03-12 | kernel-fusion | fp4, fp8, kernel-fusion |
| [#20380](../sources/prs/sglang/PR-20380.md) | fix ci by removing nvidia-cutlass-dsl-libs-base and force reinstall n… | 2026-03-11 |  | gemm |
| [#20384](../sources/prs/sglang/PR-20384.md) | [Fix] Add fallback for flashinfer allreduce fusion | 2026-03-11 | kernel-fusion | kernel-fusion |
| [#20232](../sources/prs/sglang/PR-20232.md) | [fix] qwen3.5 fuse_moe_triton_tune bug | 2026-03-10 | kernel-fusion | kernel-fusion, moe |
| [#20245](../sources/prs/sglang/PR-20245.md) | [NPU] add new fusion operator DispatchFFNCombine | 2026-03-10 | kernel-fusion | kernel-fusion, moe |
| [#20268](../sources/prs/sglang/PR-20268.md) | [4/n jit_kernel restruct] speed up CI tests and add benchmark workflow | 2026-03-10 | kernel-fusion | fp4, kernel-fusion, moe |
| [#20294](../sources/prs/sglang/PR-20294.md) | [AMD] Add 4-GPU test suite for MI325 runners | 2026-03-10 |  | attention |
| [#20302](../sources/prs/sglang/PR-20302.md) | Replace topk_ids with curr_topk_ids in fused_moe.py | 2026-03-10 | kernel-fusion | kernel-fusion, moe |
| [#20305](../sources/prs/sglang/PR-20305.md) | [Benchmark] use flashinfer bench_gpu_time instead of triton do_bench | 2026-03-10 | kernel-fusion | attention, fp4, fp8 |
| [#20187](../sources/prs/sglang/PR-20187.md) | [AMD] Fp8 prefill integration with radix cache path for dpsk models | 2026-03-09 |  | attention, fp8, prefill |
| [#20119](../sources/prs/sglang/PR-20119.md) | [PCG]add piecewise cuda graph support for marlin linear | 2026-03-08 |  | fp8, quantization |
| [#20137](../sources/prs/sglang/PR-20137.md) | [diffusion] Support nvfp4 for Flux.2 | 2026-03-08 | kernel-fusion, pipeline-stages | fp4, kernel-fusion, nvfp4 |
| [#20067](../sources/prs/sglang/PR-20067.md) | MiniMax-M2.5 - Support dp attention, dp reduce scatter, FP4 all gather, AR fusion in prepare_attn | 2026-03-07 | kernel-fusion | attention, fp4, kernel-fusion |
| [#20070](../sources/prs/sglang/PR-20070.md) | Fix streaming session with paged KV cache (SWA/MLA) | 2026-03-07 |  | mla |
| [#20082](../sources/prs/sglang/PR-20082.md) | Enable modelopt quantized FLUX deployment | 2026-03-07 |  | fp8, quantization |
| [#20086](../sources/prs/sglang/PR-20086.md) | [V32/GLM5] Change default setting of V32 nvfp4 on TP4 | 2026-03-07 |  | fp4, nvfp4 |
| [#20089](../sources/prs/sglang/PR-20089.md) | feat: [1/2] [DeepEP] Fuse shared expert into MoE dispatch under EP | 2026-03-07 | kernel-fusion | kernel-fusion, moe |
| [#20094](../sources/prs/sglang/PR-20094.md) | [diffusion] fix bug of copy_if | 2026-03-07 | kernel-fusion | kernel-fusion |
| [#20012](../sources/prs/sglang/PR-20012.md) | [JIT Kernel] Reland NVFP4 kernels to JIT | 2026-03-06 |  | fp4, gemm, moe |
| [#20039](../sources/prs/sglang/PR-20039.md) | [Bugfix] Work around FlashInfer unified transport issue on GB | 2026-03-06 | kernel-fusion | kernel-fusion |
| [#20040](../sources/prs/sglang/PR-20040.md) | Fix SM120 `triton_kernels` MXFP4 `block_k` for GPT-OSS | 2026-03-06 |  | fp4, quantization |
| [#19928](../sources/prs/sglang/PR-19928.md) | [AMD] Fix Tensor Memory Aliasing  | 2026-03-05 |  | attention |
| [#19935](../sources/prs/sglang/PR-19935.md) | [AMD] Fix FP8 assertion failure in aiter MLA decode by falling back to self.k_scale | 2026-03-05 |  | attention, decode, fp4 |
| [#19945](../sources/prs/sglang/PR-19945.md) | [AMD] Tilelang sparse fwd for dsv32 mi355/mi300 | 2026-03-05 |  | attention |
| [#19889](../sources/prs/sglang/PR-19889.md) | Use TRTLLM allreduce fusion for Qwen 3.5 | 2026-03-04 | kernel-fusion | kernel-fusion, moe |
| [#19902](../sources/prs/sglang/PR-19902.md) | Fix MLA decode path returning unwritten (padded) rows | 2026-03-04 |  | attention, decode, mla |
| [#19725](../sources/prs/sglang/PR-19725.md) | [SGLang-Diffusion] Fix custom op fake impl missing eps default for torch.compile | 2026-03-03 | kernel-fusion | kernel-fusion |
| [#19794](../sources/prs/sglang/PR-19794.md) | Add compile-time 256-bit vector guard for pre-Blackwell | 2026-03-03 | kernel-fusion | kernel-fusion |
| [#19652](../sources/prs/sglang/PR-19652.md) | [Feature] NVFP4 Marlin fallback for non-Blackwell GPUs (SM75+) | 2026-03-02 | kernel-fusion | fp4, gemm, kernel-fusion |
| [#19672](../sources/prs/sglang/PR-19672.md) | support fused_moe_triton and moe_sum_all_reduce kernel fusion[reduce … | 2026-03-02 | kernel-fusion | kernel-fusion, moe |
| [#19711](../sources/prs/sglang/PR-19711.md) | [LoRA][II] Add fused MOE LoRA Triton kernel and tests | 2026-03-02 | kernel-fusion | kernel-fusion, moe |
| [#19718](../sources/prs/sglang/PR-19718.md) | Support `triton_kernels` for GPT-OSS on SM120 | 2026-03-02 |  | fp4, quantization |
| [#19721](../sources/prs/sglang/PR-19721.md) | Various SM120 improvements | 2026-03-02 |  | fp8, quantization |
| [#19634](../sources/prs/sglang/PR-19634.md) | [miles] fix for glm5 | 2026-03-01 |  | attention |
| [#19537](../sources/prs/sglang/PR-19537.md) | [FlashInfer v0.6.4] [RL] Integrate FlashInfer mxfp8 gemm, MoE, and routed MoE | 2026-02-28 | kernel-fusion | fp8, gemm, kernel-fusion |
| [#19544](../sources/prs/sglang/PR-19544.md) | [NPU] bugs fix for Deepseek models | 2026-02-28 |  | attention, mla |
| [#19549](../sources/prs/sglang/PR-19549.md) | [diffusion][llm] macOS support | 2026-02-28 | kernel-fusion, pipeline-stages | kernel-fusion, pipeline-stages |
| [#19402](../sources/prs/sglang/PR-19402.md) | Fix nightly Mistral-Large-3 NVFP4 accuracy threshold | 2026-02-26 |  | fp4, nvfp4 |
| [#19425](../sources/prs/sglang/PR-19425.md) | [AMD] Fix weight load shape mismatch for amd dsr1 0528 mxfp4 | 2026-02-26 |  | fp4, quantization |
| [#19428](../sources/prs/sglang/PR-19428.md) | [Feature] add feature mla_ag_after_qlora for dsv3.2 | 2026-02-26 |  | attention, mla |
| [#19433](../sources/prs/sglang/PR-19433.md) | Fix/nemotron mtp quantaized | 2026-02-26 | kernel-fusion | kernel-fusion, moe, quantization |
| [#19437](../sources/prs/sglang/PR-19437.md) | [Kernel Slimming] Migrate NVFP4 kernels to JIT | 2026-02-26 |  | fp4, gemm, moe |
| [#19228](../sources/prs/sglang/PR-19228.md) | [AMD] optimize Kimi K2.5 fused_moe_triton performance by tuning  | 2026-02-24 | kernel-fusion | kernel-fusion, moe |
| [#19247](../sources/prs/sglang/PR-19247.md) | [AMD] Fix accuracy while using --enable-dp-attention | 2026-02-24 |  | attention |
| [#19174](../sources/prs/sglang/PR-19174.md) | Adjust padding size to improve triton_kernels moe performance | 2026-02-23 | kernel-fusion | fp4, kernel-fusion, moe |
| [#19143](../sources/prs/sglang/PR-19143.md) | feat: Support MXFP4 quantized dense models on AMD CDNA2/CDNA3 GPUs | 2026-02-22 |  | fp4, nvfp4, quantization |
| [#19150](../sources/prs/sglang/PR-19150.md) | [NVIDIA] Integrate FlashInfer decode kernel (Blackwell) for Qwen3.5 | 2026-02-22 |  | attention, decode |
| [#19059](../sources/prs/sglang/PR-19059.md) | [jit_kernel] Add fused_qknorm_rope JIT kernel | 2026-02-20 | kernel-fusion | kernel-fusion, moe |
| [#19089](../sources/prs/sglang/PR-19089.md) | Support skip-softmax attention | 2026-02-20 |  | attention, mla, tma |
| [#18985](../sources/prs/sglang/PR-18985.md) | Use single mma warp group for short q_len in FA to optimize decoding performance | 2026-02-18 |  | gemm |
| [#18931](../sources/prs/sglang/PR-18931.md) | Fix NSA FP8 KV cache path for both-trtllm MHA one-shot | 2026-02-17 |  | attention, fp8 |
| [#18937](../sources/prs/sglang/PR-18937.md) | [Qwen3.5] Enable nvfp4 checkpoint | 2026-02-17 |  | fp4, nvfp4 |
| [#18938](../sources/prs/sglang/PR-18938.md) | [Sarvam] Add inference support for Sarvam MoE LLMs | 2026-02-17 |  | moe |
| [#18940](../sources/prs/sglang/PR-18940.md) | Fix benchmark_sglang_fused_moe_triton.py | 2026-02-17 | kernel-fusion | kernel-fusion, moe |
| [#18876](../sources/prs/sglang/PR-18876.md) | Add DeepSeek3.2 and GlmMoeDsa into moe tune | 2026-02-16 | kernel-fusion | kernel-fusion, moe |
| [#18887](../sources/prs/sglang/PR-18887.md) | Fix CI: add flashinfer --download-cubin to install dependencies | 2026-02-16 |  | gemm |
| [#18902](../sources/prs/sglang/PR-18902.md) | [sgl-kernel] rebase FlashMLA 0217 | 2026-02-16 |  | attention, mla |
| [#18851](../sources/prs/sglang/PR-18851.md) | [Perf] Tune MiniMax M2 fused moe kernel on H100 GPU | 2026-02-15 | kernel-fusion | fp8, kernel-fusion, moe |
| [#18854](../sources/prs/sglang/PR-18854.md) | Migrate renorm kernels from sgl-kernel to FlashInfer JIT | 2026-02-15 |  | gemm |
| [#18855](../sources/prs/sglang/PR-18855.md) | Add claude skills for sgl-kernel and jit-kernel | 2026-02-15 |  | gemm |
| [#18858](../sources/prs/sglang/PR-18858.md) | [Perf] ~9.5x faster Blackwell MXFP4 MoE weight loading | 2026-02-15 |  | fp4, moe, quantization |
| [#18871](../sources/prs/sglang/PR-18871.md) | Migrate norm kernels to FlashInfer JIT implementation | 2026-02-15 |  | gemm |
| [#18833](../sources/prs/sglang/PR-18833.md) | perf: add minimax-2.5 fused_moe tuning config for h20 | 2026-02-14 | kernel-fusion | fp8, kernel-fusion, moe |
| [#18770](../sources/prs/sglang/PR-18770.md) | support xverse_moe on npu | 2026-02-13 | kernel-fusion | kernel-fusion, moe, quantization |
| [#18782](../sources/prs/sglang/PR-18782.md) | [VLM][LLM] Optimize fused_moe triton kernel tma | 2026-02-13 | kernel-fusion | kernel-fusion, moe, tma |
| [#18787](../sources/prs/sglang/PR-18787.md) | fix: add SM110 (Jetson AGX Thor) to Blackwell capability check | 2026-02-13 |  | gemm |
| [#18684](../sources/prs/sglang/PR-18684.md) | [AMD] Add MoE weights and scales padding | 2026-02-12 | kernel-fusion | fp4, fp8, kernel-fusion |
| [#18696](../sources/prs/sglang/PR-18696.md) | use flashinfer.sampling | 2026-02-12 |  | gemm |
| [#18742](../sources/prs/sglang/PR-18742.md) | [RL] Support per-layer mixed FP8/BF16 serving for FP8 checkpoints | 2026-02-12 |  | fp8, quantization |
| [#18751](../sources/prs/sglang/PR-18751.md) | fix: update Blackwell log/error messages to include SM12x | 2026-02-12 |  | gemm |
| [#18607](../sources/prs/sglang/PR-18607.md) | [AMD] Fix accuracy issue when running TP4 dsv3 model with mtp | 2026-02-11 |  | attention |
| [#18612](../sources/prs/sglang/PR-18612.md) | [Perf][Kernel] Fuse SiLU+Mul into NVFP4 Expert Quantization for CUTLASS MoE | 2026-02-11 |  | fp4, gemm, moe |
| [#18624](../sources/prs/sglang/PR-18624.md) | [AMD] DSR1/V3 use fp8 bmm in MLA for MI300X | 2026-02-11 |  | fp8, mla |
| [#18528](../sources/prs/sglang/PR-18528.md) | Fp8 prefill attn kernel integration | 2026-02-10 |  | attention, fp8, prefill |
| [#18459](../sources/prs/sglang/PR-18459.md) | fix_get_quant_method_in_fused_moe_condition | 2026-02-09 | kernel-fusion | kernel-fusion, moe, quantization |
| [#18479](../sources/prs/sglang/PR-18479.md) | docs: expand and update modelopt documentation | 2026-02-09 |  | quantization |
| [#18488](../sources/prs/sglang/PR-18488.md) | Tilelang sparse decode fwd for dsv32 mi355 | 2026-02-09 |  | attention, decode |
| [#18496](../sources/prs/sglang/PR-18496.md) | [FIX] Correct JIT kernel compilation on newer GPUs with outdated driver metadata. | 2026-02-09 |  | gemm |
| [#18442](../sources/prs/sglang/PR-18442.md) | feat: add FA4 SM90 paged KV decode support & update attention docs | 2026-02-08 |  | attention, decode, flash-attention |
| [#18389](../sources/prs/sglang/PR-18389.md) | Nsa trtllm mla sparse fp8 support with Deepseek v3.2 NVFP4 | 2026-02-07 |  | attention, fp4, fp8 |
| [#18423](../sources/prs/sglang/PR-18423.md) | [AMD] Update aiter to v0.1.10.post2 | 2026-02-07 |  | attention, decode, fp8 |
| [#18355](../sources/prs/sglang/PR-18355.md) | [AMD] Support Qwen3-Coder-Next on AMD platform | 2026-02-06 |  | attention |
| [#18357](../sources/prs/sglang/PR-18357.md) | [MUSA][10/N] Add GGUF support | 2026-02-06 |  | quantization |
| [#18361](../sources/prs/sglang/PR-18361.md) | feat(gdn): add FlashInfer K-last SSM layout support for GDN prefill and decode for Hopper | 2026-02-06 |  | attention, decode, prefill |
| [#18370](../sources/prs/sglang/PR-18370.md) | [Kimi-K2.5] Fix NVFP4 Kimi-K2.5 weight mapping and exclude list | 2026-02-06 |  | fp4, nvfp4, quantization |
| [#18311](../sources/prs/sglang/PR-18311.md) | [Hicache & JIT_kernel] Support page first layout  & mla jit kernel | 2026-02-05 |  | mla |
| [#18224](../sources/prs/sglang/PR-18224.md) | [ModelOPT] Support Qwen 3 Next Coder NVFP4 | 2026-02-04 |  | fp4, nvfp4 |
| [#18233](../sources/prs/sglang/PR-18233.md) | Support Qwen3 MoE context parallel | 2026-02-04 | kernel-fusion | attention, kernel-fusion, moe |
| [#18242](../sources/prs/sglang/PR-18242.md) | [ROCm] Optimize Deepseek R1 on MI300X | 2026-02-04 |  | attention, fp8, quantization |
| [#18136](../sources/prs/sglang/PR-18136.md) | [Blackwell] Make mxint4 flashinfer_trtllm moe gemm set by default on blackwell | 2026-02-03 |  | gemm, moe |
| [#18139](../sources/prs/sglang/PR-18139.md) | Add Intel Quantization Support in SGLang | 2026-02-03 |  | quantization |
| [#18172](../sources/prs/sglang/PR-18172.md) | [NPU]Support model Trinity-mini for Npu, accuracy 90% | 2026-02-03 |  | moe |
| [#18182](../sources/prs/sglang/PR-18182.md) | [AMD][Quantization] Online MXFP4 quantization 2/N - FP8 to MXFP4 requantization on AMD GPUs | 2026-02-03 |  | fp4, fp8, moe |
| [#18189](../sources/prs/sglang/PR-18189.md) | [ModelOpt] Fix broken Qwen3-235B-A22B-Instruct-2507-NVFP4 launch | 2026-02-03 |  | fp4, moe, nvfp4 |
| [#18085](../sources/prs/sglang/PR-18085.md) | Fix nvfp4 weight update | 2026-02-02 |  | fp4, moe, nvfp4 |
| [#18065](../sources/prs/sglang/PR-18065.md) | [Bugfix] Fix Mistral Large 3 NVFP4 TRTLLM MoE | 2026-02-01 |  | fp4, moe, nvfp4 |
| [#18070](../sources/prs/sglang/PR-18070.md) | Feat/add fi selective state update kernel call | 2026-02-01 |  | attention |
| [#18073](../sources/prs/sglang/PR-18073.md) | [Diffsuion & JIT_kernel] QKNorm cross heads kernel | 2026-02-01 |  | gemm |
| [#17965](../sources/prs/sglang/PR-17965.md) | [Fix] Triton TP MoE Dpsk V3/Qwen3 Coder with SwapAB | 2026-01-30 | kernel-fusion | fp8, kernel-fusion, moe |
| [#18000](../sources/prs/sglang/PR-18000.md) | Skipped warning on sm100 | 2026-01-30 |  | gemm |
| [#17889](../sources/prs/sglang/PR-17889.md) | [Move sgl-kernel Kernel to JIT] Add JIT concat MLA kernels | 2026-01-28 |  | mla |
| [#17891](../sources/prs/sglang/PR-17891.md) | [Perf] Tune Llama-4-Scout-17B-16E-Instruct fused moe kernel | 2026-01-28 | kernel-fusion | kernel-fusion, moe |
| [#17816](../sources/prs/sglang/PR-17816.md) | fix(quantization): add sgl_kernel fallback for FP4 quantize on Blackwell GPUs | 2026-01-27 |  | fp4, quantization |
| [#17838](../sources/prs/sglang/PR-17838.md) | Feature/support longcat flash lite | 2026-01-27 |  | gemm |
| [#17784](../sources/prs/sglang/PR-17784.md) | Upgrade transformers==5.3.0 | 2026-01-26 | pipeline-stages | gemm, moe, pipeline-stages |
| [#17627](../sources/prs/sglang/PR-17627.md) | [feat] Support nvfp4 quantized model of Qwen3-Next | 2026-01-23 |  | fp4, nvfp4, quantization |
| [#17591](../sources/prs/sglang/PR-17591.md) | [hotfix] Reenable all reduce fusion on sm100 | 2026-01-22 | kernel-fusion | kernel-fusion |
| [#17600](../sources/prs/sglang/PR-17600.md) | Make flashMLA work on: Cu13, B300 | 2026-01-22 |  | mla |
| [#17449](../sources/prs/sglang/PR-17449.md) | Add mxfp8 support for online quantization, Triton dense linear, and CUTLASS MoE | 2026-01-21 |  | fp8, moe, quantization |
| [#17480](../sources/prs/sglang/PR-17480.md) | [NPU] enhance accuracy for model kimi-vl-a3b-instruct | 2026-01-21 |  | attention, mla |
| [#17394](../sources/prs/sglang/PR-17394.md) | [NPU] Remove paged attention & Change fia to default attention | 2026-01-20 |  | attention |
| [#17327](../sources/prs/sglang/PR-17327.md) | Disable mla persistent kernel when not using fp8 kv_cache | 2026-01-19 | persistent-kernel | attention, fp8, mla |
| [#17353](../sources/prs/sglang/PR-17353.md) | Move fa4 from sgl-kernel to jit kernel | 2026-01-19 | pipeline-stages, tile-scheduling | attention, flash-attention, pipeline-stages |
| [#17300](../sources/prs/sglang/PR-17300.md) | [FIX] Always support TP > 4 for FP4 Gemm | 2026-01-18 |  | fp4, gemm, quantization |
| [#17247](../sources/prs/sglang/PR-17247.md) | [New Model] GLM4.7-Flash | 2026-01-17 |  | attention, moe |
| [#17235](../sources/prs/sglang/PR-17235.md) | [GLM 4.7] Add RTX 6000 Pro aka sm120 | 2026-01-16 | kernel-fusion | fp8, kernel-fusion, moe |
| [#17094](../sources/prs/sglang/PR-17094.md) | Optimize GDN decode for Qwen3 Next | 2026-01-15 | kernel-fusion | attention, decode, kernel-fusion |
| [#17111](../sources/prs/sglang/PR-17111.md) | [diffusion] fix: fix using upstream flash_attn on blackwell | 2026-01-15 | kernel-fusion | attention, kernel-fusion |
| [#17115](../sources/prs/sglang/PR-17115.md) | Enable XQA for SM90 and SM120 | 2026-01-15 |  | attention |
| [#17158](../sources/prs/sglang/PR-17158.md) | Inclusion of nvfp4 blockscale in EPLB Rebalance | 2026-01-15 |  | fp4, moe, nvfp4 |
| [#17166](../sources/prs/sglang/PR-17166.md) | [Fix] GLM 4.7 + NVFP4 + MTP | 2026-01-15 |  | fp4, moe, nvfp4 |
| [#17053](../sources/prs/sglang/PR-17053.md) | [MUSA][2/N] sgl-kernel build | 2026-01-14 |  | gemm |
| [#17007](../sources/prs/sglang/PR-17007.md) | [NPU]bugfix: fix for dsv3.2 and dsvl2 | 2026-01-13 |  | attention, mla |
| [#16892](../sources/prs/sglang/PR-16892.md) | Support mxint4 flashinfer_trtllm moe gemm | 2026-01-11 | kernel-fusion | gemm, kernel-fusion, moe |
| [#16791](../sources/prs/sglang/PR-16791.md) | [AMD] Support redundant expert with a2a moe in gfx95x. | 2026-01-09 |  | moe |
| [#16824](../sources/prs/sglang/PR-16824.md) | [Fix] `flashinfer_trtllm` `intermediate_size` assertion with Qwen3 + TP=8 | 2026-01-09 | kernel-fusion | kernel-fusion, moe |
| [#16622](../sources/prs/sglang/PR-16622.md) | Fix FP8 MoE NaN with DeepGEMM on Blackwell | 2026-01-07 |  | fp8, gemm, moe |
| [#16502](../sources/prs/sglang/PR-16502.md) | [Fix]Pin mooncake version to 0.3.7.post2 in grace blackwell | 2026-01-05 |  | gemm |
| [#16382](../sources/prs/sglang/PR-16382.md) | [Fix]Fix FA3 Performance in Diffusion Model  | 2026-01-04 | kernel-fusion | attention, kernel-fusion |
| [#16335](../sources/prs/sglang/PR-16335.md) | [diffusion] Fix RuntimeError in SageAttention3 on Nvidia Blackwell with Qwen-Image | 2026-01-03 | kernel-fusion | attention, kernel-fusion |
| [#16283](../sources/prs/sglang/PR-16283.md) | [Fix] Only add SM90 and SM100 to check for auto-enabling TRT Allreduce Fusion | 2026-01-02 | kernel-fusion | kernel-fusion |
| [#16308](../sources/prs/sglang/PR-16308.md) | Fix sgl-kernel jobs to skip when target_stage is specified | 2026-01-02 |  | gemm |
| [#16207](../sources/prs/sglang/PR-16207.md) | SGLANG_USE_PAGED_ATTENTION: High-performance attention operator | 2025-12-31 |  | attention |
| [#16227](../sources/prs/sglang/PR-16227.md) | [NemotronH] Add latent MoE support | 2025-12-31 | kernel-fusion | kernel-fusion, moe |
| [#16161](../sources/prs/sglang/PR-16161.md) | [Diffusion] Zimage opt with qknorm and flashinfer rope | 2025-12-30 | kernel-fusion | kernel-fusion |
| [#16162](../sources/prs/sglang/PR-16162.md) | [Feature] add aligned_vector type for JIT kernel | 2025-12-30 |  | gemm |
| [#16171](../sources/prs/sglang/PR-16171.md) | [VLM] Adopt jit qk_norm kernel in VLM | 2025-12-30 |  | attention |
| [#16034](../sources/prs/sglang/PR-16034.md) | Support fa4 decoding | 2025-12-29 |  | attention |
| [#16043](../sources/prs/sglang/PR-16043.md) | optimize get_topk_ragged by fusing get k and k_scale triton kernel | 2025-12-29 |  | attention |
| [#16055](../sources/prs/sglang/PR-16055.md) | [Diffusion] Flux support flashinfer rope | 2025-12-29 | kernel-fusion | kernel-fusion |
| [#16076](../sources/prs/sglang/PR-16076.md) | enhance accuracy for model kimi-vl-instruct-a3b | 2025-12-29 |  | attention, mla, moe |
| [#16084](../sources/prs/sglang/PR-16084.md) | fix layer intermediate size | 2025-12-29 | kernel-fusion | kernel-fusion, moe |
| [#15986](../sources/prs/sglang/PR-15986.md) | Tiny fix cannot launch nvfp4 checkpoint with bf16 kv cache | 2025-12-28 |  | fp4, nvfp4 |
| [#16014](../sources/prs/sglang/PR-16014.md) | [Performance] Force split_k=1 for MXFP4 Triton kernels on Hopper | 2025-12-28 |  | fp4, quantization |
| [#15948](../sources/prs/sglang/PR-15948.md) |  Add tuned triton==3.5.1 h200 tp2, tp4 for qwen 3 next | 2025-12-27 | kernel-fusion | kernel-fusion, moe |
| [#15888](../sources/prs/sglang/PR-15888.md) | [diffusion] model: support TurboWan2.1-T2V-1.3B/14B SLA | 2025-12-26 | kernel-fusion, pipeline-stages | attention, kernel-fusion, pipeline-stages |
| [#15891](../sources/prs/sglang/PR-15891.md) | [fix]deepgemm precompile when warmup | 2025-12-26 |  | gemm |
| [#15894](../sources/prs/sglang/PR-15894.md) | Bugfix for ds-vl2 | 2025-12-26 |  | attention, mla |
| [#15904](../sources/prs/sglang/PR-15904.md) | [NPU] NZ for non-quantized MOE, Qwen3 MOE double memory consumption fix | 2025-12-26 | kernel-fusion | kernel-fusion, moe, quantization |
| [#15836](../sources/prs/sglang/PR-15836.md) | [JIT kernel] Apply jit per_tensor_quant_fp8 kernel | 2025-12-25 |  | fp8, gemm, moe |
| [#15712](../sources/prs/sglang/PR-15712.md) | Add SwapAB Optimization for triton fused_moe_kernel on SM90. | 2025-12-24 | kernel-fusion | kernel-fusion, moe |
| [#15731](../sources/prs/sglang/PR-15731.md) | [Perf] Eliminate the slice op for Flashinfer `trtllm_fp4_block_scale_moe` | 2025-12-24 |  | block-scale, fp4, moe |
| [#15753](../sources/prs/sglang/PR-15753.md) | Fix GLM-4.7 MoE Detector complex JSON Schema type parsing | 2025-12-24 |  | moe |
| [#15754](../sources/prs/sglang/PR-15754.md) | Fix: Handle empty func_name and None values in GLM MoE detectors | 2025-12-24 |  | moe |
| [#15601](../sources/prs/sglang/PR-15601.md) | Fix BatchMLAPagedAttentionWrapper query/qo_inptr mismatch for EAGLE | 2025-12-22 |  | attention, mla |
| [#15631](../sources/prs/sglang/PR-15631.md) | [jit-kernel] Add CuTe DSL GDN Decode Kernel | 2025-12-22 |  | attention, decode |
| [#15551](../sources/prs/sglang/PR-15551.md) | Update flashinfer to 0.6.1 | 2025-12-21 | kernel-fusion | fp4, kernel-fusion, moe |
| [#15552](../sources/prs/sglang/PR-15552.md) | [sgl-kernel] Streamline kernel size report (Top 20 only) and clean up | 2025-12-21 |  | gemm |
| [#15514](../sources/prs/sglang/PR-15514.md) | [Perf] Add Flashinfer DeepGEMM SM90 for SwapAB Optimization | 2025-12-20 |  | fp8, gemm, quantization |
| [#15518](../sources/prs/sglang/PR-15518.md) | [NPU] update Mixed chunk op to FIA | 2025-12-20 |  | attention |
| [#15522](../sources/prs/sglang/PR-15522.md) | Optimize FP8 MLA KV cache writes with Triton kernel | 2025-12-20 |  | attention, fp8, mla |
| [#15526](../sources/prs/sglang/PR-15526.md) | Optimize Bailing-MoE with FlashInfer Fused All-Reduce | 2025-12-20 | kernel-fusion | kernel-fusion, moe |
| [#15539](../sources/prs/sglang/PR-15539.md) | MoE: Skip SiLU/GELU activation for masked experts | 2025-12-20 | kernel-fusion | kernel-fusion, moe |
| [#15464](../sources/prs/sglang/PR-15464.md) | Optimize MiMo-V2-Flash by flashinfer fused allreduce | 2025-12-19 | kernel-fusion | kernel-fusion |
| [#15469](../sources/prs/sglang/PR-15469.md) | Fix Illegal Memory Access when fa3 + spec + topk + page_size > 1 | 2025-12-19 |  | attention |
| [#15352](../sources/prs/sglang/PR-15352.md) | [Tiny]Add warning for deepgemm on Blackwell | 2025-12-18 |  | gemm |
| [#15363](../sources/prs/sglang/PR-15363.md) | [NPU]mindspore model support moe | 2025-12-18 |  | moe |
| [#15381](../sources/prs/sglang/PR-15381.md) | [NPU]DeepSeek-V3.2 support npu mlaprolog | 2025-12-18 |  | attention, mla, quantization |
| [#15382](../sources/prs/sglang/PR-15382.md) | [diffusion] Add Sage Attention 3 Support for sm 120 (RTX5090) | 2025-12-18 | kernel-fusion, pipeline-stages | attention, kernel-fusion, pipeline-stages |
| [#15407](../sources/prs/sglang/PR-15407.md) | Super tiny add moe_ep_rank to prometheus labels | 2025-12-18 |  | moe |
| [#15422](../sources/prs/sglang/PR-15422.md) | Flashinfer MOE FP8 support for Mistral Large 3. | 2025-12-18 |  | fp8, moe, quantization |
| [#15303](../sources/prs/sglang/PR-15303.md) | [Fix] A followup fix for TRTLLM BF16 MoE | 2025-12-17 |  | moe |
| [#15304](../sources/prs/sglang/PR-15304.md) | Fix the accuracy issue when running mxfp4 dsv3 model and enable ep | 2025-12-17 |  | fp4, moe, quantization |
| [#15306](../sources/prs/sglang/PR-15306.md) | Fix warp illegal instruction in kimi k2 thinking PCG | 2025-12-17 | kernel-fusion | kernel-fusion, moe |
| [#15325](../sources/prs/sglang/PR-15325.md) | feat: support bitsandbytes quantization algorithm | 2025-12-17 |  | quantization |
| [#15345](../sources/prs/sglang/PR-15345.md) | [distributed] Clean up MoE groups in destroy_model_parallel | 2025-12-17 |  | moe |
| [#15242](../sources/prs/sglang/PR-15242.md) | [sgl-kernel] Update flashmla to include fp8 sparse_mla optimizations | 2025-12-16 |  | fp8, mla |
| [#15280](../sources/prs/sglang/PR-15280.md) | [NVIDIA] Fixes for NVFP4 all-gather with spec decoding | 2025-12-16 |  | fp4, moe, nvfp4 |
| [#15141](../sources/prs/sglang/PR-15141.md) | [sgl-kernel][1/2] Fused qk_norm_rope for GLM4.6 | 2025-12-15 | kernel-fusion | kernel-fusion, moe |
| [#15153](../sources/prs/sglang/PR-15153.md) | Add cache for flashinfer installation | 2025-12-15 |  | gemm |
| [#15182](../sources/prs/sglang/PR-15182.md) | [NVIDIA] upstream FA4 | 2025-12-15 |  | attention, flash-attention |
| [#15127](../sources/prs/sglang/PR-15127.md) | fix(attention): Prevent trtllm_mha auto-selection with eagle3 speculative decoding | 2025-12-14 |  | attention |
| [#15049](../sources/prs/sglang/PR-15049.md) | Mistral Large 3 NVFP4 TRTLLM MoE support | 2025-12-13 | kernel-fusion | fp4, kernel-fusion, moe |
| [#14936](../sources/prs/sglang/PR-14936.md) | Fix accuracy issue when using a16w16 mla_decode_fwd | 2025-12-12 |  | attention, decode, mla |
| [#14975](../sources/prs/sglang/PR-14975.md) | [AMD] Support fused_rms_mxfp4_quant in the prefill stage for DeepSeek-R1-MXFP4 | 2025-12-12 | kernel-fusion | fp4, kernel-fusion, prefill |
| [#14998](../sources/prs/sglang/PR-14998.md) | add transformers version validation for glm-4.6v moe models | 2025-12-12 |  | moe |
| [#14897](../sources/prs/sglang/PR-14897.md) | Fix dsv3 dp accuracy issue when using bf16-kv | 2025-12-11 |  | attention |
| [#14820](../sources/prs/sglang/PR-14820.md) | [NPU][eagle3] support qwen eagle3 on NPU | 2025-12-10 |  | attention |
| [#14829](../sources/prs/sglang/PR-14829.md) | Apply back moe_sum_reduce for fused_marlin_moe | 2025-12-10 | kernel-fusion | kernel-fusion, moe |
| [#14842](../sources/prs/sglang/PR-14842.md) | fix: trtllm mha attention auto-selection on sm120 | 2025-12-10 |  | attention |
| [#14717](../sources/prs/sglang/PR-14717.md) | [diffusion] kernel fusion: gated residual layernorm scale shift and layernorm scale shift kernel fusion for Qwen-Image, WAN and HunyuanVideo | 2025-12-09 | kernel-fusion | kernel-fusion |
| [#14640](../sources/prs/sglang/PR-14640.md) | [sgl-kernel][Feat][B200][2/N] Support MXFP8 Grouped GEMM in Blackwell | 2025-12-08 |  | fp8, gemm |
| [#14544](../sources/prs/sglang/PR-14544.md) | Add CUDA kernel size analysis tool for sgl-kernel optimization | 2025-12-06 |  | gemm |
| [#14466](../sources/prs/sglang/PR-14466.md) | Add Mistral Large 3 Eagle Support | 2025-12-05 |  | attention, fp8, mla |
| [#14485](../sources/prs/sglang/PR-14485.md) | Mistral Large 3 NVFP4 support | 2025-12-05 |  | fp4, moe, nvfp4 |
| [#14385](../sources/prs/sglang/PR-14385.md) | [CPU] Implement MXFP4 Gemm kernels for intel AMX to support GPT OSS series. | 2025-12-04 |  | fp4, fp8, gemm |
| [#14395](../sources/prs/sglang/PR-14395.md) | Support FP8 MLA prefill and 128k context. | 2025-12-04 |  | attention, fp8, mla |
| [#14423](../sources/prs/sglang/PR-14423.md) | [NPU] perf update with kvcache nz & w4a8 quant | 2025-12-04 | kernel-fusion | attention, kernel-fusion, mla |
| [#14335](../sources/prs/sglang/PR-14335.md) | sync attention, deepseek doc | 2025-12-03 |  | attention |
| [#14350](../sources/prs/sglang/PR-14350.md) | [FIX] trtllm-moe-fp4-renorm for Qwen series models | 2025-12-03 | kernel-fusion | fp4, kernel-fusion, moe |
| [#14291](../sources/prs/sglang/PR-14291.md) | Tiny use trtllm_mha as default when possible | 2025-12-02 |  | attention, flash-attention |
| [#14311](../sources/prs/sglang/PR-14311.md) | [Fix] add block size logic for sm120 smem size | 2025-12-02 |  | attention |
| [#14173](../sources/prs/sglang/PR-14173.md) | fix: Increase FlashInfer workspace size for Qwen3VL models | 2025-12-01 |  | attention |
| [#14194](../sources/prs/sglang/PR-14194.md) | [feature] implement dcp for deepseek_v2 | 2025-12-01 |  | attention, mla |
| [#14213](../sources/prs/sglang/PR-14213.md) | Add Mistral Large 3 support. | 2025-12-01 | kernel-fusion | attention, fp8, kernel-fusion |
| [#14224](../sources/prs/sglang/PR-14224.md) | [bug fix] fix ima with get_mla_kv_buffer_kernel overflow | 2025-12-01 |  | mla |
| [#14122](../sources/prs/sglang/PR-14122.md) | Add new moe wna16 marlin gemm | 2025-11-29 | kernel-fusion | gemm, kernel-fusion, moe |
| [#14125](../sources/prs/sglang/PR-14125.md) | Apply new moe wna16 marlin gemm | 2025-11-29 | kernel-fusion | gemm, kernel-fusion, moe |
| [#14133](../sources/prs/sglang/PR-14133.md) | Opt moe align block size kernel | 2025-11-29 |  | moe |
| [#14134](../sources/prs/sglang/PR-14134.md) | Apply new moe align block size kernel | 2025-11-29 | kernel-fusion | kernel-fusion, moe |
| [#14147](../sources/prs/sglang/PR-14147.md) | Support checking fp8 params in weight_checker | 2025-11-29 |  | fp8 |
| [#14105](../sources/prs/sglang/PR-14105.md) | [LoRA][III] Add LoRA support for MoE layers and enable TP | 2025-11-28 | kernel-fusion | kernel-fusion, moe |
| [#14028](../sources/prs/sglang/PR-14028.md) | Fix flashinfer cutlass MoE output shape for non-FP4-packed inputs | 2025-11-27 |  | fp4, moe, quantization |
| [#13959](../sources/prs/sglang/PR-13959.md) | [DeepSeek v3.2] opt Context Parallelism: support fused moe, multi batch and fp8 kvcache | 2025-11-26 | kernel-fusion | attention, fp8, kernel-fusion |
| [#13969](../sources/prs/sglang/PR-13969.md) | [kernel][moe] add moe topk fast | 2025-11-26 |  | moe, tma |
| [#13976](../sources/prs/sglang/PR-13976.md) | Use trtllm mha decode kernel for target_verify in speculative decoding | 2025-11-26 |  | attention, decode |
| [#13983](../sources/prs/sglang/PR-13983.md) | Support  KTransformers for Qwen3-VL moe | 2025-11-26 |  | moe |
| [#13873](../sources/prs/sglang/PR-13873.md) | Feat: GLM-4.6 supports shared experts fusion | 2025-11-25 | kernel-fusion | fp8, kernel-fusion, moe |
| [#13910](../sources/prs/sglang/PR-13910.md) | Fix update weight error for blackwell DeepGEMM | 2025-11-25 |  | fp8, gemm, quantization |
| [#13848](../sources/prs/sglang/PR-13848.md) | update flashinfer_cubin==0.5.3 | 2025-11-24 |  | gemm |
| [#13864](../sources/prs/sglang/PR-13864.md) | [BugFix] fix outplace_fused_experts missing is_gated | 2025-11-24 | kernel-fusion | kernel-fusion, moe |
| [#13794](../sources/prs/sglang/PR-13794.md) | Support fp4 fp8 non gated moe | 2025-11-23 | kernel-fusion | fp4, fp8, kernel-fusion |
| [#13798](../sources/prs/sglang/PR-13798.md) | [NVIDIA] Enable TRTLLM BF16 MoE on Blackwell GPUs | 2025-11-23 | kernel-fusion | kernel-fusion, moe, quantization |
| [#13747](../sources/prs/sglang/PR-13747.md) | [AMD] Support --enable-aiter-allreduce-fusion on AMD GPUs | 2025-11-22 | kernel-fusion | kernel-fusion |
| [#13751](../sources/prs/sglang/PR-13751.md) | [chore]Upgrade flashinfer to 0.5.3 | 2025-11-22 |  | gemm |
| [#13761](../sources/prs/sglang/PR-13761.md) | [Feat][NVFP4] Enable NVFP4 MoE for Qwen series models (eg. Qwen3-Next) #13761 | 2025-11-22 | kernel-fusion | fp4, kernel-fusion, moe |
| [#13715](../sources/prs/sglang/PR-13715.md) | Fix EPLB + FP4 Quantization Compatibility Issue | 2025-11-21 |  | fp4, moe, quantization |
| [#13730](../sources/prs/sglang/PR-13730.md) | [bugfix] fix TBO crashes when attn_tp_size > 1 | 2025-11-21 |  | moe |
| [#13731](../sources/prs/sglang/PR-13731.md) | [sgl-kernel][Feat][B200][1/N]Support MXFP8 Grouped GEMM in Blackwell | 2025-11-21 |  | fp8, gemm, moe |
| [#13738](../sources/prs/sglang/PR-13738.md) | fix trtllm mla spec | 2025-11-21 |  | attention, mla |
| [#13617](../sources/prs/sglang/PR-13617.md) | [ROCM] Optimized deepseek-r1 fp8 model with + triton_gemm_a8w8 + batch_gemm_a8w8 + fused set_mla_kv_buffer kernel | 2025-11-20 | kernel-fusion | fp8, gemm, kernel-fusion |
| [#13646](../sources/prs/sglang/PR-13646.md) | [DeepSeekV3.2] Enable pure TP & Partial DP Attention | 2025-11-20 |  | attention |
| [#13555](../sources/prs/sglang/PR-13555.md) | Fix target MLA with eagle3 support for PD disaggregation | 2025-11-19 |  | mla |
| [#13573](../sources/prs/sglang/PR-13573.md) | [BugFix] fix prefixcache performance and accuracy on ascend | 2025-11-19 |  | attention, mla |
| [#13489](../sources/prs/sglang/PR-13489.md) | Flashinfer TRTLLM-GEN-MoE + Qwen3 | 2025-11-18 |  | moe |
| [#13324](../sources/prs/sglang/PR-13324.md) | Support weight update for blackwell DeepGEMM | 2025-11-15 |  | gemm |
| [#13263](../sources/prs/sglang/PR-13263.md) | diffusion: enable fa4 for blackwell | 2025-11-14 | kernel-fusion, pipeline-stages | attention, kernel-fusion, pipeline-stages |
| [#13264](../sources/prs/sglang/PR-13264.md) | [NVIDIA] Fix broken fp8 MoE of deepseek v3 | 2025-11-14 |  | fp8, moe, quantization |
| [#13274](../sources/prs/sglang/PR-13274.md) | [NVIDIA] Fix use case of SGLANG_ENABLE_FLASHINFER_GEMM | 2025-11-14 |  | fp8, gemm, quantization |
| [#13115](../sources/prs/sglang/PR-13115.md) | support mtp with deepseek r1 nvfp4 model | 2025-11-12 |  | attention, fp4, mla |
| [#13147](../sources/prs/sglang/PR-13147.md) | Aiter fp8 kv cache | 2025-11-12 |  | attention, fp8, quantization |
| [#13151](../sources/prs/sglang/PR-13151.md) | Support internvl on Blackwell (which doesn't support fa3): add `SingletonCache` support to Vision{Sdpa|Triton|Ascend}Attention | 2025-11-12 |  | attention |
| [#13158](../sources/prs/sglang/PR-13158.md) | [NPU]Optimization of `forward_npu` for `UnquantizedFusedMoEMethod` | 2025-11-12 | kernel-fusion | kernel-fusion, moe, quantization |
| [#13162](../sources/prs/sglang/PR-13162.md) | Fix nan in global scaling factor for large scale nvfp4 EP | 2025-11-12 | kernel-fusion | fp4, kernel-fusion, moe |
| [#13049](../sources/prs/sglang/PR-13049.md) | Support moe topk sigmoid kernel | 2025-11-11 |  | moe |
| [#13087](../sources/prs/sglang/PR-13087.md) | [sgl-kernel] support custom fp8 flashmla kernel | 2025-11-11 |  | fp8, mla |
| [#13022](../sources/prs/sglang/PR-13022.md) | [Deepseek V3.2] Use torch.compile to speed up torch.cat in nsa | 2025-11-10 |  | attention |
| [#12888](../sources/prs/sglang/PR-12888.md) | Apply moe_reduce_sum kernel for fused_marlin_moe | 2025-11-08 | kernel-fusion | kernel-fusion, moe |
| [#12816](../sources/prs/sglang/PR-12816.md) | [Deepseek V3.2] Only skip Indexer logits computation when is_extend_without_speculative | 2025-11-07 |  | attention |
| [#12724](../sources/prs/sglang/PR-12724.md) | [fix] Only enable flashinfer all reduce fusion by default for single-node servers | 2025-11-06 | kernel-fusion | kernel-fusion |
| [#12758](../sources/prs/sglang/PR-12758.md) | [Bugfix] Fix illegal memory access | 2025-11-06 | kernel-fusion | fp4, kernel-fusion, moe |
| [#12759](../sources/prs/sglang/PR-12759.md) | [Ascend] support Kimi-K2-Thinking | 2025-11-06 |  | moe, quantization |
| [#12778](../sources/prs/sglang/PR-12778.md) | Update dsv3 quantization auto setting for sm100 | 2025-11-06 |  | quantization |
| [#12782](../sources/prs/sglang/PR-12782.md) | ignore the deepgemm check when the model weight with nvfp4 and moe ba… | 2025-11-06 |  | fp4, gemm, moe |
| [#12788](../sources/prs/sglang/PR-12788.md) | [DeepSeek-V3.2][NSA] Enable MHA Pathway for Short Sequence Prefill on B200 (SM100) | 2025-11-06 |  | attention, prefill |
| [#12666](../sources/prs/sglang/PR-12666.md) | [sgl-kernel][5/N]Support Expert Specialization Grouped GEMM | 2025-11-05 |  | gemm, moe |
| [#12581](../sources/prs/sglang/PR-12581.md) | [NVIDIA] Fix CUDA arch requirement in nvfp4 cast | 2025-11-04 |  | fp4, gemm, nvfp4 |
| [#12612](../sources/prs/sglang/PR-12612.md) | feat: Add FP4 (E2M1) KV Cache Support for MHA | 2025-11-04 |  | fp4 |
| [#12640](../sources/prs/sglang/PR-12640.md) | [NVIDIA] Fix wrong symmetric sizes for fp4 cases | 2025-11-04 |  | fp4, quantization |
| [#12543](../sources/prs/sglang/PR-12543.md) | Enable Flashinfer TRTLLM-GEN-MoE FP8 blockwise kernel for Qwen3-Next on Blackwell | 2025-11-03 | kernel-fusion | fp8, kernel-fusion, moe |
| [#12555](../sources/prs/sglang/PR-12555.md) | [CPU] Fix MoE layer support for DeepSeek-OCR models | 2025-11-03 |  | moe |
| [#12523](../sources/prs/sglang/PR-12523.md) | chore: upgrade flashinfer 0.5.0 | 2025-11-02 |  | gemm |
| [#12482](../sources/prs/sglang/PR-12482.md) | Use sgl fp4 quant kernel by default | 2025-11-01 |  | fp4, quantization |
| [#12491](../sources/prs/sglang/PR-12491.md) | [Ascend] Support enable-mixed-chunk in non-MLA scenarios | 2025-11-01 |  | attention, mla |
| [#12435](../sources/prs/sglang/PR-12435.md) | perf: trtllm mla performance minor improvements | 2025-10-31 |  | attention, mla |
| [#12453](../sources/prs/sglang/PR-12453.md) | [Fix] `concat_mla_absorb_q_kernel` fails for long inputs | 2025-10-31 |  | mla |
| [#12376](../sources/prs/sglang/PR-12376.md) | Replace [silu_and_mul_]scaled_fp4_group_quant by Flashinfer equivalent | 2025-10-30 |  | fp4, moe, quantization |
| [#12392](../sources/prs/sglang/PR-12392.md) | [sgl-kernel] clean up fa fetch in CMakeLists.txt | 2025-10-30 |  | gemm |
| [#12308](../sources/prs/sglang/PR-12308.md) | fix: Llama 4 BF16 load on Blackwell | 2025-10-29 |  | gemm |
| [#12325](../sources/prs/sglang/PR-12325.md) | Fix Flashinfer Backend for SM120 Usage | 2025-10-29 |  | attention, mla |
| [#12347](../sources/prs/sglang/PR-12347.md) | fix: llama 4 + trtllm gen + fp8 kv cache incompatibility | 2025-10-29 |  | fp8 |
| [#12259](../sources/prs/sglang/PR-12259.md) | [hotfix] missing `w13_weight_fp8` and `w2_weight_fp8` in UE8M0 requantization | 2025-10-28 |  | fp8, gemm, moe |
| [#12294](../sources/prs/sglang/PR-12294.md) | [Deepseek V3.2] Enable flashmla_auto with MTP | 2025-10-28 |  | attention, mla |
| [#12295](../sources/prs/sglang/PR-12295.md) | fix seqlen bug for trtllm_mla's draft_extend | 2025-10-28 |  | attention, mla |
| [#12214](../sources/prs/sglang/PR-12214.md) | [Ascend][feature] support L1+ L2 radixcache on ascend | 2025-10-27 |  | attention |
| [#12215](../sources/prs/sglang/PR-12215.md) | [DeepseekV32]: use `_concat_mla_absorb_q_general` to replace `torch.cat` | 2025-10-27 |  | attention, mla |
| [#12065](../sources/prs/sglang/PR-12065.md) | (1/n)support context parallel with deepseekv3.2-DSA | 2025-10-24 |  | attention |
| [#12078](../sources/prs/sglang/PR-12078.md) | [Ascend] qwen optimization | 2025-10-24 | kernel-fusion | attention, kernel-fusion, moe |
| [#12080](../sources/prs/sglang/PR-12080.md) | [sgl-kernel][4/N]Support Expert Specialization Grouped GEMM | 2025-10-24 |  | fp8, gemm |
| [#12009](../sources/prs/sglang/PR-12009.md) | Fixed aarch64 flash-mla | 2025-10-23 |  | mla |
| [#12018](../sources/prs/sglang/PR-12018.md) | Feature/nano v2 offline modelopt fp8 and nvfp4 | 2025-10-23 |  | attention, fp4, fp8 |
| [#11892](../sources/prs/sglang/PR-11892.md) | DeepSeek-V3.2: Add Adaptive MHA Attention Pathway for Short-Sequence Prefill | 2025-10-21 |  | attention, prefill |
| [#11933](../sources/prs/sglang/PR-11933.md) | chore: upgrade flashinfer 0.4.1 | 2025-10-21 |  | gemm |
| [#11866](../sources/prs/sglang/PR-11866.md) | Support nvidia/NVIDIA-Nemotron-Nano-9B-v2-FP8/NVFP4 | 2025-10-20 |  | attention, fp4, fp8 |
| [#11805](../sources/prs/sglang/PR-11805.md) | Change bf16 to fp8 for some gemms in attention for DeepSeek ckpt v2 | 2025-10-18 |  | attention, fp8, gemm |
| [#11813](../sources/prs/sglang/PR-11813.md) | Use cutlass fp4 gemm by default | 2025-10-18 |  | fp4, gemm, quantization |
| [#11737](../sources/prs/sglang/PR-11737.md) | support cutlass fp4 kernel in sm120 | 2025-10-17 |  | fp4, gemm, moe |
| [#11708](../sources/prs/sglang/PR-11708.md) | Support running FP4 Deepseek on SM120. | 2025-10-16 |  | attention, fp4, fp8 |
| [#11717](../sources/prs/sglang/PR-11717.md) | [sgl-kernel] support flashmla libtorch | 2025-10-16 |  | mla |
| [#11655](../sources/prs/sglang/PR-11655.md) | [DeepseekV32] Enable flashmla_prefill kernel with fp8 kvcache | 2025-10-15 |  | attention, fp8, mla |
| [#11664](../sources/prs/sglang/PR-11664.md) | Use trtllm_mla decode kernel for draft extend in speculative decoding | 2025-10-15 |  | attention, decode, mla |
| [#11606](../sources/prs/sglang/PR-11606.md) | [NVIDIA] FA3/FA4 Fix  | 2025-10-14 |  | attention, flash-attention |
| [#11611](../sources/prs/sglang/PR-11611.md) | Support shared experts overlap in cutlass moe | 2025-10-14 | kernel-fusion | kernel-fusion, moe, quantization |
| [#11508](../sources/prs/sglang/PR-11508.md) | Improve Kernel Build Time | 2025-10-12 |  | gemm |
| [#11432](../sources/prs/sglang/PR-11432.md) | [sgl-kernel][1/N]Support Expert Specialization Grouped GEMM | 2025-10-10 |  | fp8, gemm, grouped-gemm |
| [#11349](../sources/prs/sglang/PR-11349.md) | [AMD] Clean up vllm dependencies in moe_runner/triton.py | 2025-10-09 |  | moe |
| [#11287](../sources/prs/sglang/PR-11287.md) | [NVIDIA] Add new SMs support for Spark & Thor | 2025-10-07 |  | fp4, gemm, nvfp4 |
| [#11274](../sources/prs/sglang/PR-11274.md) | disable sm100 for FlashMLA and fast-hadamard-transform in cuda12.6.1 | 2025-10-06 |  | mla |
| [#11056](../sources/prs/sglang/PR-11056.md) | chore: upgrade sgl-kernel 0.3.13 | 2025-09-29 |  | gemm |
| [#11081](../sources/prs/sglang/PR-11081.md) | Fix DSR1 accuracy for flashinfer_trtllm MoE with FP8 quantization | 2025-09-29 | kernel-fusion | fp8, kernel-fusion, moe |
| [#10985](../sources/prs/sglang/PR-10985.md) | Quick Fix: fix Qwen3-VL launch failure caused by MRotaryEmbedding arg | 2025-09-27 |  | moe |
| [#10937](../sources/prs/sglang/PR-10937.md) | [2/2] Support MHA prefill with FlashAttention 4. | 2025-09-26 |  | attention, prefill |
| [#10779](../sources/prs/sglang/PR-10779.md) | Fuse quantize and rope in trtllm_mla MTP | 2025-09-23 |  | attention, mla, quantization |
| [#10758](../sources/prs/sglang/PR-10758.md) | Fix MTP MoE weight loading with NVFP4 target model. | 2025-09-22 | kernel-fusion | fp4, kernel-fusion, moe |
| [#10701](../sources/prs/sglang/PR-10701.md) | Unify SGL Kernel Releases | 2025-09-21 |  | gemm |
| [#10714](../sources/prs/sglang/PR-10714.md) | Optimize cutlass int8 gemm kernel for large M on SM89 Ada GPU | 2025-09-21 |  | gemm |
| [#10688](../sources/prs/sglang/PR-10688.md) | [Auto Sync] Update modelopt_quant.py (20250920) | 2025-09-20 |  | quantization |
| [#10622](../sources/prs/sglang/PR-10622.md) | support qwen3-next-fp8 deepep | 2025-09-18 |  | fp8, moe |
| [#10543](../sources/prs/sglang/PR-10543.md) | [sgl-kernel] Optimize concat_mla_k kernel | 2025-09-17 |  | mla |
| [#10574](../sources/prs/sglang/PR-10574.md) | [Ascend]optimize Qwen3 on Ascend | 2025-09-17 |  | quantization |
| [#10579](../sources/prs/sglang/PR-10579.md) | Fix bias handling in TritonMoeQuantInfo within quantization/mxfp4.py | 2025-09-17 |  | fp4, moe, quantization |
| [#10491](../sources/prs/sglang/PR-10491.md) | Update CUTLASS. Refine KernelSchedule for fp8 (grouped) gemm. | 2025-09-16 |  | fp8, gemm, moe |
| [#10498](../sources/prs/sglang/PR-10498.md) | Cache the result of `is_blackwell` platform check | 2025-09-16 |  | gemm, quantization |
| [#10526](../sources/prs/sglang/PR-10526.md) | Enable trtllm mla prefix extend | 2025-09-16 |  | attention, mla |
| [#10433](../sources/prs/sglang/PR-10433.md) | feat: add dsv3 fp4 cutlass moe etp ut | 2025-09-15 |  | fp4, moe |
| [#10414](../sources/prs/sglang/PR-10414.md) | Fix cutlass moe accuracy drop caused by attention UB from DP padding mode | 2025-09-14 |  | attention, moe |
| [#10422](../sources/prs/sglang/PR-10422.md) | Support single batch overlap | 2025-09-14 |  | moe, quantization |
| [#10426](../sources/prs/sglang/PR-10426.md) | Fix correction bias undefined behavior for nvfp4 models | 2025-09-14 | kernel-fusion | fp4, kernel-fusion, moe |
| [#10403](../sources/prs/sglang/PR-10403.md) | support qwen3_next blackwell | 2025-09-13 |  | attention |
| [#10343](../sources/prs/sglang/PR-10343.md) | fix: resolve gb200 image link | 2025-09-11 |  | gemm |
| [#10275](../sources/prs/sglang/PR-10275.md) | Add support for bf16 x bf16 cutlass fused MoE | 2025-09-10 | kernel-fusion | kernel-fusion, moe, quantization |
| [#10281](../sources/prs/sglang/PR-10281.md) | Enables TRT-LLM backend to be used for target_verify | 2025-09-10 |  | attention, mla |
| [#10154](../sources/prs/sglang/PR-10154.md) | Enable native ModelOpt quantization support (3/3) | 2025-09-08 |  | quantization |
| [#10180](../sources/prs/sglang/PR-10180.md) | Fix chunked prefix cache for nvfp4 | 2025-09-08 |  | attention, fp4, mla |
| [#10130](../sources/prs/sglang/PR-10130.md) | [Feature] Add MLAProcess for DeepSeek MLA on NPU | 2025-09-07 |  | attention, mla |
| [#10101](../sources/prs/sglang/PR-10101.md) | Optimize nvfp4 block scaled gemm kernel when M is small. | 2025-09-06 |  | fp4, gemm, nvfp4 |
| [#10058](../sources/prs/sglang/PR-10058.md) | Disable kernel cutlass_mla_decode on SM103 | 2025-09-05 |  | attention, decode, mla |
| [#10078](../sources/prs/sglang/PR-10078.md) | feat: Add FP4 (E2M1) KV Cache Support with Quantization Utilities for MLA | 2025-09-05 |  | attention, fp4, mla |
| [#9991](../sources/prs/sglang/PR-9991.md) | Enable native ModelOpt quantization support (2/3) | 2025-09-04 |  | quantization |
| [#9946](../sources/prs/sglang/PR-9946.md) | [Fix] DeepSeek EP accuracy issue on B200 GPUs | 2025-09-03 |  | gemm, quantization |
| [#9969](../sources/prs/sglang/PR-9969.md) | CUTLASS fp8 blockwise gemm support of sm120 | 2025-09-03 |  | fp8, gemm |
| [#9928](../sources/prs/sglang/PR-9928.md) | support using fa4 on deepseek on blackwell | 2025-09-02 |  | attention |
| [#9834](../sources/prs/sglang/PR-9834.md) | perf: Avoid unnecessary data type conversions for DeepSeek-V3 on Blackwell | 2025-08-31 |  | gemm |
| [#9807](../sources/prs/sglang/PR-9807.md) | Make fp4_quantize kernels work on sm103 | 2025-08-30 |  | fp4, gemm, nvfp4 |
| [#9824](../sources/prs/sglang/PR-9824.md) | [Model] Support Meituan LongCat-Flash && LongCat-Flash-MTP | 2025-08-30 |  | moe, quantization |
| [#9784](../sources/prs/sglang/PR-9784.md) | Raise error when `topk>1` and `page>1` for paged attention backends. | 2025-08-29 |  | attention |
| [#9789](../sources/prs/sglang/PR-9789.md) | Make sm100 fp8 kernels available on sm103 | 2025-08-29 |  | fp8, gemm, moe |
| [#9744](../sources/prs/sglang/PR-9744.md) | [CPU] Add FP8 Bmm support | 2025-08-28 | kernel-fusion | attention, fp8, gemm |
| [#9678](../sources/prs/sglang/PR-9678.md) | fix mooncake store mla zero copy meta | 2025-08-27 |  | mla |
| [#9679](../sources/prs/sglang/PR-9679.md) | move is_sm90_supported/is_sm100_supported to python/sglang/srt/utils.py | 2025-08-27 |  | attention, fp4, fp8 |
| [#9712](../sources/prs/sglang/PR-9712.md) | [ModelOpt] Fix Weight Loading for DSR1-FP4 Quantization | 2025-08-27 |  | fp4, quantization |
| [#9660](../sources/prs/sglang/PR-9660.md) | Single Batch Overlap for MoE Models | 2025-08-26 | kernel-fusion | gemm, kernel-fusion, moe |
| [#9589](../sources/prs/sglang/PR-9589.md) | Tiny fix wrong comments | 2025-08-25 |  | quantization |
| [#9556](../sources/prs/sglang/PR-9556.md) | [NVIDIA] [2/N] Optimize `silu_and_mul_scaled_fp4_grouped_quant` perf | 2025-08-24 |  | fp4, gemm, nvfp4 |
| [#9559](../sources/prs/sglang/PR-9559.md) | Update CUTLASS 4.2 & Enable K-Major Scale Factor for SM90 FP8 Blockwise Group GEMM | 2025-08-24 |  | fp8, gemm, moe |
| [#9530](../sources/prs/sglang/PR-9530.md) | fix: blackwell dsv3 fp8 issue temporary solution | 2025-08-23 |  | fp8, gemm, quantization |
| [#9473](../sources/prs/sglang/PR-9473.md) | [fix] Fix mxfp4 triton MoE tp bug | 2025-08-22 | kernel-fusion | fp4, kernel-fusion, moe |
| [#9477](../sources/prs/sglang/PR-9477.md) | Optimize moe_sum_reduce_kernel | 2025-08-22 | kernel-fusion | kernel-fusion, moe |
| [#9403](../sources/prs/sglang/PR-9403.md) | [sgl-kernel] feat: Support sm120 cutlass fp8 gemm kernel | 2025-08-20 |  | fp8, gemm |
| [#9339](../sources/prs/sglang/PR-9339.md) | Support trtllm_allreduce_fusion in flashinfer for cuda<12.8 | 2025-08-19 | kernel-fusion | kernel-fusion |
| [#9346](../sources/prs/sglang/PR-9346.md) | Fix FP4 inference corruption issue in glm4.5-air model | 2025-08-19 |  | fp4, gemm |
| [#9359](../sources/prs/sglang/PR-9359.md) | Support DP attention with GPT-OSS | 2025-08-19 |  | attention |
| [#9272](../sources/prs/sglang/PR-9272.md) | [fix]:  fix cutlass moe ut and and Opt H20 cutlass groupGemm performance | 2025-08-17 |  | fp8, gemm, moe |
| [#9199](../sources/prs/sglang/PR-9199.md) | [NVIDIA] [3/N] Nvfp4 Masked Gemm: Add flashinfer grouped_gemm_nt_masked  | 2025-08-14 |  | fp4, gemm, grouped-gemm |
| [#9200](../sources/prs/sglang/PR-9200.md) | [NVIDA] [1/N] Nvfp4 Masked Gemm: Add quant op for the flashinfer grouped gemm | 2025-08-14 |  | fp4, gemm, nvfp4 |
| [#9162](../sources/prs/sglang/PR-9162.md) | Faster weight processing (trtllm-gen moe nvfp4) | 2025-08-13 |  | fp4, moe, nvfp4 |
| [#9060](../sources/prs/sglang/PR-9060.md) | [sgl-kernel] Support FlashInfer top_k_top_p_sampling_from_logits | 2025-08-11 |  | gemm |
| [#9011](../sources/prs/sglang/PR-9011.md) | [fix] fix enable_pdl for blackwell | 2025-08-09 |  | gemm |
| [#8955](../sources/prs/sglang/PR-8955.md) | [NVIDIA] Fix missing `get_col_major_tma_aligned_tensor` for Blackwell deepgemm in EpMoE | 2025-08-08 |  | gemm, moe, tma |
| [#8962](../sources/prs/sglang/PR-8962.md) | optimize: reduce shulffle and quantization overhead in cutlass_moe sm90 | 2025-08-08 |  | fp8, moe, quantization |
| [#8898](../sources/prs/sglang/PR-8898.md) | [Perf] Auto enable best flashinfer mxfp4  kernel in b200 | 2025-08-07 | kernel-fusion | fp4, kernel-fusion, moe |
| [#8908](../sources/prs/sglang/PR-8908.md) | Fix hopper launch gpt-oss model illegal memory | 2025-08-07 |  | fp4, quantization |
| [#8928](../sources/prs/sglang/PR-8928.md) | chore: support blackwell cu129 image | 2025-08-07 |  | gemm |
| [#8782](../sources/prs/sglang/PR-8782.md) | feat: add trtllm-gen mha from direct call | 2025-08-05 |  | attention |
| [#8818](../sources/prs/sglang/PR-8818.md) | [Perf] Tunings for SM100 FP8 CUTLASS kernel | 2025-08-05 |  | fp8, gemm |
| [#8766](../sources/prs/sglang/PR-8766.md) | Fix mismatch between padded_scales shape and reshape dimensions in modelopt quantization | 2025-08-04 |  | quantization |
| [#8731](../sources/prs/sglang/PR-8731.md) | fuse allreduce and residual_rmsnorm | 2025-08-03 | kernel-fusion | kernel-fusion, moe |
| [#8678](../sources/prs/sglang/PR-8678.md) | feat: support cutlass_moe_fp8 kernel for fusedmoe in sm90 | 2025-08-01 | kernel-fusion | fp8, kernel-fusion, moe |
| [#8593](../sources/prs/sglang/PR-8593.md) | Add Support for Page Size greater than 1 for Flashinfer MLA Backend | 2025-07-31 |  | attention, mla |
| [#8638](../sources/prs/sglang/PR-8638.md) | TRTLLM-MLA FP8 path | 2025-07-31 |  | attention, fp8, mla |
| [#8552](../sources/prs/sglang/PR-8552.md) | [NVIDIA] Add Low Latency NVFP4 decode kernels from Flashinfer | 2025-07-30 | kernel-fusion | decode, fp4, kernel-fusion |
| [#8535](../sources/prs/sglang/PR-8535.md) | Update cutlass_moe.py | 2025-07-29 |  | moe |
| [#8545](../sources/prs/sglang/PR-8545.md) | Update cutlass_moe.py | 2025-07-29 |  | moe |
| [#8464](../sources/prs/sglang/PR-8464.md) | [2/N]Support DeepSeek-R1 w4a8 low latency deepep | 2025-07-28 |  | fp8, moe, quantization |
| [#8328](../sources/prs/sglang/PR-8328.md) | [new feat] ascend backend support fia fusion kernel | 2025-07-24 | kernel-fusion | attention, kernel-fusion, mla |
| [#8247](../sources/prs/sglang/PR-8247.md) | [1/N]Support  DeepSeek-R1 w4a8 normal deepep | 2025-07-22 |  | fp8, moe, quantization |
| [#8258](../sources/prs/sglang/PR-8258.md) | Support triton kernels v3.4.0 for fused_moe | 2025-07-22 | kernel-fusion | kernel-fusion, moe, quantization |
| [#8195](../sources/prs/sglang/PR-8195.md) | [fix] fix modelopt fp4 on b200 | 2025-07-20 |  | fp4, quantization |
| [#8130](../sources/prs/sglang/PR-8130.md) | [sgl-kernel] Opt per_token_quant_fp8 with warp reduce | 2025-07-18 |  | fp8, gemm |
| [#8118](../sources/prs/sglang/PR-8118.md) | [feat] Support tp mode for DeepSeek-R1-W4AFP8 | 2025-07-17 | kernel-fusion | fp8, kernel-fusion, moe |
| [#8127](../sources/prs/sglang/PR-8127.md) | [Fix][Ready]Fix register spilling in cutlass nvfp4 gemm kernel on Blackwell | 2025-07-17 |  | fp4, gemm, nvfp4 |
| [#7884](../sources/prs/sglang/PR-7884.md) | [kernel] opt moe align block kernel by block/warp scan algorithm | 2025-07-09 |  | moe |
| [#7912](../sources/prs/sglang/PR-7912.md) | Qwen FP8/NVFP4 ModelOPT Quantization support | 2025-07-09 |  | fp4, fp8, nvfp4 |
| [#7762](../sources/prs/sglang/PR-7762.md) | feat: support DeepSeek-R1-W4AFP8 model with ep-moe mode | 2025-07-04 |  | fp8, moe, quantization |
| [#7772](../sources/prs/sglang/PR-7772.md) | [1/n]: add cutlass W4A8 moe kernel for hopper architecture | 2025-07-04 |  | gemm, moe, tma |
| [#7722](../sources/prs/sglang/PR-7722.md) | Ascend attention backend(PA&MLA) | 2025-07-02 | kernel-fusion | attention, kernel-fusion, mla |
| [#7725](../sources/prs/sglang/PR-7725.md) | Support FlashAttention3 page_size > 1 and topk > 1 case with paged attn and spec decode | 2025-07-02 |  | attention, decode, quantization |
| [#7689](../sources/prs/sglang/PR-7689.md) | Integrate triton moe kernel | 2025-07-01 | kernel-fusion | kernel-fusion, moe |
| [#7649](../sources/prs/sglang/PR-7649.md) | [Feature] CUDA Green Context Support | 2025-06-30 |  | gemm |
| [#7663](../sources/prs/sglang/PR-7663.md) | chore: upgrade flashinfer v0.2.7 jit | 2025-06-30 |  | gemm |
| [#7667](../sources/prs/sglang/PR-7667.md) | Add fp4 quantize before all-gather for Flashinfer cutlass MoE DP (max throughput) | 2025-06-30 | kernel-fusion | attention, fp4, kernel-fusion |
| [#7634](../sources/prs/sglang/PR-7634.md) | [Feature] Layer-wise Prefill | 2025-06-29 |  | gemm, moe, prefill |
| [#7635](../sources/prs/sglang/PR-7635.md) | Apply dsv3_fused_a_gemm kernel | 2025-06-29 | kernel-fusion | gemm, kernel-fusion |
| [#7621](../sources/prs/sglang/PR-7621.md) | [b200] support trt-llm allreduce fuse rms_norm_add kernel | 2025-06-28 | kernel-fusion | kernel-fusion |
| [#7543](../sources/prs/sglang/PR-7543.md) | [CMake] Fix sgl-kernel CMakeLists for Blackwell | 2025-06-26 |  | gemm |
| [#7549](../sources/prs/sglang/PR-7549.md) | Add Tencent HunYuanMoEV1 model support | 2025-06-26 |  | moe |
| [#7462](../sources/prs/sglang/PR-7462.md) | Support non-contiguous query input for extend/decode attention | 2025-06-23 |  | attention, decode |
| [#7437](../sources/prs/sglang/PR-7437.md) | Fuse sorted_token_ids padding to moe_align_block_size kernel | 2025-06-22 |  | moe |
| [#7444](../sources/prs/sglang/PR-7444.md) | fix: fix apply_shuffle_mul_sum | 2025-06-22 |  | moe |
| [#7409](../sources/prs/sglang/PR-7409.md) | Fix CPU offloading for MLA memory pool | 2025-06-21 |  | mla |
| [#7376](../sources/prs/sglang/PR-7376.md) | Fix MTP with Deepseek R1 Fp4 | 2025-06-20 | kernel-fusion | fp4, kernel-fusion, moe |
| [#7378](../sources/prs/sglang/PR-7378.md) | Quick fix for DeepGemm requant to also cover MTP. | 2025-06-20 |  | gemm |
| [#7391](../sources/prs/sglang/PR-7391.md) | Fix torch compile run | 2025-06-20 | kernel-fusion | fp8, kernel-fusion, moe |
| [#7392](../sources/prs/sglang/PR-7392.md) | [AMD][Quantization] Add `int4fp8_moe` online quantization on ROCm | 2025-06-20 |  | fp8, moe, quantization |
| [#7302](../sources/prs/sglang/PR-7302.md) | Support NVFP4 quantized dense models on AMD CDNA2/CDNA3 GPUs | 2025-06-18 |  | fp4, nvfp4, quantization |
| [#7310](../sources/prs/sglang/PR-7310.md) | Let EP prefill support new DeepGEMM | 2025-06-18 |  | gemm, moe, prefill |
| [#7313](../sources/prs/sglang/PR-7313.md) | Kernels for efficient KV cache IO | 2025-06-18 |  | gemm |
| [#7327](../sources/prs/sglang/PR-7327.md) | FlashInfer NVFP4 MoE with EP & 2-stream shared expert | 2025-06-18 | kernel-fusion | fp4, kernel-fusion, moe |
| [#7331](../sources/prs/sglang/PR-7331.md) | fix: resolve blackwell deepep image issue | 2025-06-18 |  | gemm |
| [#7268](../sources/prs/sglang/PR-7268.md) | [AMD] add aiter fused moe in DeepEP path | 2025-06-17 | kernel-fusion | kernel-fusion, moe |
| [#7278](../sources/prs/sglang/PR-7278.md) | Add CUTLASS FP8 Blockscale MoE kernel for Hopper architecture | 2025-06-17 |  | fp8, gemm, moe |
| [#7225](../sources/prs/sglang/PR-7225.md) | feat: support compatibility between MTP and two-batch-overlap | 2025-06-16 |  | attention |
| [#7228](../sources/prs/sglang/PR-7228.md) | Minor style and doc fix | 2025-06-16 |  | attention, mla |
| [#7247](../sources/prs/sglang/PR-7247.md) | [fix] fix DeepGEMM blackwell input quant & ut & fix style and log | 2025-06-16 |  | fp8, gemm, moe |
| [#7198](../sources/prs/sglang/PR-7198.md) | Fix error when disabling new DeepGEMM | 2025-06-15 |  | gemm, moe |
| [#7204](../sources/prs/sglang/PR-7204.md) | Fix grammar abort & Minor style fixes | 2025-06-15 |  | attention, decode, mla |
| [#7172](../sources/prs/sglang/PR-7172.md) | Support new DeepGEMM | 2025-06-14 | kernel-fusion | fp8, gemm, kernel-fusion |
| [#7182](../sources/prs/sglang/PR-7182.md) | Tiny let DeepGEMM scale checks cover more cases | 2025-06-14 |  | fp8, gemm, quantization |
| [#7186](../sources/prs/sglang/PR-7186.md) | chore: upgrade sgl-kernel v0.1.8.post2 | 2025-06-14 |  | attention, gemm, mla |
| [#7187](../sources/prs/sglang/PR-7187.md) | [AMD] Fail gracefully when AITER is unavailable gfx90a GPUs | 2025-06-14 |  | fp8, quantization |
| [#7191](../sources/prs/sglang/PR-7191.md) | Fix a minor bug related to DeepGEMM upgrade | 2025-06-14 |  | gemm, quantization |
| [#7149](../sources/prs/sglang/PR-7149.md) | Enable native ModelOpt quantization support (1/3)  | 2025-06-13 |  | quantization |
| [#7160](../sources/prs/sglang/PR-7160.md) | [amd] Opt dsv3 moe | 2025-06-13 |  | fp8, moe, quantization |
| [#7163](../sources/prs/sglang/PR-7163.md) | [EAGLE] Refactor code for page size > 1 & more simplifications | 2025-06-13 |  | attention, mla |
| [#7164](../sources/prs/sglang/PR-7164.md) | Fix Deepseek R1 0528 FP4 tensor name mismatch issue during weights loading. | 2025-06-13 |  | fp4 |
| [#7119](../sources/prs/sglang/PR-7119.md) | feat: update blackwell setup | 2025-06-12 |  | gemm |
| [#7125](../sources/prs/sglang/PR-7125.md) | fix amd EP MoE FP8 issue | 2025-06-12 |  | fp8, moe |
| [#7129](../sources/prs/sglang/PR-7129.md) | Enable ModelOpt Llama4 fp8 checkpoint deployment in SGLang | 2025-06-12 | kernel-fusion | fp8, kernel-fusion, moe |
| [#7093](../sources/prs/sglang/PR-7093.md) | Fix positional argument | 2025-06-11 | kernel-fusion | fp8, kernel-fusion, moe |
| [#7023](../sources/prs/sglang/PR-7023.md) | Update default settings for blackwell | 2025-06-10 | kernel-fusion | fp8, kernel-fusion, moe |
| [#7037](../sources/prs/sglang/PR-7037.md) | Clean up server_args.py | 2025-06-10 |  | gemm, quantization |
| [#7057](../sources/prs/sglang/PR-7057.md) | Tiny fix cutlass_mla_get_workspace_size stub incorrect signature | 2025-06-10 |  | attention, mla |
| [#6998](../sources/prs/sglang/PR-6998.md) | Fix cutlass MLA gets almost zero accuracy | 2025-06-09 |  | attention, mla |
| [#7015](../sources/prs/sglang/PR-7015.md) | Fix torchvision version for Blackwell | 2025-06-09 |  | gemm |
| [#6958](../sources/prs/sglang/PR-6958.md) | chore: upgrade flashinfer v0.2.6.post1 jit | 2025-06-08 | kernel-fusion | kernel-fusion, moe, quantization |
| [#6970](../sources/prs/sglang/PR-6970.md) | Fuse routed scaling factor in deepseek | 2025-06-08 | kernel-fusion | fp8, kernel-fusion, moe |
| [#6942](../sources/prs/sglang/PR-6942.md) | [sgl-kernel] update deepgemm | 2025-06-07 |  | gemm |
| [#6916](../sources/prs/sglang/PR-6916.md) | Add a CUDA kernel for fusing mapping and weighted sum for MoE. | 2025-06-06 |  | fp8, moe |
| [#6919](../sources/prs/sglang/PR-6919.md) | [sgl-kernel] Add cuda kernel for moe_ep_silu_and_mul | 2025-06-06 |  | moe |
| [#6929](../sources/prs/sglang/PR-6929.md) | [perf][sgl-kernel] extend cutlass_mla_decode to support num_head < 128 | 2025-06-06 | tile-scheduling | attention, decode, flash-attention |
| [#6930](../sources/prs/sglang/PR-6930.md) | [Feature] Support Flashinfer fmha on Blackwell | 2025-06-06 |  | attention, flash-attention, fp8 |
| [#6890](../sources/prs/sglang/PR-6890.md) | Use deepgemm instead of triton for fused_qkv_a_proj_with_mqa | 2025-06-05 | kernel-fusion | fp8, gemm, kernel-fusion |
| [#6853](../sources/prs/sglang/PR-6853.md) | [DeepseekR1-FP4] Add Support for nvidia/DeepSeekR1-FP4 model | 2025-06-04 | kernel-fusion | fp4, kernel-fusion, moe |
| [#6858](../sources/prs/sglang/PR-6858.md) | fix ep_moe_reorder kernel bugs | 2025-06-04 |  | moe |
| [#6821](../sources/prs/sglang/PR-6821.md) | feat: integrate deepgemm into EPMoE | 2025-06-03 |  | fp8, gemm, moe |
| [#6833](../sources/prs/sglang/PR-6833.md) | CPU: map changes from developing branch in sgl-kernel | 2025-06-03 |  | decode, fp8, gemm |
| [#6837](../sources/prs/sglang/PR-6837.md) | [EP] Add cuda kernel for moe_ep_post_reorder | 2025-06-03 |  | moe |
| [#6842](../sources/prs/sglang/PR-6842.md) | Fix AWQ Dequant and Weight Loading of deepseek v2 | 2025-06-03 |  | gemm |
| [#6803](../sources/prs/sglang/PR-6803.md) | Correctly abort the failed grammar requests & Improve the handling of abort | 2025-06-02 |  | gemm |
| [#6793](../sources/prs/sglang/PR-6793.md) | [PD] Add different TP sizes support for no-MLA models | 2025-05-31 |  | mla, prefill |
| [#6769](../sources/prs/sglang/PR-6769.md) | [CPU] add optimizations for INT8 and FP8 DeepSeek | 2025-05-30 | kernel-fusion | fp8, kernel-fusion, moe |
| [#6771](../sources/prs/sglang/PR-6771.md) | [CPU] support the case where num_attention_heads or intermediate_size is not divisible by the TP size | 2025-05-30 | kernel-fusion | attention, kernel-fusion, moe |
| [#6782](../sources/prs/sglang/PR-6782.md) | Support token-level quantization for EP MoE | 2025-05-30 |  | moe, quantization |
| [#6736](../sources/prs/sglang/PR-6736.md) | Set `num_fused_shared_experts` as `num_shared_experts` when shared_experts fusion is not disabled | 2025-05-29 | kernel-fusion | fp8, kernel-fusion, moe |
| [#6699](../sources/prs/sglang/PR-6699.md) | [EP] Add cuda kernel for moe_ep_pre_reorder | 2025-05-28 |  | moe |
| [#6709](../sources/prs/sglang/PR-6709.md) | Fix PP for Qwen3 MoE | 2025-05-28 |  | moe |
| [#6641](../sources/prs/sglang/PR-6641.md) | [CPU] [BF16] Call fused_experts_cpu, weight_packed_linear and bmm_cpu kernel in DeepSeek model | 2025-05-27 | kernel-fusion | gemm, kernel-fusion, moe |
| [#6673](../sources/prs/sglang/PR-6673.md) | Fix DeepEP error in Qwen 3 MoE models | 2025-05-27 |  | moe |
| [#6627](../sources/prs/sglang/PR-6627.md) | Refine pre_reorder_triton_kernel slightly to improve performance | 2025-05-26 | kernel-fusion | kernel-fusion, moe |
| [#6598](../sources/prs/sglang/PR-6598.md) | qwen3moe support two batch overlap | 2025-05-25 |  | moe |
| [#6474](../sources/prs/sglang/PR-6474.md) | Fix topk inference performance reduce | 2025-05-21 |  | moe |
| [#6479](../sources/prs/sglang/PR-6479.md) | [Feature] Support Flashinfer fp8 blockwise GEMM kernel on Blackwell | 2025-05-21 |  | fp8, gemm, quantization |
| [#6449](../sources/prs/sglang/PR-6449.md) | Fix bug of deepseek-v3 under DP+EP mode with large batchsize/seqlen | 2025-05-20 |  | fp8, gemm, quantization |
| [#6404](../sources/prs/sglang/PR-6404.md) | Add fp8 fused_experts kernel for CPU in sgl-kernel and add UT | 2025-05-19 | kernel-fusion | fp8, gemm, kernel-fusion |
| [#6389](../sources/prs/sglang/PR-6389.md) | [Feature] Comprehensive Hybrid Parallelism Support | 2025-05-18 |  | attention, decode, prefill |
| [#6369](../sources/prs/sglang/PR-6369.md) | reduce torch.zeros overhead in moe align block size kernel | 2025-05-17 | kernel-fusion | kernel-fusion, moe |
| [#6334](../sources/prs/sglang/PR-6334.md) | [Fix] Improve dependencies for Blackwell image | 2025-05-15 |  | gemm |
| [#6336](../sources/prs/sglang/PR-6336.md) | Upgrade  CUTLASS 4.0 | 2025-05-15 |  | fp8, gemm |
| [#6287](../sources/prs/sglang/PR-6287.md) | fix: fix MLA for ShardedModelLoader/RemoteModelLoader | 2025-05-14 |  | mla |
| [#6295](../sources/prs/sglang/PR-6295.md) | fix: enable multi-GPU Triton fused MoE tuning | 2025-05-14 | kernel-fusion | kernel-fusion, moe |
| [#6226](../sources/prs/sglang/PR-6226.md) | enable auto-round quantization model | 2025-05-12 | kernel-fusion | kernel-fusion, moe, quantization |
| [#6230](../sources/prs/sglang/PR-6230.md) | Enable FlashInfer support encoder models and add head_dim padding workaround | 2025-05-12 |  | attention |
| [#6147](../sources/prs/sglang/PR-6147.md) | Reduce MoE memory usage | 2025-05-09 |  | moe |
| [#6121](../sources/prs/sglang/PR-6121.md) | feat: add dp attention support for Qwen 2/3 MoE models, fixes #6088 | 2025-05-08 |  | attention, moe |
| [#6073](../sources/prs/sglang/PR-6073.md) | chore: upgrade deepgemm | 2025-05-07 |  | gemm |
| [#6081](../sources/prs/sglang/PR-6081.md) | feat: mtp support dp-attention | 2025-05-07 |  | attention, mla |
| [#6093](../sources/prs/sglang/PR-6093.md) | [1/2] Add Kernel support for Cutlass based Fused FP4 MoE | 2025-05-07 | kernel-fusion | fp4, gemm, kernel-fusion |
| [#6101](../sources/prs/sglang/PR-6101.md) | Cutlass MLA: Disable split kv due to https://github.com/NVIDIA/cutlass/issues/2274 | 2025-05-07 |  | attention, mla |
| [#6042](../sources/prs/sglang/PR-6042.md) | Support tuning moe for llama 4 model | 2025-05-06 | kernel-fusion | kernel-fusion, moe |
| [#6016](../sources/prs/sglang/PR-6016.md) | KV‑Cache (MHA, MLA): add missing start_layer / end_layer fields to MHATokenToKVPoolHost and MLATokenToKVPoolHost | 2025-05-05 |  | mla |
| [#6004](../sources/prs/sglang/PR-6004.md) | chore: upgrade cutlass 3.9.2 | 2025-05-04 |  | fp8, gemm |
| [#5981](../sources/prs/sglang/PR-5981.md) | [Feat] Enable PDL automatically on Hopper architecture | 2025-05-02 |  | gemm |
| [#5903](../sources/prs/sglang/PR-5903.md) | Add sm_120 for blackwell | 2025-04-30 |  | gemm |
| [#5917](../sources/prs/sglang/PR-5917.md) | [qwen3] support qwen3 ep moe | 2025-04-30 |  | moe |
| [#5868](../sources/prs/sglang/PR-5868.md) | Cutlass MLA decode - fix dtype error | 2025-04-29 |  | attention, decode, mla |
| [#5875](../sources/prs/sglang/PR-5875.md) | [Fix] Fix a bug for flashmla to run R1 model | 2025-04-29 |  | attention, mla |
| [#5889](../sources/prs/sglang/PR-5889.md) | Improve dp attention port assignment scheme | 2025-04-29 |  | attention |
| [#5390](../sources/prs/sglang/PR-5390.md) | Add Cutlass MLA attention backend | 2025-04-28 | warp-specialization, persistent-kernel, tile-scheduling | tcgen05, mla, moe |
| [#5820](../sources/prs/sglang/PR-5820.md) | cutlass 3.9 supported to improve fp8_blockwise_gemm | 2025-04-28 |  | fp8, gemm |
| [#5822](../sources/prs/sglang/PR-5822.md) | opt flashinfer mla cat | 2025-04-28 |  | attention, mla |
| [#5748](../sources/prs/sglang/PR-5748.md) | Fuse MLA set kv cache kernel | 2025-04-25 |  | attention, mla |
| [#5694](../sources/prs/sglang/PR-5694.md) | [2/2] Add python wrapper for CUTLASS FP8 Blockscale MoE Kernel.  | 2025-04-24 |  | fp8, moe, quantization |
| [#5724](../sources/prs/sglang/PR-5724.md) | [PP] Add pipeline parallelism | 2025-04-24 | pipeline-stages | attention, pipeline-stages |
| [#5662](../sources/prs/sglang/PR-5662.md) | [perf] dsv3 bmm fallback to bf16 | 2025-04-23 |  | fp8, quantization |
| [#5618](../sources/prs/sglang/PR-5618.md) | [fix] force use deepgemm in compile_deep_gemm | 2025-04-22 |  | gemm |
| [#5626](../sources/prs/sglang/PR-5626.md) |  DeepEP normal support deepgemm-contiguous | 2025-04-22 |  | fp8, gemm, moe |
| [#5628](../sources/prs/sglang/PR-5628.md) | Turn on DeepGemm By Default and Update Doc | 2025-04-22 |  | gemm, quantization |
| [#5432](../sources/prs/sglang/PR-5432.md) | [perf] introduce deep gemm group_gemm_masked as gemm | 2025-04-20 | fine-grained-quantization, kernel-fusion | gemm, moe, decode |
| [#5580](../sources/prs/sglang/PR-5580.md) | [feature] enable pre compile jit deep_gemm | 2025-04-20 |  | fp8, gemm, quantization |
| [#5546](../sources/prs/sglang/PR-5546.md) | Fix sampler nan check when calling top_k_top_p_sampling_from_probs | 2025-04-19 |  | gemm |
| [#5547](../sources/prs/sglang/PR-5547.md) | feat: use flashinfer jit package | 2025-04-19 |  | gemm |
| [#5476](../sources/prs/sglang/PR-5476.md) | Avoid computing lse in Ragged Prefill when there's no prefix. | 2025-04-16 |  | attention, mla, prefill |
| [#5395](../sources/prs/sglang/PR-5395.md) | chore: upgrade DeepGEMM | 2025-04-15 |  | gemm |
| [#5415](../sources/prs/sglang/PR-5415.md) | [PD] Fix dynamic port support and MLA buffer for Mooncake | 2025-04-15 |  | decode, mla, prefill |
| [#5417](../sources/prs/sglang/PR-5417.md) | [Feat] upgrade pytorch2.6 | 2025-04-15 |  | attention |
| [#5431](../sources/prs/sglang/PR-5431.md) | BLackwell cutlass mla: Add check for bad page size/block num combinations | 2025-04-15 |  | attention, mla |
| [#5370](../sources/prs/sglang/PR-5370.md) | [perf] experimental enhance fp8 per-tensor quant | 2025-04-14 |  | fp8, quantization |
| [#5371](../sources/prs/sglang/PR-5371.md) | apply fused moe gate in ds v3/r1 | 2025-04-14 | kernel-fusion | kernel-fusion, moe |
| [#5384](../sources/prs/sglang/PR-5384.md) | [PD Bug] fix  MLA get_contiguous_buf_infos error | 2025-04-14 |  | mla |
| [#5336](../sources/prs/sglang/PR-5336.md) | fix: determine if flashinfer is installed | 2025-04-13 |  | gemm |
| [#5340](../sources/prs/sglang/PR-5340.md) | Fix DeepGEMM masked cannot be run on groups not being multiple or 4 | 2025-04-13 |  | gemm, moe |
| [#5310](../sources/prs/sglang/PR-5310.md) | fix: use deepgemm only on hopper | 2025-04-12 |  | fp8, gemm, quantization |
| [#5318](../sources/prs/sglang/PR-5318.md) | Add Speculative Decoding Eagle3 topk > 1 | 2025-04-12 |  | attention |
| [#5331](../sources/prs/sglang/PR-5331.md) | fix: solve cu118 issue for cutlass mla | 2025-04-12 |  | attention, mla |
| [#5263](../sources/prs/sglang/PR-5263.md) | [Fix] Turn off DeepGEMM by default | 2025-04-11 |  | fp8, gemm, quantization |
| [#5281](../sources/prs/sglang/PR-5281.md) | [1/2] Add FP8 Blockscale MoE CUTLASS kernel for Blackwell | 2025-04-11 |  | fp8, moe |
| [#5176](../sources/prs/sglang/PR-5176.md) | feat: add DeepGEMM build warning | 2025-04-09 |  | fp8, gemm, quantization |
| [#5210](../sources/prs/sglang/PR-5210.md) | feat: use fa3 mla by default on hopper | 2025-04-09 |  | attention, mla |
| [#5150](../sources/prs/sglang/PR-5150.md) | Add optimized native kernels in sgl-kernel | 2025-04-08 |  | decode, gemm, moe |
| [#5113](../sources/prs/sglang/PR-5113.md) | Support MHA with chunked prefix cache for DeepSeek chunked prefill | 2025-04-07 |  | attention, prefill |
| [#5142](../sources/prs/sglang/PR-5142.md) | Blackwell Cutlass MLA kernel | 2025-04-07 |  | attention, mla |
| [#5086](../sources/prs/sglang/PR-5086.md) | reduce moe_align_block_size_kernel small batch mode overhead | 2025-04-05 | kernel-fusion | kernel-fusion, moe |
| [#5068](../sources/prs/sglang/PR-5068.md) | [Fix] DeepEP Compatibility with Low Latency | 2025-04-04 |  | moe |
| [#5074](../sources/prs/sglang/PR-5074.md) | support sgl-kernel on blackwell | 2025-04-04 |  | gemm |
| [#5011](../sources/prs/sglang/PR-5011.md) | update cutlass tag | 2025-04-03 |  | gemm |
| [#5030](../sources/prs/sglang/PR-5030.md) | fix deepgemm as well | 2025-04-03 |  | gemm |
| [#4953](../sources/prs/sglang/PR-4953.md) | [Build] Fix cuda12.8 build error in nvfp4_scaled_mm_kernels.cu | 2025-03-31 |  | fp4, gemm, nvfp4 |
| [#4918](../sources/prs/sglang/PR-4918.md) | Add DeepSeek V3/R1 shared experts fusion | 2025-03-30 | kernel-fusion | fp8, kernel-fusion, moe |
| [#4887](../sources/prs/sglang/PR-4887.md) | Feat/support encoder model (like bert) | 2025-03-29 |  | attention |
| [#4836](../sources/prs/sglang/PR-4836.md) | Introduce moe_dense_tp_size to fix dense layer errors in DeepSeek V3 + 4x8xH100 | 2025-03-28 |  | moe |
| [#4864](../sources/prs/sglang/PR-4864.md) | [Feat] support deepgemm for cmake | 2025-03-28 |  | gemm |
| [#4767](../sources/prs/sglang/PR-4767.md) | [Feature] Support DeepEP Low Latency | 2025-03-25 |  | moe |
| [#4770](../sources/prs/sglang/PR-4770.md) | Support (1 <= dp < tp) in the dp attention in DeepEP | 2025-03-25 |  | attention, moe |
| [#4706](../sources/prs/sglang/PR-4706.md) | support cmake for sgl-kernel | 2025-03-24 |  | attention, decode, gemm |
| [#4686](../sources/prs/sglang/PR-4686.md) | Fix loading KV quantization scale; Enable modelopt kv cache | 2025-03-23 |  | attention, gemm, moe |
| [#4693](../sources/prs/sglang/PR-4693.md) | [Model] Adding Qwen3 and Qwen3MoE | 2025-03-23 |  | attention, moe |
| [#4643](../sources/prs/sglang/PR-4643.md) | Optimize Permute Kernel in DeepEP | 2025-03-21 |  | moe |
| [#4613](../sources/prs/sglang/PR-4613.md) | Set deepgemm to the default value in the hopper architecture. | 2025-03-20 |  | fp8, gemm, quantization |
| [#4577](../sources/prs/sglang/PR-4577.md) | avoid cudaStreamSynchronize in DeepSeekV2AttentionMLA | 2025-03-19 |  | attention, mla |
| [#4596](../sources/prs/sglang/PR-4596.md) | [quantization] fix channelwise conversion with scalar weight scale | 2025-03-19 |  | fp8, quantization |
| [#4530](../sources/prs/sglang/PR-4530.md) | Add deepseek style fused moe group gate selection kernel | 2025-03-18 | kernel-fusion | kernel-fusion, moe |
| [#4557](../sources/prs/sglang/PR-4557.md) | [Fix] Fix raw_bs bug when using flashinfer mla and eagle | 2025-03-18 |  | mla |
| [#4558](../sources/prs/sglang/PR-4558.md) | Support fp8 gemm for blackwell | 2025-03-18 |  | fp8, gemm |
| [#4510](../sources/prs/sglang/PR-4510.md) | [ROCm] fix dtype | 2025-03-17 |  | fp8, quantization |
| [#4515](../sources/prs/sglang/PR-4515.md) | Create col-major and tma-aligned x_scale for deep_gemm.gemm_fp8_fp8_bf16_nt | 2025-03-17 |  | fp8, gemm, quantization |
| [#4359](../sources/prs/sglang/PR-4359.md) | [FIX] fix incorrect output when enable both deepgemm and torch compile | 2025-03-13 |  | fp8, gemm, quantization |
| [#4284](../sources/prs/sglang/PR-4284.md) | update deepgemm | 2025-03-11 |  | gemm |
| [#4317](../sources/prs/sglang/PR-4317.md) | upgrade flashinfer 0.2.3 | 2025-03-11 |  | gemm |
| [#4165](../sources/prs/sglang/PR-4165.md) | DeepGemm integrate to gemm | 2025-03-10 | jit-compilation, fine-grained-quantization | gemm, jit-compilation |
| [#4272](../sources/prs/sglang/PR-4272.md) | add THIRDPARTYNOTICES for DeepGEMM | 2025-03-10 |  | gemm |
| [#4278](../sources/prs/sglang/PR-4278.md) | Support Blackwell Block Scale FP8 Gemm | 2025-03-10 |  | fp8, gemm |
| [#4230](../sources/prs/sglang/PR-4230.md) | Clean up fp8 support | 2025-03-09 |  | fp8, quantization |
| [#4231](../sources/prs/sglang/PR-4231.md) | fix per_token_group_quant_fp8 illegal memory when num_groups % 16 != 0 | 2025-03-09 |  | fp8, gemm |
| [#4232](../sources/prs/sglang/PR-4232.md) | [Feature] Integrate DeepEP into SGLang | 2025-03-09 |  | moe |
| [#4199](../sources/prs/sglang/PR-4199.md) | linear support deepgemm | 2025-03-08 |  | fp8, gemm, quantization |
| [#4215](../sources/prs/sglang/PR-4215.md) | Accelerate FP8 CUDA Kernel by 20-28% | 2025-03-08 | kernel-fusion | fp8, gemm, kernel-fusion |
| [#4068](../sources/prs/sglang/PR-4068.md) | Support overlapping two batches | 2025-03-04 |  | attention, gemm, quantization |
| [#4009](../sources/prs/sglang/PR-4009.md) | Hierarchical Caching supports MLA | 2025-03-03 |  | mla |
| [#4012](../sources/prs/sglang/PR-4012.md) | [Revision] Add fast decode plan for flashinfer mla  | 2025-03-03 |  | attention, decode, mla |
| [#3987](../sources/prs/sglang/PR-3987.md) | Add fast decode plan for flashinfer mla | 2025-03-02 |  | attention, decode, mla |
| [#3959](../sources/prs/sglang/PR-3959.md) | [tools] add fp8 max/min constant in utils | 2025-02-28 |  | fp8 |
| [#3934](../sources/prs/sglang/PR-3934.md) | upgrade flashinfer v0.2.2.post1 | 2025-02-27 |  | gemm |
| [#3888](../sources/prs/sglang/PR-3888.md) | [Feature] DeepSeek V3/R1 INT8 Quantization (channel-wise)  | 2025-02-26 | kernel-fusion | kernel-fusion, moe, quantization |
| [#3899](../sources/prs/sglang/PR-3899.md) | Support FP4 gemm (1/2) | 2025-02-26 |  | fp4, gemm, nvfp4 |
| [#3727](../sources/prs/sglang/PR-3727.md) | add control for cutlass fp8 blockwise gemm | 2025-02-20 |  | fp8, gemm, quantization |
| [#3730](../sources/prs/sglang/PR-3730.md) | Feature DeepSeek V3/R1 INT8 Quantization (block-wise) | 2025-02-20 | kernel-fusion | kernel-fusion, moe, quantization |
| [#3629](../sources/prs/sglang/PR-3629.md) | [Feature] Apply Cublas Grouped Gemm kernel | 2025-02-17 |  | gemm, grouped-gemm |
| [#3643](../sources/prs/sglang/PR-3643.md) | feat: support flashinfer mla with prefix cache | 2025-02-17 |  | attention, mla |
| [#3557](../sources/prs/sglang/PR-3557.md) | update flashinfer-python | 2025-02-14 |  | gemm |
| [#3550](../sources/prs/sglang/PR-3550.md) | feat: support flashinfer mla attention for deepseek v3 | 2025-02-13 |  | attention, mla |
| [#3529](../sources/prs/sglang/PR-3529.md) | integrate blockwise fp8 kernel | 2025-02-12 | fine-grained-quantization | fp8, fine-grained-quantization, moe |
| [#3372](../sources/prs/sglang/PR-3372.md) | fix undefined symbol cudaGetDriverEntryPointByVersion | 2025-02-07 |  | gemm |
| [#3267](../sources/prs/sglang/PR-3267.md) | support blockwise fp8 matmul kernel | 2025-02-03 |  | fp8, gemm, tma |
| [#3255](../sources/prs/sglang/PR-3255.md) | Tune paged attention parameters for AMD GPU. | 2025-02-01 |  | attention, decode |
| [#3216](../sources/prs/sglang/PR-3216.md) | add tensorrt_llm common and cutlass_extensions as 3rdparty | 2025-01-30 | epilogue-fusion, kernel-fusion, pipeline-stages | epilogue-fusion, fp8, gemm |
| [#3148](../sources/prs/sglang/PR-3148.md) | Apply sgl w8a8 fp8 kernel | 2025-01-26 |  | fp8, quantization |
| [#3037](../sources/prs/sglang/PR-3037.md) | Allow local cutlass directory to be used in sgl-kernel build | 2025-01-22 |  | gemm |
| [#3047](../sources/prs/sglang/PR-3047.md) | support w8a8 fp8 kernel with CUTLASS | 2025-01-22 |  | fp8, gemm |
| [#3051](../sources/prs/sglang/PR-3051.md) | sync the upstream updates of flashinfer | 2025-01-22 |  | gemm |
| [#3056](../sources/prs/sglang/PR-3056.md) | feat: integrate gemm_fp8 kernel into gemm | 2025-01-22 | fine-grained-quantization | fp8, gemm, moe |
| [#3033](../sources/prs/sglang/PR-3033.md) | feat: add flashinfer as 3rdparty and use rmsnorm as example | 2025-01-21 |  | gemm |
| [#3035](../sources/prs/sglang/PR-3035.md) | Support sm90 Int8 gemm | 2025-01-21 |  | gemm |
| [#2967](../sources/prs/sglang/PR-2967.md) | upgrade cutlass v3.7.0 | 2025-01-18 |  | gemm |
| [#2752](../sources/prs/sglang/PR-2752.md) | Support cutlass Int8 gemm | 2025-01-06 | epilogue-fusion | epilogue-fusion, gemm |

<a id="triton-langtriton"></a>
## triton-lang/triton
69 PRs

| PR | Title | Date | Techniques | Tags |
|-----|-------|------|------------|------|
| [#10988](../sources/prs/triton/PR-10988.md) | reland "[Blackwell] Fix incorrectly pipelined back-to-back MMAs sharing the same accumulator" | 2026-07-21 | pipeline-stages | pipeline-stages, tmem |
| [#10979](../sources/prs/triton/PR-10979.md) | [AMD][GLUON] Allow compact scales for fp `scaled_upcast' | 2026-07-20 |  | gemm |
| [#10960](../sources/prs/triton/PR-10960.md) | [Gluon] Remove low-level multi-CTA instructions | 2026-07-19 |  | mbarrier |
| [#10922](../sources/prs/triton/PR-10922.md) | [BACKEND] Emit .{release,acquire}.cluster in barrier ops | 2026-07-17 |  | 2sm-cooperative, block-scale, clc |
| [#10936](../sources/prs/triton/PR-10936.md) | [NVIDIA] Initial support for Rubin | 2026-07-17 | kernel-fusion | block-scale, kernel-fusion, moe |
| [#10911](../sources/prs/triton/PR-10911.md) | [AMD]Enable Partition Conflicts Tests for MXFP GEMM on GFX1250 | 2026-07-16 |  | gemm |
| [#10912](../sources/prs/triton/PR-10912.md) | [AMD]Add Benchmark Utility to MoE and MXFP GEMM Kernels for GFX1250 | 2026-07-16 |  | gemm, moe |
| [#10913](../sources/prs/triton/PR-10913.md) | [AMD]Support TDM Fusion and Split for MXFP GEMM Kernel on GFX1250 | 2026-07-16 | kernel-fusion | gemm, kernel-fusion |
| [#10898](../sources/prs/triton/PR-10898.md) | [BACKEND] Disable unaligned descriptor -> TMA conversion | 2026-07-15 | pipeline-stages | pipeline-stages, tma |
| [#10873](../sources/prs/triton/PR-10873.md) | Avoid output TMA for Blackwell matmul accumulation | 2026-07-14 |  | tma |
| [#10879](../sources/prs/triton/PR-10879.md) | [CONSAN] Track compiler-owned shared scratch | 2026-07-14 |  | gemm |
| [#10858](../sources/prs/triton/PR-10858.md) | [Blackwell] Fix incorrectly pipelined back-to-back MMAs sharing the same accumulator | 2026-07-13 | pipeline-stages | pipeline-stages, tmem |
| [#10862](../sources/prs/triton/PR-10862.md) | [BACKEND] Allow to split TMEM along M | 2026-07-13 |  | mla, tmem |
| [#10864](../sources/prs/triton/PR-10864.md) | Scope ConSan cluster visibility to warp partitions | 2026-07-13 |  | gemm |
| [#10866](../sources/prs/triton/PR-10866.md) | [AMD] f16_gemm_gfx1250.py support multi-cta | 2026-07-13 |  | gemm |
| [#10839](../sources/prs/triton/PR-10839.md) | [NVIDIA] Enable batched TMA GEMM tests on sm120 | 2026-07-10 |  | gemm, tma |
| [#10846](../sources/prs/triton/PR-10846.md) | [AMD][NFC] Remove incorrect fp32->fp8 conversion via f16 intermediate value | 2026-07-10 |  | fp8 |
| [#10809](../sources/prs/triton/PR-10809.md) | [AMD] Rewrite gluon partition layout derivation | 2026-07-06 |  | gemm |
| [#10784](../sources/prs/triton/PR-10784.md) | Support local atomic scatter RMW in ConSan | 2026-07-02 |  | gemm |
| [#10747](../sources/prs/triton/PR-10747.md) | [AMD] Fix fp8 e4m3fnuz/e5m2fnuz NaN (0x80) silently decoding to a finite value | 2026-06-29 |  | fp8 |
| [#10736](../sources/prs/triton/PR-10736.md) | Support local gather and scatter in ConSan | 2026-06-26 |  | gemm |
| [#10685](../sources/prs/triton/PR-10685.md) | [AMD][gfx1250] TDM: make async_tdm_gather/scatter pure | 2026-06-20 | pipeline-stages | moe, pipeline-stages |
| [#10668](../sources/prs/triton/PR-10668.md) | [CONSAN] Implementing fence-async-proxy modeling | 2026-06-18 |  | gemm |
| [#10650](../sources/prs/triton/PR-10650.md) | Tensor scales in matmul. | 2026-06-17 |  | gemm |
| [#10623](../sources/prs/triton/PR-10623.md) | Remove obsolete empty pad workarounds | 2026-06-16 |  | gemm |
| [#10628](../sources/prs/triton/PR-10628.md) | [Gluon][AMD][gfx1250] Add explicit fused TDM copy ops | 2026-06-16 | kernel-fusion, pipeline-stages | gemm, kernel-fusion, pipeline-stages |
| [#10634](../sources/prs/triton/PR-10634.md) | float64 matmul support | 2026-06-16 |  | gemm |
| [#10638](../sources/prs/triton/PR-10638.md) | [AMD]Update MXFP GEMM Kernel on GFX1250 | 2026-06-16 |  | gemm |
| [#10594](../sources/prs/triton/PR-10594.md) | [AMD] Fix race in f16 streamK gemm kernel | 2026-06-13 |  | gemm |
| [#10598](../sources/prs/triton/PR-10598.md) | [AMD][gfx1250] TDM: make the async_tdm_copy ops pure | 2026-06-13 | pipeline-stages | pipeline-stages |
| [#10576](../sources/prs/triton/PR-10576.md) | [GSan] Misc bug-fixes | 2026-06-11 |  | gemm |
| [#10550](../sources/prs/triton/PR-10550.md) | [AMD][Gluon][Test] Fix test bug in MXFP GEMM sliceK | 2026-06-09 |  | gemm |
| [#10551](../sources/prs/triton/PR-10551.md) | [Blackwell] Add pass to combine TMEM load followed by row reduction for sm103+ | 2026-06-09 |  | tmem |
| [#10554](../sources/prs/triton/PR-10554.md) | Add direct layout storage shape queries | 2026-06-09 |  | gemm |
| [#10557](../sources/prs/triton/PR-10557.md) | [AMD][GLUON] Expose amdg.extract_slice | 2026-06-09 |  | gemm |
| [#10530](../sources/prs/triton/PR-10530.md) | [Blackwell] Decide TMEMCopy compatibility for block scales early  | 2026-06-08 |  | tmem |
| [#10519](../sources/prs/triton/PR-10519.md) | [BACKEND] Fix bankconflicts for MMAv2 with bitwidth=64 | 2026-06-06 | swizzling | swizzling |
| [#10473](../sources/prs/triton/PR-10473.md) | [FPSAN] Load TCGen shared operands directly | 2026-06-04 |  | gemm |
| [#10475](../sources/prs/triton/PR-10475.md) | Infer NVMMA SMEM attributes from linear layouts | 2026-06-04 |  | gemm |
| [#10458](../sources/prs/triton/PR-10458.md) | [AMD][BACKEND] Fix fp8 conversions on architectures not natively supporting fnuz | 2026-06-03 |  | fp8 |
| [#10463](../sources/prs/triton/PR-10463.md) | Handle empty matmul outputs with custom layouts | 2026-06-03 |  | gemm |
| [#10444](../sources/prs/triton/PR-10444.md) | [BACKEND] Use read-only waits for pipelined TMA stores | 2026-06-02 | pipeline-stages | gemm, grouped-gemm, pipeline-stages |
| [#10432](../sources/prs/triton/PR-10432.md) | Support zero-sized MX layout conversions | 2026-06-01 |  | gemm |
| [#10440](../sources/prs/triton/PR-10440.md) | [BACKEND] Allow to lower a cluster_barrier within a warp_specialized region | 2026-06-01 | warp-specialization | warp-specialization |
| [#10428](../sources/prs/triton/PR-10428.md) | Fix Hopper MXFP4 packed value padding | 2026-05-31 |  | fp4 |
| [#10422](../sources/prs/triton/PR-10422.md) | [KERNELS] Adjust warp size to avoid fp8 regression. | 2026-05-30 |  | fp8 |
| [#10415](../sources/prs/triton/PR-10415.md) | [TMA] Update TMAStoreWaitOp to wait for the memory write to complete | 2026-05-29 | pipeline-stages | pipeline-stages, tma |
| [#10399](../sources/prs/triton/PR-10399.md) | Fix gluon attention example for fp8 | 2026-05-28 |  | attention, fp8 |
| [#10401](../sources/prs/triton/PR-10401.md) | Make `triton_kernels.tensor.convert_layout` idempotent | 2026-05-28 |  | gemm |
| [#10385](../sources/prs/triton/PR-10385.md) | [KERNELS] fix hopper mxfp4 swizzle bug | 2026-05-27 | swizzling | fp4, swizzling |
| [#10386](../sources/prs/triton/PR-10386.md) | [kernels] change heuristic of smem calculation to be more accurate | 2026-05-27 |  | gemm |
| [#10357](../sources/prs/triton/PR-10357.md) | [Gluon] Add support for nv local_store_async | 2026-05-22 |  | gemm |
| [#10343](../sources/prs/triton/PR-10343.md) | Fix small-K swizzled MXFP4 matmul | 2026-05-21 | swizzling | fp4, swizzling |
| [#10348](../sources/prs/triton/PR-10348.md) | [AMD][GLUON] Lower local_atomic_scatter_rmw | 2026-05-21 |  | gemm |
| [#10330](../sources/prs/triton/PR-10330.md) | [FRONTEND] [FPSAN] Introduce `tl.expect_zero` primitive | 2026-05-17 |  | gemm |
| [#10324](../sources/prs/triton/PR-10324.md) | [BACKEND] Allow two_ctas=False barriers in TMA ops in a 2CTA kernel | 2026-05-15 |  | 2sm-cooperative, tma |
| [#10316](../sources/prs/triton/PR-10316.md) | Enable microscaled lhs with dense FP16/BF16 matmul weights | 2026-05-14 |  | gemm |
| [#10302](../sources/prs/triton/PR-10302.md) | [AMD][GLUON] Expose compute_padded_layout_cdna4 to Python | 2026-05-13 |  | gemm |
| [#10275](../sources/prs/triton/PR-10275.md) | Support zero-sized Hopper MX scale layouts | 2026-05-09 |  | gemm |
| [#10263](../sources/prs/triton/PR-10263.md) | [BACKEND] add TMEM barrier insertion pass | 2026-05-08 |  | mbarrier, tmem |
| [#10030](../sources/prs/triton/PR-10030.md) | [Nvidia][Gluon] Refactor convolution fprop kernel, add wgrad and dgrad kernels | 2026-04-21 | warp-specialization, pipeline-stages, persistent-kernel | tma, tcgen05, tmem |
| [#10040](../sources/prs/triton/PR-10040.md) | [Gluon] Expose TMA atomic ops | 2026-04-15 | shared-memory-optimization | tma, triton, shared-memory-optimization |
| [#9967](../sources/prs/triton/PR-9967.md) | [KERNELS] Fix tmem overflow for fp32 matmul. | 2026-04-09 | shared-memory-optimization | gemm, tmem |
| [#9393](../sources/prs/triton/PR-9393.md) | [Kernels] Enable persistent matmul for fp32 inputs | 2026-02-20 | persistent-kernel, pipeline-stages, shared-memory-optimization | gemm, persistent-kernel, pipeline-stages |
| [#9279](../sources/prs/triton/PR-9279.md) | [Kernels] Don't flatten persistent hopper mixed precision matmul | 2026-01-22 | persistent-kernel, epilogue-fusion | gemm, persistent-kernel, epilogue-fusion |
| [#9248](../sources/prs/triton/PR-9248.md) | [Kernels] Enable high occupancy persistent matmul | 2026-01-20 | persistent-kernel, shared-memory-optimization | gemm, persistent-kernel, shared-memory-optimization |
| [#6660](../sources/prs/triton/PR-6660.md) | [Pipeliner] Implement more sophisticated partitioning strategy for attention | 2025-05-08 | pipeline-stages, warp-specialization | attention, pipeline-stages, warp-specialization |
| [#6394](../sources/prs/triton/PR-6394.md) | [NVIDIA] Enable Programmatic Dependent Launch in Triton | 2025-04-29 | persistent-kernel | persistent-kernel, gemm |
| [#6299](../sources/prs/triton/PR-6299.md) | [TritonGPU] Improve persistent matmul warp specialization | 2025-03-27 | warp-specialization, persistent-kernel, pipeline-stages | gemm, persistent-kernel, warp-specialization |

<a id="vllm-projectvllm"></a>
## vllm-project/vllm
1360 PRs

| PR | Title | Date | Techniques | Tags |
|-----|-------|------|------------|------|
| [#49395](../sources/prs/vllm/PR-49395.md) | [XPU] WA of topk_softmax arg mismatch on XPU | 2026-07-22 |  | tma |
| [#49427](../sources/prs/vllm/PR-49427.md) | [Bugfix] Restore `gather_and_maybe_dequant_cache` OOB guard | 2026-07-22 |  | gemm |
| [#49467](../sources/prs/vllm/PR-49467.md) | [Bugfix] Fix DeepGEMM warmup when using `FlashInferFp8DeepGEMMDynamicBlockScaledKernel` | 2026-07-22 |  | fp8, gemm |
| [#49487](../sources/prs/vllm/PR-49487.md) | [Performance][Model] Avoid transient Inkling result allocations (performance, and OOM prevention on smaller memory configurations) | 2026-07-22 |  | moe |
| [#49489](../sources/prs/vllm/PR-49489.md) | [Bugfix] Make shared NVFP4 MoE scales writable | 2026-07-22 |  | fp4, moe, nvfp4 |
| [#49492](../sources/prs/vllm/PR-49492.md) | Add quantization label automation | 2026-07-22 |  | quantization |
| [#49294](../sources/prs/vllm/PR-49294.md) | [Bugfix][Attention] Ignore empty MLA context chunks during merge | 2026-07-21 |  | attention, fp4, mla |
| [#49177](../sources/prs/vllm/PR-49177.md) | Propagate Flash Attention cache configuration to Ray workers | 2026-07-20 |  | attention |
| [#49178](../sources/prs/vllm/PR-49178.md) | [Bugfix][SpecDecode] Scope MTP completeness checks outside bucketed updates | 2026-07-20 |  | decode, moe |
| [#49108](../sources/prs/vllm/PR-49108.md) | [Bugfix] Fix broken NVVM caused by CuteDSL 4.6.0 | 2026-07-19 |  | tcgen05 |
| [#49021](../sources/prs/vllm/PR-49021.md) | [Bugfix][CPU] Fix Clang OpenMP build on macOS | 2026-07-18 |  | gemm |
| [#48917](../sources/prs/vllm/PR-48917.md) | [Bugfix] Propagate quant_config to LFM2 ShortConv projections | 2026-07-17 |  | moe |
| [#48952](../sources/prs/vllm/PR-48952.md) | Cosmos3 FP8 ModelOpt/Diffusers remapping | 2026-07-17 |  | fp8 |
| [#48957](../sources/prs/vllm/PR-48957.md) | [DSv4 Perf] Skip empty c128 kernel launch, around 2x kernel performance improvement. | 2026-07-17 |  | gemm |
| [#48979](../sources/prs/vllm/PR-48979.md) | skip cudagraph/DP padding in topk | 2026-07-17 | kernel-fusion | kernel-fusion, moe, tma |
| [#48990](../sources/prs/vllm/PR-48990.md) | [Model] Use standard ModelOpt config for Inkling NVFP4 | 2026-07-17 |  | fp4, moe, nvfp4 |
| [#48993](../sources/prs/vllm/PR-48993.md) | [Core][DSV4] Compact MXFP4 indexer KV cache and packed group overlays | 2026-07-17 |  | attention, fp4 |
| [#48797](../sources/prs/vllm/PR-48797.md) | [Kernel][Helion] Disable warp specialization in rms_norm_per_block_quant B200 configs | 2026-07-16 |  | gemm |
| [#48799](../sources/prs/vllm/PR-48799.md) | [Model] Add Inkling model support [1/N] | 2026-07-16 |  | attention, fp4, gemm |
| [#48822](../sources/prs/vllm/PR-48822.md) | [Model] Add PW CUDA graph support for Inkling [2/N] | 2026-07-16 |  | attention, decode |
| [#48849](../sources/prs/vllm/PR-48849.md) | Fix: Restore data_parallel_size > 1 for use_sequence_parallel_moe | 2026-07-16 |  | moe |
| [#48855](../sources/prs/vllm/PR-48855.md) | [Bugfix] Enable FlashAttention MLA prefill for Mistral Small 4 head dims | 2026-07-16 |  | attention, mla, prefill |
| [#48858](../sources/prs/vllm/PR-48858.md) | [Model] Add Hopper FA4 relative attention for Inkling | 2026-07-16 |  | attention |
| [#48868](../sources/prs/vllm/PR-48868.md) | [Helion] Fix degenerate scale_ub in kernel input generators | 2026-07-16 |  | fp8 |
| [#48884](../sources/prs/vllm/PR-48884.md) | [Model] Add Inkling LoRA support [4/N] | 2026-07-16 | kernel-fusion | kernel-fusion, moe |
| [#48717](../sources/prs/vllm/PR-48717.md) | [Misc][Nixl] Unify `_logical_to_remote_kernel_block_ids` | 2026-07-15 |  | gemm |
| [#48759](../sources/prs/vllm/PR-48759.md) | [LoRA] Optimize TrtLlmLoRAExperts | 2026-07-15 | kernel-fusion | kernel-fusion, moe |
| [#48785](../sources/prs/vllm/PR-48785.md) | [Bugfix] Fix activation quantization dispatch for WNA4Int/WNA8Int | 2026-07-15 |  | quantization |
| [#48788](../sources/prs/vllm/PR-48788.md) | [ROCm][Perf][DSV4] Improve sparse decode reduction occupancy on gfx950 | 2026-07-15 |  | attention, decode, mla |
| [#48563](../sources/prs/vllm/PR-48563.md) | [Bugfix][Gemma4] Fix ModelOpt mixed-precision MoE config mapping | 2026-07-14 |  | gemm, moe, quantization |
| [#48582](../sources/prs/vllm/PR-48582.md) | [M3] Improve indexer for long-context decode (sm100) | 2026-07-14 |  | attention, decode |
| [#48632](../sources/prs/vllm/PR-48632.md) | [LoRA][1/N] Integrate flashinfer MoE LoRA for BF16 model | 2026-07-14 | kernel-fusion | kernel-fusion, moe, quantization |
| [#48642](../sources/prs/vllm/PR-48642.md) | [Bugfix] Sparse MLA: enable fp8_ds_mla dense prefill | 2026-07-14 |  | attention, fp8, mla |
| [#48660](../sources/prs/vllm/PR-48660.md) | [Perf] Optimize dsv4 routing using specialized kernel, 2.94% E2E TPOT improvement | 2026-07-14 | kernel-fusion | kernel-fusion, moe |
| [#48451](../sources/prs/vllm/PR-48451.md) | [feature]Add int4 quantization support for emulation moe backend | 2026-07-13 | kernel-fusion | kernel-fusion, moe, quantization |
| [#48507](../sources/prs/vllm/PR-48507.md) | [Bug][Quantization] Fix humming is_layer_skipped for compressed-tensors "re:" ignore entries | 2026-07-13 |  | quantization |
| [#48512](../sources/prs/vllm/PR-48512.md) | [Kernel][Helion] Add Helion kernel benchmark script | 2026-07-13 |  | fp8 |
| [#48519](../sources/prs/vllm/PR-48519.md) | [ROCm][Perf] Optimize sparse attention prefill kernel for DeepSeek-V4 | 2026-07-13 |  | attention, mla, prefill |
| [#48520](../sources/prs/vllm/PR-48520.md) | [Bugfix] Make MLA+SWA check the layer's backend, not the model config | 2026-07-13 |  | attention, mla |
| [#48538](../sources/prs/vllm/PR-48538.md) | [Quant] Add `nvfp4_per_token` online MoE quantization | 2026-07-13 | kernel-fusion | fp4, kernel-fusion, moe |
| [#48379](../sources/prs/vllm/PR-48379.md) | [Bugfix] Set kv_quant_mode on the generic MLA KV-cache spec | 2026-07-12 |  | attention, mla |
| [#48385](../sources/prs/vllm/PR-48385.md) | add pad-aware reduce path | 2026-07-12 | kernel-fusion | kernel-fusion, moe |
| [#48417](../sources/prs/vllm/PR-48417.md) | [Performance] Use CuTe-DSL for FlashInfer MXFP4 quantization | 2026-07-12 |  | fp4, quantization |
| [#48429](../sources/prs/vllm/PR-48429.md) | [BugFix] Restore full tokens for Qwen MTP When MoE SP | 2026-07-12 |  | moe |
| [#48334](../sources/prs/vllm/PR-48334.md) | [XPU] FP8 o_proj with fp8_bmm and load-time scale transpose | 2026-07-11 | kernel-fusion | fp8, kernel-fusion |
| [#48350](../sources/prs/vllm/PR-48350.md) | [Model] Optimize Qwen3.5 on H20 | 2026-07-11 | kernel-fusion | kernel-fusion, moe |
| [#48251](../sources/prs/vllm/PR-48251.md) | [Bugfix][Attention] Preserve post-load tensors across weight reloads | 2026-07-10 |  | attention, mla |
| [#48256](../sources/prs/vllm/PR-48256.md) | [Bugfix][KV Cache] Don't route uniform-page-size MLA+SWA models into DeepseekV4 packing | 2026-07-10 |  | mla |
| [#48264](../sources/prs/vllm/PR-48264.md) | [Kernel][Helion] Helion kernel lazy registration | 2026-07-10 |  | gemm |
| [#48268](../sources/prs/vllm/PR-48268.md) | Add VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS and skip CuTeDSL fp4_gemm autotuning by default | 2026-07-10 |  | fp4, gemm |
| [#48287](../sources/prs/vllm/PR-48287.md) | add pad-aware swiglu limit kernel | 2026-07-10 | kernel-fusion | kernel-fusion, moe |
| [#48068](../sources/prs/vllm/PR-48068.md) | [Bugfix][Spec Decode] Fix eagle3 first-layer qkv_proj prefix for quantized drafts | 2026-07-09 |  | decode, quantization |
| [#48125](../sources/prs/vllm/PR-48125.md) | [PD][Bugfix] Fix validation of cache shape for attn backends enforcing different `kernel_block_size` | 2026-07-09 |  | gemm |
| [#48135](../sources/prs/vllm/PR-48135.md) | [Bugfix] Preserve tensor causal metadata for grouped attention | 2026-07-09 |  | attention |
| [#48143](../sources/prs/vllm/PR-48143.md) | [Perf] Optimize `clamp` to `clamp_` | 2026-07-09 |  | attention, decode, mla |
| [#48144](../sources/prs/vllm/PR-48144.md) | update marlin M size for EP | 2026-07-09 | kernel-fusion | kernel-fusion, moe |
| [#48158](../sources/prs/vllm/PR-48158.md) | [Refactor] Remove unused rocm kernel `combine_topk_swa_indices_ragged` | 2026-07-09 |  | attention |
| [#48167](../sources/prs/vllm/PR-48167.md) | [Bugfix] Fix FlashInfer non-causal draft attention (DFlash/DSpark) on Blackwell | 2026-07-09 |  | attention, decode |
| [#47946](../sources/prs/vllm/PR-47946.md) | [Test] Skip DeepEP MoE layer tests without P2P access | 2026-07-08 |  | moe |
| [#47962](../sources/prs/vllm/PR-47962.md) | [XPU] [Fusion passes] Disable fuse_rope_kvcache_cat_mla & qk_norm_rope_ fusion on XPU | 2026-07-08 | kernel-fusion | kernel-fusion, mla |
| [#47973](../sources/prs/vllm/PR-47973.md) | BF16x3 router GEMM | 2026-07-08 | kernel-fusion | gemm, kernel-fusion, moe |
| [#47984](../sources/prs/vllm/PR-47984.md) | [ROCm][MiniMax-M3][Spec Decode] Support speculative decode with AITER sparse PA | 2026-07-08 |  | attention, decode, sparse-attention |
| [#48011](../sources/prs/vllm/PR-48011.md) | [Attention] Make sliding-window support an explicit backend capability | 2026-07-08 |  | attention |
| [#48012](../sources/prs/vllm/PR-48012.md) | [Attention] Allow selecting a different attention backend per KV-cache group | 2026-07-08 |  | attention |
| [#48045](../sources/prs/vllm/PR-48045.md) | [Bugfix] Fix FlashMLA dense fp8 metadata crash (num_sm_parts clamp) | 2026-07-08 |  | fp8, mla |
| [#48046](../sources/prs/vllm/PR-48046.md) | [Bugfix] Use int8 workspace for FlashInfer MLA decode | 2026-07-08 |  | attention, decode, mla |
| [#47801](../sources/prs/vllm/PR-47801.md) | [Bugfix][DCP] Cast LSE to fp32 in a2a combine to fix bf16 bitcast crash | 2026-07-07 |  | attention |
| [#47851](../sources/prs/vllm/PR-47851.md) | [Quantization] Bound peak memory when repacking FP4 MoE weights for Marlin | 2026-07-07 |  | fp4, moe, quantization |
| [#47857](../sources/prs/vllm/PR-47857.md) | [Model] Add LongCat-Flash-Lite (n-gram embedding) | 2026-07-07 |  | gemm |
| [#47881](../sources/prs/vllm/PR-47881.md) | [Feature] Migrate moe sp support to non-torch compiled path for GLM5.2 | 2026-07-07 | kernel-fusion | fp8, kernel-fusion, moe |
| [#47884](../sources/prs/vllm/PR-47884.md) | [Bug] Fix Batched DeepGEMM | 2026-07-07 | kernel-fusion | gemm, kernel-fusion, moe |
| [#47908](../sources/prs/vllm/PR-47908.md) | [Bugfix] Allow non-contiguous query in FlashInfer FP8 query quantization | 2026-07-07 |  | attention, fp8, quantization |
| [#47910](../sources/prs/vllm/PR-47910.md) | [Bugfix] Patch Hopper MXFP4 OOB scales reads leading to NaN | 2026-07-07 |  | fp4, quantization |
| [#47912](../sources/prs/vllm/PR-47912.md) | [ROCm] Fix pooling startup workspace lock | 2026-07-07 |  | attention, prefill |
| [#47914](../sources/prs/vllm/PR-47914.md) | [Spec Decode] Support hybrid (SWA + full attention) DFlash drafters | 2026-07-07 |  | attention, decode |
| [#47671](../sources/prs/vllm/PR-47671.md) | [Bugfix] Fix int32 overflow in triton_decode_attention page offsets | 2026-07-06 |  | attention, decode |
| [#47680](../sources/prs/vllm/PR-47680.md) | [Bugfix][V1/V2] Fix prompt_logprobs to respect logprobs_mode | 2026-07-06 | kernel-fusion | decode, gemm, kernel-fusion |
| [#47688](../sources/prs/vllm/PR-47688.md) | [XPU] Route mm_prefix models to Triton attention backend | 2026-07-06 |  | attention |
| [#47716](../sources/prs/vllm/PR-47716.md) | [Bugfix]Fix DeepSeek-V4 fp8_ds_mla KV cache reshape | 2026-07-06 |  | attention, fp8, mla |
| [#47718](../sources/prs/vllm/PR-47718.md) | [ROCm][Perf] DSv4 two-stage compressor kernel for HCA prefill | 2026-07-06 | kernel-fusion | kernel-fusion, prefill |
| [#47766](../sources/prs/vllm/PR-47766.md) | [ROCm][Bugfix] Key sparse-MLA persistent metadata on per-request context lengths | 2026-07-06 | persistent-kernel | attention, mla, persistent-kernel |
| [#47770](../sources/prs/vllm/PR-47770.md) | [ROCm][BugFix] Triton W4A16 handling for GPTQ/AutoGPTQ qzeros layout  | 2026-07-06 |  | quantization |
| [#47780](../sources/prs/vllm/PR-47780.md) | [Bugfix] [Quantization] Fix loading for CT DSV2 | 2026-07-06 |  | quantization |
| [#47781](../sources/prs/vllm/PR-47781.md) | [CI/Build][BugFix][The Rock] Fix get_ssm_device_name to return sanitized, usable filename | 2026-07-06 | kernel-fusion | fp8, kernel-fusion, moe |
| [#47785](../sources/prs/vllm/PR-47785.md) | handle topk_ids padding in align sum kernel | 2026-07-06 |  | moe |
| [#47641](../sources/prs/vllm/PR-47641.md) | [Hardware][CPU] Enable granite-4 model on cpu | 2026-07-05 |  | attention |
| [#47493](../sources/prs/vllm/PR-47493.md) | [Bugfix] DSV4 TP16 garbage output | 2026-07-03 |  | attention, mla |
| [#47502](../sources/prs/vllm/PR-47502.md) | [Minimax-M3] Using tok_sparse_select from MSA instead of triton kernels | 2026-07-03 |  | attention, flash-attention, sparse-attention |
| [#47516](../sources/prs/vllm/PR-47516.md) | [XPU][UT]fix _POSSIBLE_KERNELS error on XPU | 2026-07-03 |  | gemm |
| [#47521](../sources/prs/vllm/PR-47521.md) | [Quantization][INC][ARK] Support INT2 XPU WOQ Linear | 2026-07-03 |  | quantization |
| [#47532](../sources/prs/vllm/PR-47532.md) | [Bugfix][CPU][RISC-V] Fix VLEN detection for RVV attention path | 2026-07-03 |  | attention |
| [#47546](../sources/prs/vllm/PR-47546.md) | [Perf][3/N] Expand Triton kernel warmup coverage, Qwen | 2026-07-03 |  | gemm |
| [#47567](../sources/prs/vllm/PR-47567.md) | [ROCm] Disable persistent sparse-MLA kernel for chunked-prefill continuations | 2026-07-03 | persistent-kernel | attention, mla, persistent-kernel |
| [#47568](../sources/prs/vllm/PR-47568.md) | Added sliding window attention support for qwen-eagle3 architecture | 2026-07-03 |  | attention |
| [#47574](../sources/prs/vllm/PR-47574.md) | [Bugfix] Zero new KV blocks for quantized + sliding-window hybrid caches | 2026-07-03 |  | quantization |
| [#47383](../sources/prs/vllm/PR-47383.md) | [Bugfix][Model Runner V2][Spec Decode] Fix int32 offset overflow in block verification kernels | 2026-07-02 |  | decode |
| [#47404](../sources/prs/vllm/PR-47404.md) | [ROCm] Synchronize sparse MLA metadata before graph replay | 2026-07-02 |  | attention, mla |
| [#47408](../sources/prs/vllm/PR-47408.md) | [Kernel]  Applies routed_scaling_factor internally | 2026-07-02 | kernel-fusion | kernel-fusion, moe, tma |
| [#47427](../sources/prs/vllm/PR-47427.md) | [MoE] FI autotuning: max bucket = max token count [e.g. `DP_size*MNBT`] | 2026-07-02 | kernel-fusion | fp4, fp8, kernel-fusion |
| [#47433](../sources/prs/vllm/PR-47433.md) | [Attention Backend] HPC_ATTN backend support mtp and dynamic scheduled attention | 2026-07-02 |  | attention |
| [#47437](../sources/prs/vllm/PR-47437.md) | [Bugfix] Fix pooled Whisper encoder sliding-window kernel size | 2026-07-02 |  | gemm |
| [#47445](../sources/prs/vllm/PR-47445.md) | [BugFix] Fix ModelOpt quantization inference for fused siblings | 2026-07-02 | kernel-fusion | kernel-fusion, quantization |
| [#47451](../sources/prs/vllm/PR-47451.md) | [Feat][Perf] Add new warmup infrastructure for JITs | 2026-07-02 |  | attention, mla, prefill |
| [#47463](../sources/prs/vllm/PR-47463.md) | [Perf] Optimize `fused_topk_bias` for DSv4, 1.5~2x kernel performance improvement | 2026-07-02 | kernel-fusion | kernel-fusion, moe |
| [#47474](../sources/prs/vllm/PR-47474.md) | [Perf] Cache `token_to_req_indices` for dsv4, 5x~6x kernel performance improvement | 2026-07-02 |  | attention, mla |
| [#47485](../sources/prs/vllm/PR-47485.md) | [BugFix] Derive FlashInfer Q dtype from resolved per-group builder state | 2026-07-02 |  | attention |
| [#47229](../sources/prs/vllm/PR-47229.md) | [DSV4] Better MXFP8 quantization kernel | 2026-07-01 |  | fp8, quantization |
| [#47268](../sources/prs/vllm/PR-47268.md) | Fixes non-coalesced HBM access in marlin_int4_fp8_preprocess_kernel_awq | 2026-07-01 |  | fp8, quantization |
| [#47276](../sources/prs/vllm/PR-47276.md) | [Bugfix][ROCm] Fix memory access fault in AITER MLA backend for DPA+FP8 KV  | 2026-07-01 |  | attention, fp8, mla |
| [#47285](../sources/prs/vllm/PR-47285.md) | [Model Runner V2][Perf] Warm up GLM-5.2 DSA indexer prefill metadata kernel | 2026-07-01 |  | mla, prefill |
| [#47287](../sources/prs/vllm/PR-47287.md) | [ROCm][MiniMax-M3] Add AITER sparse paged attention | 2026-07-01 | kernel-fusion | attention, fp8, kernel-fusion |
| [#47290](../sources/prs/vllm/PR-47290.md) | [BugFix for #47070] Lora should not be with MoE SP | 2026-07-01 |  | moe |
| [#47305](../sources/prs/vllm/PR-47305.md) | [Bugfix] Don't read KV cache past `seq_len` in triton paged attn kernels | 2026-07-01 |  | attention, decode, prefill |
| [#47309](../sources/prs/vllm/PR-47309.md) | [BugFix][MLA] Support kv_cache_dtype_skip_layers for MLA attention | 2026-07-01 |  | attention, mla |
| [#47314](../sources/prs/vllm/PR-47314.md) | [BugFix] Fix packed HND KV cache reshape for FlashAttention | 2026-07-01 |  | attention |
| [#47318](../sources/prs/vllm/PR-47318.md) | [BugFix] Fix ModelOpt mixed-precision quantization for sparse `quantized_layers` configs. | 2026-07-01 |  | quantization |
| [#47321](../sources/prs/vllm/PR-47321.md) | [HARDWARE][POWER]  optimize math functions of VSX power | 2026-07-01 |  | gemm |
| [#47332](../sources/prs/vllm/PR-47332.md) | [Bugfix][Gemma4] Fix FA4 mm_prefix mask: add sliding window and absolute q_idx | 2026-07-01 |  | attention, gemm |
| [#47090](../sources/prs/vllm/PR-47090.md) | [GLM5] Support FlashMLA FP8 KV cache (Hopper & Blackwell) | 2026-06-30 | kernel-fusion | attention, fp8, kernel-fusion |
| [#47091](../sources/prs/vllm/PR-47091.md) | [Bugfix] [Gemma4] Fix Gemma4 MTP draft model layers ignoring quant_config | 2026-06-30 |  | gemm |
| [#47102](../sources/prs/vllm/PR-47102.md) | Add Triton Backend for Unlimited-OCR R-SWA | 2026-06-30 |  | attention |
| [#47122](../sources/prs/vllm/PR-47122.md) | [XPU] [MoE] add quant input when prepare for fusedmoe | 2026-06-30 | kernel-fusion | kernel-fusion, moe |
| [#47128](../sources/prs/vllm/PR-47128.md) | [ROCm] [PyTorch] Move to stable abi since ROCm upgraded to torch 2.11 | 2026-06-30 |  | gemm |
| [#47140](../sources/prs/vllm/PR-47140.md) | [Platform] Replace `torch.cuda.Event` with `torch.Event` | 2026-06-30 | kernel-fusion | attention, decode, kernel-fusion |
| [#47154](../sources/prs/vllm/PR-47154.md) | [Bugfix] compressed-tensors: allow int8 grouped WNA16 MoE on Marlin | 2026-06-30 |  | moe, quantization |
| [#47156](../sources/prs/vllm/PR-47156.md) | [Perf][MoE] Write FlashInfer combine into final output | 2026-06-30 | kernel-fusion | kernel-fusion, moe |
| [#47187](../sources/prs/vllm/PR-47187.md) | Make the Transformers modeling backend as fast as native vLLM | 2026-06-30 |  | moe |
| [#47201](../sources/prs/vllm/PR-47201.md) | [ROCm][Bugfix] Convert ModelOpt FP8 per-channel weights to e4m3fnuz on MI300/MI325 | 2026-06-30 |  | fp8, quantization |
| [#47209](../sources/prs/vllm/PR-47209.md) | [ROCm][Bugfix] Fix Triton "out of resource: shared memory" Error In One-Shot LoRA MoE | 2026-06-30 | kernel-fusion | kernel-fusion, moe |
| [#47217](../sources/prs/vllm/PR-47217.md) | [Bugfix][Gemma4] Keep image bidirectional attention within the sliding window | 2026-06-30 |  | attention, gemm |
| [#47220](../sources/prs/vllm/PR-47220.md) | [AMD][EPLB] Enable EPLB for Quark OCP MXFP4 MoE | 2026-06-30 |  | fp4, moe, quantization |
| [#46997](../sources/prs/vllm/PR-46997.md) | [Bugfix][ROCm][MLA] Pass q/kv dtypes to get_mla_metadata_v1 in FP8 decode | 2026-06-29 |  | attention, decode, fp8 |
| [#47006](../sources/prs/vllm/PR-47006.md) | [Perf][Qwen] Replace MOE all-reduce with reduce-scatter | 2026-06-29 |  | moe |
| [#47031](../sources/prs/vllm/PR-47031.md) | [Bugfix] Fix GraniteMoeShared weight loading broken by #41184 | 2026-06-29 |  | moe |
| [#47035](../sources/prs/vllm/PR-47035.md) | [ROCm] Fix encoder-decoder cross-attention KV layout aliasing | 2026-06-29 |  | attention, decode |
| [#47039](../sources/prs/vllm/PR-47039.md) | [Bugfix] Restore part of bugfix #42650 after accidental deletion in #43241 | 2026-06-29 |  | attention |
| [#47058](../sources/prs/vllm/PR-47058.md) | Remove more unnecessary `load_weights` methods | 2026-06-29 | kernel-fusion | gemm, kernel-fusion, moe |
| [#47060](../sources/prs/vllm/PR-47060.md) | [Attention] Mirror Triton KV dtype checks in MLA | 2026-06-29 |  | attention, mla |
| [#47074](../sources/prs/vllm/PR-47074.md) | [Bugfix] Use larger workspace size for Flashinfer MLA LSE | 2026-06-29 |  | attention, mla |
| [#47079](../sources/prs/vllm/PR-47079.md) | [Bugfix][MLA] Fix LSE log-base mismatch in DCP + FlashInfer MLA decode | 2026-06-29 |  | attention, decode, mla |
| [#47083](../sources/prs/vllm/PR-47083.md) | [Bug] Fix sparse attention issue for GLM5.2 non-torch compile path | 2026-06-29 |  | attention |
| [#46942](../sources/prs/vllm/PR-46942.md) | [MRV2] Enable mm prefix bidi attention support on MRV2 | 2026-06-28 |  | attention |
| [#46944](../sources/prs/vllm/PR-46944.md) | [ROCm][Test] Fix test_per_token_group_quant_fp8 tolerance for 1-ULP FP8 rounding on gfx950 | 2026-06-28 |  | fp8, quantization |
| [#46915](../sources/prs/vllm/PR-46915.md) | [Bugfix][Test] Fix test_flashinfer_cutlass_mxfp4_fused_moe on sm90 (stale weight/scale interleave) | 2026-06-27 | kernel-fusion | fp4, kernel-fusion, moe |
| [#46807](../sources/prs/vllm/PR-46807.md) | PD disagg with Mooncake Connector: GDN support (Qwen3.5) and MLA support (Deepseek-V4-Flash) | 2026-06-26 |  | mla |
| [#46808](../sources/prs/vllm/PR-46808.md) | [GLM-5] Add DSV3.2/GLM5 to `vllm/models/` | 2026-06-26 |  | attention |
| [#46818](../sources/prs/vllm/PR-46818.md) | [Misc] Fix incorrect layer type annotation in Fp8LinearMethod | 2026-06-26 |  | fp8, quantization |
| [#46819](../sources/prs/vllm/PR-46819.md) | [Kernel] Triton MLA logits workspace | 2026-06-26 |  | attention, mla |
| [#46820](../sources/prs/vllm/PR-46820.md) | Fix Transformers backend FP8 MoE and remove some boilerplate | 2026-06-26 |  | fp8, gemm, moe |
| [#46832](../sources/prs/vllm/PR-46832.md) | [ROCm][DSv3.2][Perf] Cap sparse MLA decode KV-splits with a work-per-split heuristic | 2026-06-26 |  | attention, decode, mla |
| [#46838](../sources/prs/vllm/PR-46838.md) | [Bugfix][Kernel] Correct FlashInfer CUTLASS MoE tuning token bound | 2026-06-26 | kernel-fusion | kernel-fusion, moe |
| [#46842](../sources/prs/vllm/PR-46842.md) | [Refactor] Remove dead minimax allreduce rms kernel | 2026-06-26 |  | gemm |
| [#46860](../sources/prs/vllm/PR-46860.md) | [Bugfix][Quantization] Fix W8A8 int-quantized scheme selection regression | 2026-06-26 |  | quantization |
| [#46862](../sources/prs/vllm/PR-46862.md) | [GLM5.2 Perf] `fused_indexer_q_rope_quant` triton kernel, 1.9% ~ 3.3% E2E Throughput improvement. | 2026-06-26 | kernel-fusion | kernel-fusion |
| [#46876](../sources/prs/vllm/PR-46876.md) | [GLM5] Implement op fusion for GLM5/DSV3.2 | 2026-06-26 | kernel-fusion | attention, kernel-fusion, moe |
| [#46880](../sources/prs/vllm/PR-46880.md) | [Bugfix][NVFP4 MoE] Pad gated intermediate to 64 for FlashInfer TRT-LLM shuffle (M%128) | 2026-06-26 |  | fp4, moe, nvfp4 |
| [#46730](../sources/prs/vllm/PR-46730.md) | [ROCm][Perf][Bugfix] DSv4 indexer: use platform FP8 dtype (fnuz) for Q-quant on gfx942 | 2026-06-25 | kernel-fusion | fp8, kernel-fusion |
| [#46739](../sources/prs/vllm/PR-46739.md) | [CPU][BugFix] Multiple fixes to w4a8_int8 CPU MoE path | 2026-06-25 | kernel-fusion | kernel-fusion, moe, quantization |
| [#46750](../sources/prs/vllm/PR-46750.md) | [Perf][2/N] Expand Triton kernel warmup coverage, Qwen | 2026-06-25 |  | gemm |
| [#46753](../sources/prs/vllm/PR-46753.md) | [ModelRunner V2] Fix cross-attention block table sizing | 2026-06-25 |  | attention |
| [#46756](../sources/prs/vllm/PR-46756.md) | Add MiniMax-M3 modelopt nvfp4 support | 2026-06-25 | kernel-fusion | fp4, kernel-fusion, moe |
| [#46757](../sources/prs/vllm/PR-46757.md) | Fix Quark mxfp4 quantized model loading issue under mtp | 2026-06-25 |  | fp4, quantization |
| [#46758](../sources/prs/vllm/PR-46758.md) | [ROCm][CI TG] refactor and fix deepep_moe test group | 2026-06-25 |  | moe |
| [#46761](../sources/prs/vllm/PR-46761.md) | [DFlash] Fuse precompute kv per-layer rmsnorms | 2026-06-25 |  | gemm |
| [#46770](../sources/prs/vllm/PR-46770.md) | [Model Runner V2][DFlash] Enable dflash attention backend selection | 2026-06-25 |  | attention, decode |
| [#46548](../sources/prs/vllm/PR-46548.md) | [ROCm] Fix OOB During Model Warmup With `ROCM_ATTN` and MRV2 | 2026-06-24 |  | attention, quantization |
| [#46549](../sources/prs/vllm/PR-46549.md) | [MoE] Free unused MXFP4 scales in OAI Triton Backend | 2026-06-24 | kernel-fusion | fp4, kernel-fusion, moe |
| [#46560](../sources/prs/vllm/PR-46560.md) | [Bugfix][Model Runner V2][Spec Decode] Fix int32 offset overflow in sampler kernels | 2026-06-24 |  | decode |
| [#46570](../sources/prs/vllm/PR-46570.md) | [Core] Add MRV2 virtual-batch PCP for MLA | 2026-06-24 | kernel-fusion | attention, fp4, kernel-fusion |
| [#46621](../sources/prs/vllm/PR-46621.md) | [Feat] Improve Triton JIT diagnostics | 2026-06-24 |  | gemm |
| [#46629](../sources/prs/vllm/PR-46629.md) | [OCP MX ] Add back emulation to available OCP MX backends list | 2026-06-24 | kernel-fusion | fp4, kernel-fusion, moe |
| [#46634](../sources/prs/vllm/PR-46634.md) | [Perf][1/N] Expand Triton kernel warmup coverage, DSv4 | 2026-06-24 |  | mla |
| [#46635](../sources/prs/vllm/PR-46635.md) | [GLM5.2 Perf] Replace MOE all-reduce with reduce-scatter | 2026-06-24 |  | moe |
| [#46642](../sources/prs/vllm/PR-46642.md) | [Kernel][MoE] Tune block-FP8 fused MoE for low-batch decode | 2026-06-24 | kernel-fusion | decode, fp8, kernel-fusion |
| [#46643](../sources/prs/vllm/PR-46643.md) | [Kernel] Vectorized fp32 `moe_sum` reduction and support any topk | 2026-06-24 |  | moe |
| [#46644](../sources/prs/vllm/PR-46644.md) | [Build] Update vllm to point to vllm-project/flash-attention commit that builds FA3 with torch stable API.  | 2026-06-24 |  | attention, flash-attention |
| [#46656](../sources/prs/vllm/PR-46656.md) | New stable abi cleanup | 2026-06-24 | epilogue-fusion, kernel-fusion | epilogue-fusion, fp4, fp8 |
| [#46659](../sources/prs/vllm/PR-46659.md) | Fix FA4 dynamic_causal for full attention layers | 2026-06-24 |  | attention |
| [#46661](../sources/prs/vllm/PR-46661.md) | Allow FlashInfer A2A backends for TRTLLM FP8 MoE Modular | 2026-06-24 | kernel-fusion | fp8, kernel-fusion, moe |
| [#46428](../sources/prs/vllm/PR-46428.md) | [Optimization] Skip DP padding tokens in MoE | 2026-06-23 | kernel-fusion | kernel-fusion, moe |
| [#46432](../sources/prs/vllm/PR-46432.md) | [DeepEP V2] Fill invalid recv_topk_idx with -1 | 2026-06-23 | kernel-fusion | kernel-fusion, moe |
| [#46492](../sources/prs/vllm/PR-46492.md) | [Bugfix] Allow flashinfer_cutlass as a clamped NVFP4 MoE backend | 2026-06-23 | kernel-fusion | fp4, kernel-fusion, moe |
| [#46508](../sources/prs/vllm/PR-46508.md) | [Kernel] Enable PDL for per_token_group_quant_8bit_kernel | 2026-06-23 |  | fp8, quantization |
| [#46518](../sources/prs/vllm/PR-46518.md) | [Kernel][MoE] Allow FlashInfer MXINT4 MoE for gated SiLU | 2026-06-23 | kernel-fusion | kernel-fusion, moe |
| [#46545](../sources/prs/vllm/PR-46545.md) | [ROCm] [MoE] [Perf] Shared-expert fusion for bias-routed MoE; enable on MiniMax-M3 mxfp8 model | 2026-06-23 | kernel-fusion | fp8, kernel-fusion, moe |
| [#46546](../sources/prs/vllm/PR-46546.md) | [ROCm][ [Perf] sparse attention optimization on minimax-m3  | 2026-06-23 |  | attention, sparse-attention |
| [#46339](../sources/prs/vllm/PR-46339.md) | [Bugfix] Re-enable FP8 MoE on NVIDIA Thor | 2026-06-22 |  | fp8, moe, quantization |
| [#46346](../sources/prs/vllm/PR-46346.md) | [GDN] Improve kkt kernel of CuteDSL prefill backend | 2026-06-22 |  | prefill, tcgen05 |
| [#46353](../sources/prs/vllm/PR-46353.md) | [CPU][Perf] Accelerate unquantized MoE for AArch64 | 2026-06-22 | kernel-fusion | gemm, kernel-fusion, moe |
| [#46361](../sources/prs/vllm/PR-46361.md) | [INC][ARK] Direct Register Custom Op for ARK | 2026-06-22 |  | quantization |
| [#46379](../sources/prs/vllm/PR-46379.md) | [Bugfix][KV Offload] Fix swap_blocks_batch on the default stream | 2026-06-22 |  | gemm |
| [#46381](../sources/prs/vllm/PR-46381.md) | [Bugfix][ROCm] Preserve MoE weight padding for unquantized Triton path | 2026-06-22 | kernel-fusion | kernel-fusion, moe, quantization |
| [#46385](../sources/prs/vllm/PR-46385.md) | [Kernel] GLM5 Router GEMM | 2026-06-22 | kernel-fusion | gemm, kernel-fusion, moe |
| [#46389](../sources/prs/vllm/PR-46389.md) | Humming support for 2/3/5/6/7-bit pack-quantized weight-only inference | 2026-06-22 |  | quantization |
| [#46390](../sources/prs/vllm/PR-46390.md) | [Quant] Enable humming w[2-7]a[4,8] inference with compressed-tensors | 2026-06-22 |  | quantization |
| [#46393](../sources/prs/vllm/PR-46393.md) | [Kernel] Add FlashInferCutedslMxfp8LinearKernel (cute-dsl mm_mxfp8) | 2026-06-22 |  | fp8 |
| [#46405](../sources/prs/vllm/PR-46405.md) | [Refactor] Remove dead kernel code | 2026-06-22 |  | gemm |
| [#46406](../sources/prs/vllm/PR-46406.md) | [Bugfix] Support non-power-of-2 top_k in legacy triton_kernels routing | 2026-06-22 | kernel-fusion | kernel-fusion, moe |
| [#46408](../sources/prs/vllm/PR-46408.md) | [Bugfix] Support -1 (invalid/non-local) slots in topk_ids for Triton MoE | 2026-06-22 | kernel-fusion | kernel-fusion, moe |
| [#46414](../sources/prs/vllm/PR-46414.md) | [ROCm] Fix AITER FP8 quantization schema tests | 2026-06-22 |  | fp8, quantization |
| [#46419](../sources/prs/vllm/PR-46419.md) | [ROCm]Enable AITER MoE backend for MiniMax-M3-MXFP4 | 2026-06-22 | kernel-fusion | fp4, kernel-fusion, moe |
| [#46260](../sources/prs/vllm/PR-46260.md) | [ROCm][Test] Fix stale test_gfx950_moe MXFP4 oracle tests | 2026-06-21 |  | fp4, moe, quantization |
| [#46275](../sources/prs/vllm/PR-46275.md) | [ROCm][Perf][DSV4] Enable split sparse decode on gfx942 | 2026-06-21 |  | attention, decode, mla |
| [#46276](../sources/prs/vllm/PR-46276.md) | [BugFix] weights processing peak memory reduction for nvfp4 MoE layers | 2026-06-21 |  | fp4, moe, nvfp4 |
| [#46202](../sources/prs/vllm/PR-46202.md) | [CPU] Enable chunked prefill and prefix caching for qwen3.5 | 2026-06-20 |  | prefill |
| [#46222](../sources/prs/vllm/PR-46222.md) | [ROCm] [Bugfix] Bugfix ROCm Sparse Indexer | 2026-06-20 |  | attention, mla |
| [#46236](../sources/prs/vllm/PR-46236.md) | [Bugfix][Quant] Raise actionable error instead of bare assert for group-size/TP mismatch (#46230) | 2026-06-20 |  | fp8, quantization |
| [#46245](../sources/prs/vllm/PR-46245.md) | [Bugfix][Model Runner V2] Preserve all allowed_token_ids in the logit bias kernel | 2026-06-20 |  | gemm |
| [#46254](../sources/prs/vllm/PR-46254.md) | [Bugfix] Fix NVFP4/OCP MX MoE emulation | 2026-06-20 | kernel-fusion | fp4, kernel-fusion, moe |
| [#46117](../sources/prs/vllm/PR-46117.md) | [ROCm][Perf] MXFP8 dense-linear + grouped-MoE GEMM optimizations for MiniMax-M3 | 2026-06-19 | kernel-fusion | fp8, gemm, kernel-fusion |
| [#46122](../sources/prs/vllm/PR-46122.md) | [ROCm] [Performance] Optimize aiter moe for DeepSeekV4 | 2026-06-19 | kernel-fusion | fp4, kernel-fusion, moe |
| [#46168](../sources/prs/vllm/PR-46168.md) | [Bugfix] Preserve FP8 indexer WK pairs across incremental load_weights | 2026-06-19 |  | fp8 |
| [#46176](../sources/prs/vllm/PR-46176.md) | [ROCm] Use vLLM's fp8 quant max in AITER hipBLASLt accuracy test | 2026-06-19 |  | fp8 |
| [#46177](../sources/prs/vllm/PR-46177.md) | [Bugfix][Model] Support tensor parallelism for DiffusionGemma (#45719) | 2026-06-19 | kernel-fusion | fp8, gemm, kernel-fusion |
| [#46182](../sources/prs/vllm/PR-46182.md) | [Feat][1/N] CuTeDSL warmup infrastructure, FA4 MLA | 2026-06-19 |  | attention, mla, prefill |
| [#46184](../sources/prs/vllm/PR-46184.md) | [ROCm][Perf] Use flydsl moe with Minimax-M3 mxfp8 weights on gfx950 and implemented moe-backend selection | 2026-06-19 | kernel-fusion | fp8, kernel-fusion, moe |
| [#46189](../sources/prs/vllm/PR-46189.md) | [Attention] Add FLASH_ATTN_MLA_SPARSE backend for Hopper sparse MLA | 2026-06-19 |  | attention, mla |
| [#45991](../sources/prs/vllm/PR-45991.md) | [XPU][DeepSeekV4]Add DeepSeek-V4 fuse_index_q SYCL kernel path | 2026-06-18 | kernel-fusion | kernel-fusion |
| [#46001](../sources/prs/vllm/PR-46001.md) | [DeepSeek-V4] Support TEP=16 for the block-FP8 shared expert | 2026-06-18 |  | fp8 |
| [#46006](../sources/prs/vllm/PR-46006.md) | [Kernel] Add PDL support for DeepGEMM kernel | 2026-06-18 | kernel-fusion | fp8, gemm, kernel-fusion |
| [#46020](../sources/prs/vllm/PR-46020.md) | [Attention Backend] add HPC-Ops Attention backend | 2026-06-18 |  | attention |
| [#46076](../sources/prs/vllm/PR-46076.md) | [Attention][DSA] support dcp for FLASHINFER_MLA_SPARSE | 2026-06-18 |  | attention, mla |
| [#46101](../sources/prs/vllm/PR-46101.md) | [Bugfix] Normalize slashes in Helion GPU names | 2026-06-18 |  | gemm |
| [#46108](../sources/prs/vllm/PR-46108.md) | [Model] ColQwen3.5: fix retrieval correctness (bias + bidirectional) | 2026-06-18 |  | attention |
| [#46114](../sources/prs/vllm/PR-46114.md) | [ROCm][Bugfix] Fix chunk alignment when using context parallelism with TRITON_MLA | 2026-06-18 |  | attention, mla |
| [#45892](../sources/prs/vllm/PR-45892.md) | [Minimax-M3] BF16/FP8 Indexer using MSA | 2026-06-17 | kernel-fusion | attention, flash-attention, fp8 |
| [#45895](../sources/prs/vllm/PR-45895.md) | [bugfix]Indexer init skip and MTP TopK share for iteration | 2026-06-17 |  | attention, decode, mla |
| [#45896](../sources/prs/vllm/PR-45896.md) | [feature] MiniMax-M3-MXFP4 support added | 2026-06-17 | kernel-fusion | fp4, kernel-fusion, moe |
| [#45924](../sources/prs/vllm/PR-45924.md) | [MoE Backend] add HPC-Ops MoE backend | 2026-06-17 | kernel-fusion | fp8, kernel-fusion, moe |
| [#45935](../sources/prs/vllm/PR-45935.md) | [Model]Fix MiniMaxM2ForCausalLM perf  regression | 2026-06-17 |  | gemm |
| [#45961](../sources/prs/vllm/PR-45961.md) | [Bugfix] Use native SiLU activation in CPU fused MoE | 2026-06-17 | kernel-fusion | kernel-fusion, moe |
| [#45964](../sources/prs/vllm/PR-45964.md) | [Attention][MLA][DCP] Query replication for MLA decode (DeepSeek-V2/R1 + Kimi-K2.5) | 2026-06-17 |  | attention, decode, mla |
| [#45758](../sources/prs/vllm/PR-45758.md) | [XPU] Fix Triton attn fp8/bf16 check failing | 2026-06-16 |  | attention, fp8 |
| [#45781](../sources/prs/vllm/PR-45781.md) | [Misc] Rename VLLM_TRITON_ATTN_USE_TD to VLLM_TRITON_USE_TD | 2026-06-16 |  | attention |
| [#45782](../sources/prs/vllm/PR-45782.md) | [ROCm][Bugfix]: Fallback GFX942 sparse MLA ops to Triton | 2026-06-16 |  | attention, mla |
| [#45836](../sources/prs/vllm/PR-45836.md) | [NVFP4 MoE/Deepseek V4] Marlin: wire SwiGLU clamp + allow it for clamped models on non-Blackwell | 2026-06-16 | kernel-fusion | fp4, kernel-fusion, moe |
| [#45844](../sources/prs/vllm/PR-45844.md) | [Bugfix] Fix CPU split-KV scratchpad sizing | 2026-06-16 |  | attention |
| [#45845](../sources/prs/vllm/PR-45845.md) | [v1][kvcache] Honor prefix-cache retention interval for Mamba/linear attention | 2026-06-16 |  | attention |
| [#45849](../sources/prs/vllm/PR-45849.md) | [BUG] fix hidden states nan for hybrid attention models | 2026-06-16 |  | attention |
| [#45854](../sources/prs/vllm/PR-45854.md) | [ROCm][Quant][Perf] Minimax-M3:  Enable fp8_per_channel for bf16 weights on mi300x | 2026-06-16 | kernel-fusion | fp8, kernel-fusion, moe |
| [#45631](../sources/prs/vllm/PR-45631.md) | Add Triton recompile detection | 2026-06-15 |  | gemm |
| [#45656](../sources/prs/vllm/PR-45656.md) | [Bugfix] Restore is_sym guard for zp in GPTQ/CT MoE to fix symmetric quant regression | 2026-06-15 |  | moe, quantization |
| [#45666](../sources/prs/vllm/PR-45666.md) | [ROCM] [Communication] Add INT3 quantization method for quickreduce | 2026-06-15 |  | quantization |
| [#45681](../sources/prs/vllm/PR-45681.md) | [ROCm][DSv4] Functional fixes for DeepSeek V4 on MI300X/MI325X | 2026-06-15 | kernel-fusion | attention, fp8, kernel-fusion |
| [#45690](../sources/prs/vllm/PR-45690.md) | [CPU] Support Gemma Diffusion | 2026-06-15 | kernel-fusion | attention, gemm, kernel-fusion |
| [#45703](../sources/prs/vllm/PR-45703.md) | [Kernel] Extend Marlin thread-tile padding to MoE (WNA16 + FP8/MXFP8) | 2026-06-15 | kernel-fusion | fp8, kernel-fusion, moe |
| [#45706](../sources/prs/vllm/PR-45706.md) | [ROCm][Spec Decode] Fix probabilistic draft probs test attention backend | 2026-06-15 |  | attention, decode |
| [#45707](../sources/prs/vllm/PR-45707.md) | [Bugfix][MoE] Restore routed output unpadding before shared expert add | 2026-06-15 | kernel-fusion | kernel-fusion, moe, quantization |
| [#45720](../sources/prs/vllm/PR-45720.md) | [Bugfix][ROCm] Fix MiniMax-M3 FP8 KV cache dtype | 2026-06-15 |  | attention, fp8, sparse-attention |
| [#45723](../sources/prs/vllm/PR-45723.md) | [MoE] Plumb gemm1_alpha/beta/clamp_limit into TRT-LLM FP8 MoE | 2026-06-15 | kernel-fusion | fp8, gemm, kernel-fusion |
| [#45743](../sources/prs/vllm/PR-45743.md) | [M3] Tune Triton indexer score decode for spec-decode | 2026-06-15 |  | attention, decode |
| [#45744](../sources/prs/vllm/PR-45744.md) | [M3] Enable FP8 sparse GQA | 2026-06-15 | kernel-fusion | attention, flash-attention, fp8 |
| [#45564](../sources/prs/vllm/PR-45564.md) | [Bugfix][V1] Split V2 model-runner attention groups on num_heads_q | 2026-06-14 |  | attention |
| [#45589](../sources/prs/vllm/PR-45589.md) | [Bugfix] Fix MoE model load OOM in FlashInfer_TRTLLM  backend with sleep mode | 2026-06-14 | kernel-fusion | kernel-fusion, moe, quantization |
| [#45606](../sources/prs/vllm/PR-45606.md) | nixl_ep: Skip post-receive quantization for NVFP4 | 2026-06-14 | kernel-fusion | fp4, kernel-fusion, moe |
| [#45487](../sources/prs/vllm/PR-45487.md) | [Bugfix][DCP] Fix illegal memory access in DCP a2a decode under full CUDA graphs | 2026-06-13 |  | attention, decode |
| [#45544](../sources/prs/vllm/PR-45544.md) | [Bugfix] Default tie_weights to sharing the weight (fix tied quantized embeddings, e.g. ModelOpt Gemma4) | 2026-06-13 |  | gemm, quantization |
| [#45361](../sources/prs/vllm/PR-45361.md) | [Kernel][Bugfix] Fix INT8 per-token-head KV cache rounding in Triton reshape-and-cache | 2026-06-12 |  | attention, quantization |
| [#45368](../sources/prs/vllm/PR-45368.md) | [xpu][lora]: Align LoRA implementation with Punica GPU: fix _apply_expand rank mismatch, add_inputs hardcode, and MoE EP | 2026-06-12 |  | moe |
| [#45375](../sources/prs/vllm/PR-45375.md) | [Quant] Enable modelopt_mixed on Turing (SM75) | 2026-06-12 |  | attention, quantization |
| [#45381](../sources/prs/vllm/PR-45381.md) | [Model] Add MiniMax M3 support | 2026-06-12 | kernel-fusion | attention, flash-attention, fp8 |
| [#45391](../sources/prs/vllm/PR-45391.md) | [CPU] Refine CPU attention frontend | 2026-06-12 |  | attention |
| [#45401](../sources/prs/vllm/PR-45401.md) | [Bugfix][CPU] Don't build triton-cpu on arm64 release image | 2026-06-12 |  | gemm |
| [#45404](../sources/prs/vllm/PR-45404.md) | fix(moe_wna16): access tp_size via moe_config for RoutedExperts compatibility | 2026-06-12 |  | moe, quantization |
| [#45415](../sources/prs/vllm/PR-45415.md) | [12/n]  final _C library kernel migration | 2026-06-12 |  | fp4, gemm, moe |
| [#45454](../sources/prs/vllm/PR-45454.md) | [Refactor] Remove dead quantization code and tests | 2026-06-12 |  | fp8, quantization |
| [#45466](../sources/prs/vllm/PR-45466.md) | [Bugfix][Kernel] Check output alignment in vectorize_with_alignment (fixes misaligned-address crash for non-multiple-of-8 head sizes) | 2026-06-12 |  | attention, quantization |
| [#45232](../sources/prs/vllm/PR-45232.md) | [FlexAttention] make custom mask mods fully cudagraphable | 2026-06-11 |  | attention |
| [#45251](../sources/prs/vllm/PR-45251.md) | [Bugfix] Restrict FlashInfer cuDNN FP8 ViT attention gate to Blackwell (SM 100) | 2026-06-11 |  | attention, fp8 |
| [#45255](../sources/prs/vllm/PR-45255.md) | [Bugfix] Fix gridDim.y overflow for large row counts | 2026-06-11 |  | fp8, quantization |
| [#45277](../sources/prs/vllm/PR-45277.md) | [Build] Fix CUDA arch build coverage gaps | 2026-06-11 | kernel-fusion, pipeline-stages | fp4, gemm, kernel-fusion |
| [#45295](../sources/prs/vllm/PR-45295.md) | [Kernel] Consolidate Marlin thread-tile padding across all dense Marlin paths | 2026-06-11 |  | fp4, fp8, quantization |
| [#45304](../sources/prs/vllm/PR-45304.md) | [11b/n] Migrate Machete kernels to torch stable ABI | 2026-06-11 |  | quantization |
| [#45306](../sources/prs/vllm/PR-45306.md) | [Quant] Support modelopt_mixed on Ampere (SM80/SM86) | 2026-06-11 |  | quantization |
| [#45312](../sources/prs/vllm/PR-45312.md) | [Bugfix][Quantization] Reject unsupported compressed tensors KV cache schemes | 2026-06-11 |  | quantization |
| [#45111](../sources/prs/vllm/PR-45111.md) | [Attention] Re-enable cross-layer KV cache layout for MLA via stride-aware kernels | 2026-06-10 |  | attention, decode, mla |
| [#45136](../sources/prs/vllm/PR-45136.md) | [XPU] Support int4 group_size=32 W4A16 MoE | 2026-06-10 | kernel-fusion | kernel-fusion, moe |
| [#45140](../sources/prs/vllm/PR-45140.md) | [Kernel][XPU] Adjust kernel unit tests for XPU | 2026-06-10 |  | gemm |
| [#45176](../sources/prs/vllm/PR-45176.md) | [11a/n]  Migrate Marlin kernels to torch stable ABI | 2026-06-10 |  | fp8, moe, quantization |
| [#45181](../sources/prs/vllm/PR-45181.md) | [Spec Decode] Support mixed KV page sizes for DFlash | 2026-06-10 |  | attention, decode |
| [#45182](../sources/prs/vllm/PR-45182.md) | [Perf] Integrate TRTLLM BF16 MoE Modular Kernel  | 2026-06-10 | kernel-fusion | kernel-fusion, moe, quantization |
| [#45200](../sources/prs/vllm/PR-45200.md) | [Models] Fix MiMo v2.x QKV TP sharding + FP4 support | 2026-06-10 |  | fp4, fp8, quantization |
| [#44945](../sources/prs/vllm/PR-44945.md) | [ROCm][Perf] Use fused softplus-sqrt-topk router under AITER fused-MoE | 2026-06-09 | kernel-fusion | kernel-fusion, moe |
| [#44977](../sources/prs/vllm/PR-44977.md) | [ROCm][MLA] Fuse MLA q/kv RMSNorm + FP8 per-token quant in the FP8 attention path | 2026-06-09 | kernel-fusion | attention, fp8, kernel-fusion |
| [#44991](../sources/prs/vllm/PR-44991.md) | [CPU] Skip Triton kernel monkey-patches when Triton-CPU is available | 2026-06-09 |  | decode |
| [#45011](../sources/prs/vllm/PR-45011.md) | [Refactor] Rename rocm_moe.py to rocm_moe_rdna.py | 2026-06-09 |  | moe, quantization |
| [#45040](../sources/prs/vllm/PR-45040.md) | [Bugfix][Quantization] Don't reject fp8_e5m2 KV cache for non-fp8 quantized checkpoints | 2026-06-09 |  | attention, fp8, quantization |
| [#45047](../sources/prs/vllm/PR-45047.md) | [Bugfix] Fix Llama4 weight loading | 2026-06-09 |  | moe |
| [#45052](../sources/prs/vllm/PR-45052.md) | [Bug] Fix test flashmla for DSv4 | 2026-06-09 |  | attention, mla |
| [#45054](../sources/prs/vllm/PR-45054.md) | [Bugfix] Fix weight loading issues caused by #41184 | 2026-06-09 |  | moe |
| [#45072](../sources/prs/vllm/PR-45072.md) | [bugfix] skip conch kernel for g_idx reordering | 2026-06-09 |  | gemm |
| [#45074](../sources/prs/vllm/PR-45074.md) | [Perf][Attention] Pin MLA chunked-context metadata tensors so H2D copies are truly non-blocking | 2026-06-09 |  | attention, mla |
| [#44830](../sources/prs/vllm/PR-44830.md) | [Kernel][Perf] Tune fused_moe FP8 config for Qwen3-Next-80B tp=4 on H100 (+25% at batch 96-512) | 2026-06-08 | kernel-fusion | fp8, kernel-fusion, moe |
| [#44863](../sources/prs/vllm/PR-44863.md) | [BugFix] Initialize model_config for Qwen3-VL MoE | 2026-06-08 |  | moe |
| [#44880](../sources/prs/vllm/PR-44880.md) | [Feature] Support MTP speculative decoding for Bailing hybrid models | 2026-06-08 |  | attention, moe |
| [#44892](../sources/prs/vllm/PR-44892.md) | [DSV4][Minor] Fix supported KV cache dtypes | 2026-06-08 |  | attention, mla |
| [#44893](../sources/prs/vllm/PR-44893.md) | [ROCm][gpt-oss] Pass GateMode.INTERLEAVE for MXFP4 W4A16 fused MoE | 2026-06-08 | kernel-fusion | fp4, kernel-fusion, moe |
| [#44897](../sources/prs/vllm/PR-44897.md) | [Bugfix][MoE] Fix fused MoE expert mapping helper call sites | 2026-06-08 | kernel-fusion | kernel-fusion, moe |
| [#44899](../sources/prs/vllm/PR-44899.md) | [ROCm][DSv4][Perf] Flash-decode split-K decode attention kernel | 2026-06-08 |  | attention, decode, mla |
| [#44907](../sources/prs/vllm/PR-44907.md) | [Cohere] Cohere2 moe parser fix | 2026-06-08 |  | moe |
| [#44912](../sources/prs/vllm/PR-44912.md) | [Bugfix][ROCm] Fix FP8 per-tensor scale rank mismatch causing Inductor assertion failure | 2026-06-08 |  | fp8 |
| [#44921](../sources/prs/vllm/PR-44921.md) | [Bugfix] Lazily import the humming quantization backend | 2026-06-08 | kernel-fusion | kernel-fusion, moe, quantization |
| [#44763](../sources/prs/vllm/PR-44763.md) | Add weights padding for fp8 per-block online quantization | 2026-06-07 |  | fp8, quantization |
| [#44771](../sources/prs/vllm/PR-44771.md) | [XPU][Minor] format moe kernel name and add in kernel list | 2026-06-07 | kernel-fusion | fp4, fp8, kernel-fusion |
| [#44774](../sources/prs/vllm/PR-44774.md) | [KV Connector] Mooncake store: prefix-cache retention interval for sparse attention | 2026-06-07 |  | attention |
| [#44733](../sources/prs/vllm/PR-44733.md) | [KV offload] Parallel-agnostic fs-tier cache for single full-attention group | 2026-06-06 |  | attention |
| [#44735](../sources/prs/vllm/PR-44735.md) | [Bugfix] Canonicalize FP8 weight layout to (K, N) at the source | 2026-06-06 |  | fp8, quantization |
| [#44613](../sources/prs/vllm/PR-44613.md) | [Bugfix][MoE] Snapshot max_cudagraph_capture_size into FusedMoEConfig | 2026-06-05 | kernel-fusion | fp4, kernel-fusion, moe |
| [#44626](../sources/prs/vllm/PR-44626.md) | [ROCm][AITER][Quark] Tag per-channel FP8 weights as PER_CHANNEL so AITER pre-shuffled GEMM is selected | 2026-06-05 |  | fp8, gemm, quantization |
| [#44639](../sources/prs/vllm/PR-44639.md) | [CPU][Perf]Added tanh AOR for faster gelu activations. | 2026-06-05 |  | gemm |
| [#44667](../sources/prs/vllm/PR-44667.md) | [NVFP4][Emulation] Fuse NVFP4 weight dequantization with compute in triton kernel for w13/w2 MOE MLP linears | 2026-06-05 | kernel-fusion | fp4, kernel-fusion, moe |
| [#44679](../sources/prs/vllm/PR-44679.md) | [ROCm][Bugfix] Make intermediate_pad TP-aware in rocm_aiter_fused_experts | 2026-06-05 | kernel-fusion | kernel-fusion, moe |
| [#44687](../sources/prs/vllm/PR-44687.md) | [CI/Build] Skip test_use_trtllm_attention on non-CUDA platforms | 2026-06-05 |  | attention |
| [#44694](../sources/prs/vllm/PR-44694.md) | [Bugfix] Fix Qwen3.5-FP8 nightly fail. Guard fused_add_rms_norm input/weight dtype mismatch in RMSNorm + quant fusion | 2026-06-05 | kernel-fusion | fp8, kernel-fusion |
| [#44470](../sources/prs/vllm/PR-44470.md) | [XPU] Cap topk/topp Triton BLOCK_SIZE to 4096 to fix Top-p mask difference failures | 2026-06-04 |  | gemm |
| [#44492](../sources/prs/vllm/PR-44492.md) | [Bugfix][Spec-Decode] Populate draft seq_lens_cpu_upper_bound for spec-decode attention metadata | 2026-06-04 |  | attention, decode |
| [#44565](../sources/prs/vllm/PR-44565.md) | [10c/n] Migrate MoE kernels to torch stable ABI  | 2026-06-04 |  | gemm, moe, tma |
| [#44571](../sources/prs/vllm/PR-44571.md) | [Bugfix] Exclude vision embedder from quantization in Gemma4 Unified | 2026-06-04 |  | gemm, quantization |
| [#44572](../sources/prs/vllm/PR-44572.md) | [Perf] SM90 cutlass fp8 mm supports odd M by swap_ab, 180~290% kernel performance improvement | 2026-06-04 |  | fp8, quantization |
| [#44380](../sources/prs/vllm/PR-44380.md) | [Bugfix] Fix test_cutlass_moe.py | 2026-06-03 | kernel-fusion | kernel-fusion, moe |
| [#44393](../sources/prs/vllm/PR-44393.md) | [Attention][CPU] Standardize kv layout to blocks first | 2026-06-03 |  | attention |
| [#44400](../sources/prs/vllm/PR-44400.md) | [ROCm][Perf] Enable W4A16 FlyDSL MoE | 2026-06-03 | kernel-fusion | kernel-fusion, moe, quantization |
| [#44408](../sources/prs/vllm/PR-44408.md) | Fix MiDashengLM TP>1 crash in audio encoder attention | 2026-06-03 |  | attention |
| [#44446](../sources/prs/vllm/PR-44446.md) | [Model Runner V2] Migration to support quantized model by default [5/N] | 2026-06-03 |  | quantization |
| [#44455](../sources/prs/vllm/PR-44455.md) | [2/N][KV-Cache Layout Refactor] Pack K/V into the content dim across attention backends | 2026-06-03 | kernel-fusion | attention, kernel-fusion, sparse-attention |
| [#44456](../sources/prs/vllm/PR-44456.md) | [3/N][KV-Cache Layout Refactor] Standardize Mamba cache; drop `get_transfer_cache_regions` | 2026-06-03 |  | attention |
| [#44462](../sources/prs/vllm/PR-44462.md) | up FI fp8 moe topk to 32 | 2026-06-03 | kernel-fusion | fp8, kernel-fusion, moe |
| [#44264](../sources/prs/vllm/PR-44264.md) | [Bugfix][Model] Qwen3-Omni: move cu_seqlens to GPU before VIT attention | 2026-06-02 |  | attention, moe |
| [#44313](../sources/prs/vllm/PR-44313.md) | [ROCm][Perf] Add Fused Shared Expert (FSE) support for GLM-4.5/6/7 | 2026-06-02 | kernel-fusion | kernel-fusion, moe |
| [#44347](../sources/prs/vllm/PR-44347.md) | [Bugfix] Update TrtLLM MoE routing methods | 2026-06-02 | kernel-fusion | fp4, fp8, kernel-fusion |
| [#44161](../sources/prs/vllm/PR-44161.md) | [Kernel][DSv4] Optimize sparse FP8 compressor kernels | 2026-06-01 |  | fp8 |
| [#44178](../sources/prs/vllm/PR-44178.md) | [ROCm][Cleanup] Remove stale AITER FA hybrid KV-cache TODO | 2026-06-01 |  | attention |
| [#44214](../sources/prs/vllm/PR-44214.md) | [RL Infra][FlashInfer] Enable router replay output from FlashInfer monolithic MoE kernel | 2026-06-01 | kernel-fusion | fp4, fp8, kernel-fusion |
| [#44217](../sources/prs/vllm/PR-44217.md) | [Perf] Fix dsv3_router_gemm heuristic | 2026-06-01 | kernel-fusion | gemm, kernel-fusion, moe |
| [#44220](../sources/prs/vllm/PR-44220.md) | [Perf] use triton moe backend on hopper by default | 2026-06-01 | kernel-fusion | kernel-fusion, moe, quantization |
| [#44230](../sources/prs/vllm/PR-44230.md) | optimize the compressor 128 split cutedsl kernel  | 2026-06-01 |  | gemm |
| [#44260](../sources/prs/vllm/PR-44260.md) | Add the QuantizedActivation linear-kernel contract | 2026-06-01 | kernel-fusion | fp4, fp8, kernel-fusion |
| [#44109](../sources/prs/vllm/PR-44109.md) | [Kernel] Add weightless RMSNorm CUDA kernels for has_weight=False (#41430) | 2026-05-31 |  | gemm |
| [#44118](../sources/prs/vllm/PR-44118.md) | docs: fix MLA attention docstring examples | 2026-05-31 |  | attention, mla |
| [#44120](../sources/prs/vllm/PR-44120.md) | [MoE Refactor] Migrate MoeWNA16Method quantization method over to using the new MK oracle scheme. | 2026-05-31 | kernel-fusion | kernel-fusion, moe, quantization |
| [#44132](../sources/prs/vllm/PR-44132.md) | [Quantization] add online fp8 ptpc | 2026-05-31 |  | fp8, quantization |
| [#44041](../sources/prs/vllm/PR-44041.md) | [Bugfix] Fix benchmark_moe.py after inplace mechanism removal | 2026-05-30 |  | moe |
| [#44044](../sources/prs/vllm/PR-44044.md) | [Feature] Support DCP with FP8 KV cache in MLA decode path | 2026-05-30 |  | attention, decode, fp8 |
| [#44053](../sources/prs/vllm/PR-44053.md) | [Bugfix][V1][TurboQuant] Reserve workspace before CUDA graph capture | 2026-05-30 |  | attention, quantization |
| [#44075](../sources/prs/vllm/PR-44075.md) | [ROCm][Perf] Fused MoE W4A16 HIP kernel for AMD RDNA3 (gfx1100) | 2026-05-30 | kernel-fusion | gemm, kernel-fusion, moe |
| [#43947](../sources/prs/vllm/PR-43947.md) | [XPU] fix xpu install document triton-xpu version | 2026-05-29 |  | gemm |
| [#43958](../sources/prs/vllm/PR-43958.md) | [XPU] Fix FP8 block-scaled scheme selection on non-CUDA platforms | 2026-05-29 |  | block-scale, fp8, quantization |
| [#43961](../sources/prs/vllm/PR-43961.md) | [Bugfix] Corrupted MLA + linear attention | 2026-05-29 |  | attention, mla |
| [#43979](../sources/prs/vllm/PR-43979.md) | [ROCm][Bugfix] Fix GPT-OSS Quark MXFP4 MoE loading - emulation buffer not block-aligned | 2026-05-29 | kernel-fusion | fp4, kernel-fusion, moe |
| [#43981](../sources/prs/vllm/PR-43981.md) | [AMD][Bugfix][Quantization] Honor fused-name match in is_layer_skipped | 2026-05-29 | kernel-fusion | kernel-fusion, quantization |
| [#43994](../sources/prs/vllm/PR-43994.md) | [Kernel][Helion][1/N] Add Helion kernel for silu_and_mul_per_block_quant | 2026-05-29 |  | gemm |
| [#44005](../sources/prs/vllm/PR-44005.md) | [Bug] Fix torch device issue for MOE permute | 2026-05-29 | kernel-fusion | kernel-fusion, moe |
| [#44010](../sources/prs/vllm/PR-44010.md) | [Kernel][Helion][1/N] Add Helion kernel for fused_qk_norm_rope | 2026-05-29 | kernel-fusion | kernel-fusion |
| [#44013](../sources/prs/vllm/PR-44013.md) | Migrate header files to torch stable abi | 2026-05-29 | epilogue-fusion, kernel-fusion, persistent-kernel | epilogue-fusion, fp4, kernel-fusion |
| [#43827](../sources/prs/vllm/PR-43827.md) | [DSv4] Adding TRTLLM gen attention kernel | 2026-05-28 | kernel-fusion | attention, kernel-fusion, mla |
| [#43853](../sources/prs/vllm/PR-43853.md) | Feature: Enable Flashinfer non-gated MoE bf16 | 2026-05-28 | kernel-fusion | kernel-fusion, moe, quantization |
| [#43896](../sources/prs/vllm/PR-43896.md) | Correct model layer aliasing for Bert style models | 2026-05-28 |  | gemm |
| [#43898](../sources/prs/vllm/PR-43898.md) | [ROCm][DSv4] Remove device pipeline stall in sparse attention | 2026-05-28 | pipeline-stages | attention, mla, pipeline-stages |
| [#43914](../sources/prs/vllm/PR-43914.md) | Remove redundant Triton KV cache dtype asserts and enforce architectural support (fp8 >= sm89) | 2026-05-28 |  | attention, fp8 |
| [#43721](../sources/prs/vllm/PR-43721.md) | [ROCm][Quantization][4/N] refactor quark_moe fp8 w/ oracle | 2026-05-27 |  | fp8, moe, quantization |
| [#43727](../sources/prs/vllm/PR-43727.md) | [MoE] Remove inplace fused experts mechanism | 2026-05-27 | kernel-fusion | fp4, fp8, gemm |
| [#43769](../sources/prs/vllm/PR-43769.md) | [Bugfix] Pass `routed_scaling_factor` to FlashInfer TRTLLM BF16 MoE | 2026-05-27 | kernel-fusion | kernel-fusion, moe |
| [#43803](../sources/prs/vllm/PR-43803.md) | [Perf] remove seqlen from Mamba SSD chunk kernels | 2026-05-27 |  | gemm |
| [#43817](../sources/prs/vllm/PR-43817.md) | [ROCm] Add attention sink support to AITer flash attention backend | 2026-05-27 |  | attention |
| [#43629](../sources/prs/vllm/PR-43629.md) | [ROCm] Remove MegaMoE integration in deepseek v4 | 2026-05-26 |  | moe |
| [#43645](../sources/prs/vllm/PR-43645.md) | [XPU] Add W8A8 FP8 linear kernel with multi-granularity quant support | 2026-05-26 |  | fp8, quantization |
| [#43646](../sources/prs/vllm/PR-43646.md) | [XPU] Fix fused MoE LoRA kernel crash on XPU by using platform-agnos num_compute_units | 2026-05-26 | kernel-fusion | kernel-fusion, moe |
| [#43660](../sources/prs/vllm/PR-43660.md) | [Attention][AMD] Standardize kv layout to blocks first for AMD | 2026-05-26 |  | attention |
| [#43673](../sources/prs/vllm/PR-43673.md) | [ROCm][Perf] DSv3.2: fuse MLA Q concat+fp8-quant in forward_mqa | 2026-05-26 |  | attention, fp8, mla |
| [#43679](../sources/prs/vllm/PR-43679.md) | [ROCm][DSV4] Enable Tilelang MHC replacing torch/triton mhc | 2026-05-26 |  | gemm |
| [#43706](../sources/prs/vllm/PR-43706.md) | [Perf] Optimize cutlass fp8 scaled mm bypassing padding, 20% kernel performance improvement | 2026-05-26 |  | fp8 |
| [#43554](../sources/prs/vllm/PR-43554.md) | [Kernel] Remove NormGateLinear | 2026-05-25 | kernel-fusion | gemm, kernel-fusion, moe |
| [#43556](../sources/prs/vllm/PR-43556.md) | [Attention] Mamba attention module refactor - LINEAR | 2026-05-25 |  | attention, moe |
| [#43557](../sources/prs/vllm/PR-43557.md) | Fix the E8M0 scale computation in the MXFP4 (W4A4) MOE CUTLASS kernel | 2026-05-25 |  | fp4, moe, nvfp4 |
| [#43565](../sources/prs/vllm/PR-43565.md) | [XPU] support MTP of gdn attention | 2026-05-25 |  | attention |
| [#43591](../sources/prs/vllm/PR-43591.md) | [MM][CG] Gemma3 Encoder CUDA Graph | 2026-05-25 |  | gemm |
| [#43597](../sources/prs/vllm/PR-43597.md) | attention: pass None for unused args in unified attention TD path | 2026-05-25 |  | attention |
| [#43599](../sources/prs/vllm/PR-43599.md) | [Bugfix][Kernel] TRTLLM NVFP4 MoE chunking | 2026-05-25 | kernel-fusion | fp4, fp8, kernel-fusion |
| [#43540](../sources/prs/vllm/PR-43540.md) | [Quantization] Fix Humming RoutedExperts import | 2026-05-24 |  | quantization |
| [#43461](../sources/prs/vllm/PR-43461.md) | [MoE] [MoE Refactor] Add moe kernel oracle abc 37753 | 2026-05-23 | kernel-fusion | kernel-fusion, moe, quantization |
| [#43373](../sources/prs/vllm/PR-43373.md) | [MoE Refactor] Standardize Humming MoE experts + utilities | 2026-05-22 | kernel-fusion | fp4, fp8, kernel-fusion |
| [#43404](../sources/prs/vllm/PR-43404.md) | [XPU] add awq format for INCXPULinear | 2026-05-22 |  | quantization |
| [#43409](../sources/prs/vllm/PR-43409.md) | [CPU] Support CPU W4A16 INT4 MoE | 2026-05-22 | kernel-fusion | kernel-fusion, moe, quantization |
| [#43421](../sources/prs/vllm/PR-43421.md) | [XPU][Mamba] Triton-based selective scan forward op for XPU | 2026-05-22 |  | gemm |
| [#43307](../sources/prs/vllm/PR-43307.md) | [Kernel][Test] Extend lightning_attn and awq_triton kernel tests to XPU | 2026-05-21 |  | attention, quantization |
| [#43325](../sources/prs/vllm/PR-43325.md) | [MLA][Attention] Add OOT MLA prefill backend registration mechanism | 2026-05-21 |  | attention, mla, prefill |
| [#43328](../sources/prs/vllm/PR-43328.md) | Enable B12x backend for non-gated MoEs (like Nemotron)  | 2026-05-21 | kernel-fusion | kernel-fusion, moe |
| [#43330](../sources/prs/vllm/PR-43330.md) | Allow native KV cache dtype in Triton cache update | 2026-05-21 |  | attention |
| [#43362](../sources/prs/vllm/PR-43362.md) | [Bugfix] FusedMoE: coerce shape-(1,) per-tensor scales to 0-D scalar … | 2026-05-21 | kernel-fusion | kernel-fusion, moe |
| [#43167](../sources/prs/vllm/PR-43167.md) | Remove KV cache scale boilerplate from model weight loading methods | 2026-05-20 |  | fp8, gemm, moe |
| [#43223](../sources/prs/vllm/PR-43223.md) | Fix FlashInfer TRTLLM NvFP4 monolithic MoE routing | 2026-05-20 | kernel-fusion | fp4, kernel-fusion, moe |
| [#43225](../sources/prs/vllm/PR-43225.md) | [CPU] Experimentally enable Triton and MRV2 | 2026-05-20 |  | gemm |
| [#43232](../sources/prs/vllm/PR-43232.md) | Xqa decode kernels | 2026-05-20 |  | attention, decode |
| [#43234](../sources/prs/vllm/PR-43234.md) | [Refactor] Remove dead code | 2026-05-20 | kernel-fusion | kernel-fusion, moe, quantization |
| [#43050](../sources/prs/vllm/PR-43050.md) | feat: MLA prefill enable FA4 fp8 output | 2026-05-19 |  | attention, fp8, mla |
| [#43083](../sources/prs/vllm/PR-43083.md) | Tuning script and configs for Triton Mamba SSU kernel | 2026-05-19 |  | gemm |
| [#43100](../sources/prs/vllm/PR-43100.md) | [BugFix] Fix Humming MoE deploy error | 2026-05-19 |  | moe, quantization |
| [#43135](../sources/prs/vllm/PR-43135.md) | [Perf][gpt-oss] Downgrade triton_kernels to v3.5.1 | 2026-05-19 | kernel-fusion | fp4, kernel-fusion, moe |
| [#42915](../sources/prs/vllm/PR-42915.md) | [XPU] reudce host overhead of XPU MOE | 2026-05-18 | kernel-fusion | kernel-fusion, moe |
| [#42920](../sources/prs/vllm/PR-42920.md) | [CPU] Support cpu compressed-tensor w8a8 int8 moe | 2026-05-18 | kernel-fusion | kernel-fusion, moe, quantization |
| [#42951](../sources/prs/vllm/PR-42951.md) | [XPU]feat: add XPU fallback for MoE topk routing and MXFP4 backend | 2026-05-18 | kernel-fusion | fp4, kernel-fusion, moe |
| [#42952](../sources/prs/vllm/PR-42952.md) | [XPU]feat: enable FP8 block-scaled quantization on XPU | 2026-05-18 |  | block-scale, fp8, quantization |
| [#42953](../sources/prs/vllm/PR-42953.md) | feat: add DeepSeek-V4 XPU attention decode path | 2026-05-18 |  | attention, decode, fp8 |
| [#42958](../sources/prs/vllm/PR-42958.md) | Support ModelOpt MXFP8 non-gated MoE | 2026-05-18 | kernel-fusion | fp8, kernel-fusion, moe |
| [#42978](../sources/prs/vllm/PR-42978.md) | [ROCm][MLA][Bugfix] Reserve FP8 prefill workspace before lock for Kimi-K2.5 | 2026-05-18 |  | attention, fp8, mla |
| [#42996](../sources/prs/vllm/PR-42996.md) | [Kernel] Add PDL support for DeepGEMM kernel | 2026-05-18 | kernel-fusion | fp8, gemm, kernel-fusion |
| [#43004](../sources/prs/vllm/PR-43004.md) | [Model Refactoring] Migrate DeepSeek V4 to vllm/models/ [1/N]  | 2026-05-18 |  | moe, quantization |
| [#42890](../sources/prs/vllm/PR-42890.md) | Support nvfp4 kv with kv-cache-dtype-skip-layers sliding_window | 2026-05-17 |  | attention, fp4, nvfp4 |
| [#42832](../sources/prs/vllm/PR-42832.md) | [ROCm][GPT-OSS] Fuse RoPE + static Q FP8 quant on fused RoPE+KV path | 2026-05-16 | kernel-fusion | fp8, kernel-fusion |
| [#42692](../sources/prs/vllm/PR-42692.md) | [Bugfix] DFlash FP8 KV-Cache | 2026-05-15 |  | decode, fp8 |
| [#42727](../sources/prs/vllm/PR-42727.md) | fix(quantization): Fix AWQ dequantize on Intel XPU and refactor AutoAWQ config | 2026-05-15 | kernel-fusion | kernel-fusion, moe, quantization |
| [#42736](../sources/prs/vllm/PR-42736.md) | [Kernel][Test] Make kernel tests for mamba dual-HW (CUDA + XPU) | 2026-05-15 |  | gemm |
| [#42767](../sources/prs/vllm/PR-42767.md) | [Refactor] Remove dead cuda kernels | 2026-05-15 |  | attention, moe |
| [#42768](../sources/prs/vllm/PR-42768.md) | [MoE Refactor] Migrate ModelOptMxFp8FusedMoE to oracle | 2026-05-15 | kernel-fusion | fp8, kernel-fusion, moe |
| [#42782](../sources/prs/vllm/PR-42782.md) | [Bugfix] Respect explicit --kv-cache-dtype over checkpoint kv_cache_scheme | 2026-05-15 |  | attention |
| [#42787](../sources/prs/vllm/PR-42787.md) | [MM] Enable FlashInfer metadata support for Qwen2.5-VL vision attention | 2026-05-15 |  | attention |
| [#42789](../sources/prs/vllm/PR-42789.md) | [MoE Refactor] W4a8 int8 oracle | 2026-05-15 | kernel-fusion | kernel-fusion, moe, quantization |
| [#42595](../sources/prs/vllm/PR-42595.md) | [Bugfix] [ROCm] [DSV4] Fix AITER MXFP4 MoE weight loading and shuffle… | 2026-05-14 | kernel-fusion | fp4, kernel-fusion, moe |
| [#42647](../sources/prs/vllm/PR-42647.md) | [MoE Refactor] Migrate MoeWNA16Method quantization to MK oracle | 2026-05-14 | kernel-fusion | kernel-fusion, moe, quantization |
| [#42680](../sources/prs/vllm/PR-42680.md) | [MoE] Migrate W4A8 CT to oracle kernel setup | 2026-05-14 | kernel-fusion | fp8, kernel-fusion, moe |
| [#42685](../sources/prs/vllm/PR-42685.md) | [FlashAttn] Fix supports_kv_cache_dtype() accepting unhandled fp8 kv-cache dtype variants | 2026-05-14 |  | attention, fp8, quantization |
| [#42483](../sources/prs/vllm/PR-42483.md) | Refactor AWQ Marlin MoE onto modular WNA16 oracle | 2026-05-13 | kernel-fusion | kernel-fusion, moe, quantization |
| [#42497](../sources/prs/vllm/PR-42497.md) | [Perf] Wire silu_and_mul_per_block_quant into TritonFP8MoE (MiniMax-M2)  | 2026-05-13 | kernel-fusion | fp8, kernel-fusion, moe |
| [#42527](../sources/prs/vllm/PR-42527.md) | [Kernel] Pack topk id/weights triton kernel | 2026-05-13 | kernel-fusion | fp4, kernel-fusion, moe |
| [#42553](../sources/prs/vllm/PR-42553.md) | [MoE Refactor] WNA16 MoE backend selection into oracle module | 2026-05-13 | kernel-fusion | kernel-fusion, moe, quantization |
| [#42566](../sources/prs/vllm/PR-42566.md) | [Quantization][ModelOpt] W4A16 NVFP4 fused MoE + mixed-precision dispatch | 2026-05-13 | kernel-fusion | fp4, kernel-fusion, moe |
| [#42569](../sources/prs/vllm/PR-42569.md) | [Attention] FlashAttention 4 SM100 FP8 kv cache support | 2026-05-13 |  | attention, fp8, quantization |
| [#42373](../sources/prs/vllm/PR-42373.md) | fix: MoE model using shared routed experts crashes on AMD GPUs | 2026-05-12 | kernel-fusion | kernel-fusion, moe |
| [#42425](../sources/prs/vllm/PR-42425.md) | [Perf] Add VLLM_TRITON_FORCE_FIRST_CONFIG to skip Triton autotuning | 2026-05-12 |  | gemm |
| [#42287](../sources/prs/vllm/PR-42287.md) | [Bugfix] Fix DSV4 swiglu_limit on marlin backend | 2026-05-11 | kernel-fusion | kernel-fusion, moe |
| [#42290](../sources/prs/vllm/PR-42290.md) | [LoRA] Add one shot triton kernel For MoE LoRA | 2026-05-11 | kernel-fusion | kernel-fusion, moe |
| [#42304](../sources/prs/vllm/PR-42304.md) | [Experimental] Breakable CUDA graph | 2026-05-11 |  | attention, mla |
| [#42212](../sources/prs/vllm/PR-42212.md) | [Perf] Triton fast path for small CPU→GPU `swap_blocks_batch` in the offloading connector | 2026-05-10 |  | gemm |
| [#42139](../sources/prs/vllm/PR-42139.md) | [XPU][MoE] support block_fp8_moe on xpu | 2026-05-09 | kernel-fusion | fp8, kernel-fusion, moe |
| [#42175](../sources/prs/vllm/PR-42175.md) | [Core][Model] Gemma4: Unified FA4 for all layers + FlashAttention mm_prefix support | 2026-05-09 |  | attention, decode, gemm |
| [#42181](../sources/prs/vllm/PR-42181.md) | [Bugfix] Accept canonicalized `modelopt_*` quant_method in `_extract_modelopt_quant_algo` | 2026-05-09 |  | quantization |
| [#42027](../sources/prs/vllm/PR-42027.md) | [Kernel][MoE] Add GELU_TANH to CPU, CUTLASS, and WNA16 MoE backends | 2026-05-08 | kernel-fusion | kernel-fusion, moe, quantization |
| [#42061](../sources/prs/vllm/PR-42061.md) | [ROCm][Bugfix] Re-tag AITER MoE weights as preshuffled after replace_parameter | 2026-05-08 | kernel-fusion | fp4, fp8, kernel-fusion |
| [#42080](../sources/prs/vllm/PR-42080.md) | [feat] Add FP8 per-tensor Q scale support to Triton attention backend | 2026-05-08 |  | attention, fp8 |
| [#42098](../sources/prs/vllm/PR-42098.md) | Use hidden_pad and intermediate_pad from vLLM #34301 | 2026-05-08 | kernel-fusion | kernel-fusion, moe |
| [#42120](../sources/prs/vllm/PR-42120.md) | [Bugfix] Fix corrupt outputs in MoE FP8 LoRA responses and MoE base model responses when LoRAs are loaded | 2026-05-08 | kernel-fusion | fp8, kernel-fusion, moe |
| [#42121](../sources/prs/vllm/PR-42121.md) | [Attention][Cleanup] Remove tree attention | 2026-05-08 |  | attention, decode |
| [#42124](../sources/prs/vllm/PR-42124.md) | Add LM head quantization support for ModelOpt | 2026-05-08 |  | quantization |
| [#41892](../sources/prs/vllm/PR-41892.md) | [Bugfix][Quark] Fix W8A8 INT8 garbage outputs on Step-3.5-Flash (and other 3-key fused-MoE Quark exports) | 2026-05-07 | kernel-fusion | kernel-fusion, moe, quantization |
| [#41922](../sources/prs/vllm/PR-41922.md) | [CPU] Add MXFP4 W4A16 MoE support | 2026-05-07 | kernel-fusion | fp4, fp8, gemm |
| [#41979](../sources/prs/vllm/PR-41979.md) | [MoE] Move various experts classes to fused_moe/experts/ | 2026-05-07 | kernel-fusion | fp4, fp8, gemm |
| [#42002](../sources/prs/vllm/PR-42002.md) | [Bugfix] Plumb hidden_dim_unpadded through moe_forward fake to fix gpt-oss MXFP4 + torch.compile | 2026-05-07 | kernel-fusion | fp4, kernel-fusion, moe |
| [#41797](../sources/prs/vllm/PR-41797.md) | [Attention] add triton diff-kv backend for mimo | 2026-05-06 |  | attention |
| [#41713](../sources/prs/vllm/PR-41713.md) | Eliminate redundant MoE buffer copies in AITER fused experts (without dependency on AITER changes) | 2026-05-05 | kernel-fusion | kernel-fusion, moe |
| [#41722](../sources/prs/vllm/PR-41722.md) | [MyPy] Fix mypy for `vllm/lora` | 2026-05-05 | kernel-fusion | kernel-fusion, moe |
| [#41633](../sources/prs/vllm/PR-41633.md) | [EPLB] Nixl communicator optimization. Zero-copy transfers | 2026-05-04 | kernel-fusion | fp4, kernel-fusion, moe |
| [#41646](../sources/prs/vllm/PR-41646.md) | [Bugfix] Restore moe_forward output shape invariant on TRTLLM MXFP4 path | 2026-05-04 | kernel-fusion | fp4, kernel-fusion, moe |
| [#41652](../sources/prs/vllm/PR-41652.md) | [Quantization] add humming moe backend to all dense/moe oracles | 2026-05-04 | kernel-fusion | fp4, fp8, gemm |
| [#41566](../sources/prs/vllm/PR-41566.md) | [Quantization] Rework quantization_config to use QuantKey and allow for activation override | 2026-05-03 | kernel-fusion | fp4, fp8, kernel-fusion |
| [#41314](../sources/prs/vllm/PR-41314.md) | [CPU] Add FP8 W8A16 MoE support | 2026-04-30 | kernel-fusion | fp8, kernel-fusion, moe |
| [#41426](../sources/prs/vllm/PR-41426.md) | [XPU][MoE] Add WNA16 oracle backend for GPTQ sym-int4 (xpu_fused_moe) | 2026-04-30 | kernel-fusion | kernel-fusion, moe |
| [#41184](../sources/prs/vllm/PR-41184.md) | [MoE Refactor] FusedMoE/MoERunner inversion refactor | 2026-04-29 | kernel-fusion | fp4, fp8, gemm |
| [#41083](../sources/prs/vllm/PR-41083.md) | [Quantization] add humming mxfp4 moe backend | 2026-04-28 | kernel-fusion | fp4, kernel-fusion, moe |
| [#41126](../sources/prs/vllm/PR-41126.md) | [Attention] Mamba attention module refactor | 2026-04-28 |  | attention |
| [#41175](../sources/prs/vllm/PR-41175.md) | [ROCm][Bugfix]: W4A4 MOE using emulation instead of AITER on MXFP4-supported hardware | 2026-04-28 | kernel-fusion | fp4, kernel-fusion, moe |
| [#40977](../sources/prs/vllm/PR-40977.md) | [ROCm][Kernel] Add HybridW4A16LinearKernel: Triton prefill + HIP skinny decode | 2026-04-27 |  | decode, gemm, prefill |
| [#40996](../sources/prs/vllm/PR-40996.md) | DCP supports hybrid attention | 2026-04-27 |  | attention |
| [#41026](../sources/prs/vllm/PR-41026.md) | [Model] Add support for openai/privacy-filter | 2026-04-27 | kernel-fusion | attention, kernel-fusion, moe |
| [#41050](../sources/prs/vllm/PR-41050.md) | [Kernel][MoE] Support GELU on TRT-LLM NvFP4 fused MoE for Gemma4 | 2026-04-27 | kernel-fusion | fp4, gemm, kernel-fusion |
| [#41052](../sources/prs/vllm/PR-41052.md) | [Attention] Sync FA with upstream | 2026-04-27 |  | attention |
| [#41055](../sources/prs/vllm/PR-41055.md) | [MoE Refactor]  EPLB refactoring for FusedMoE | 2026-04-27 | kernel-fusion | fp4, fp8, kernel-fusion |
| [#40881](../sources/prs/vllm/PR-40881.md) | elastic_ep: stage/commit MoE quant method on reconfigure | 2026-04-25 | kernel-fusion | kernel-fusion, moe |
| [#40810](../sources/prs/vllm/PR-40810.md) | [EPLB] Fix replica selection bias in fused_moe router | 2026-04-24 | kernel-fusion | kernel-fusion, moe |
| [#40835](../sources/prs/vllm/PR-40835.md) | [Feature] Triton INT4 per-token-head KV cache quantization | 2026-04-24 |  | attention, quantization |
| [#40671](../sources/prs/vllm/PR-40671.md) | [MoE Refactor] Rename FusedMoE.make_expert_params_mapping to fused_moe_make_expert_params_mapping | 2026-04-23 | kernel-fusion | gemm, kernel-fusion, moe |
| [#40735](../sources/prs/vllm/PR-40735.md) | [MoE Refactor] Introduce RoutedExperts alias for FusedMoE and don't store SharedExperts in MK | 2026-04-23 | kernel-fusion | fp4, fp8, kernel-fusion |
| [#40568](../sources/prs/vllm/PR-40568.md) | [MoE] Move xpu moe to fused_moe/experts/ | 2026-04-22 | kernel-fusion | fp4, fp8, kernel-fusion |
| [#40574](../sources/prs/vllm/PR-40574.md) | [MoE] Move cutlass moe to fused_moe/experts/ | 2026-04-22 | kernel-fusion | fp4, fp8, gemm |
| [#40601](../sources/prs/vllm/PR-40601.md) | [quant][autoround]Refactor INC quantization into package with INCScheme orchestrator | 2026-04-22 |  | quantization |
| [#40470](../sources/prs/vllm/PR-40470.md) | [Attention] Extract KV-cache update from CPU attention backend | 2026-04-21 |  | attention |
| [#40410](../sources/prs/vllm/PR-40410.md) | [Model Runner V2] Skip attention metadata rebuild before draft prefill | 2026-04-20 |  | attention, decode, prefill |
| [#40412](../sources/prs/vllm/PR-40412.md) | fused_moe: treat NIXL EP as batched experts | 2026-04-20 | kernel-fusion | fp4, kernel-fusion, moe |
| [#40273](../sources/prs/vllm/PR-40273.md) | Fix MoE backend selection for LoRA (unquantized MoE) | 2026-04-19 | kernel-fusion | kernel-fusion, moe, quantization |
| [#40193](../sources/prs/vllm/PR-40193.md) | [Bugfix] Make Attention Backend Auto-Selection Batch-Invariance-Aware | 2026-04-18 |  | attention, mla |
| [#40082](../sources/prs/vllm/PR-40082.md) | Integrate flashinfer b12x MoE and FP4 GEMM kernels for SM120/121 | 2026-04-17 | kernel-fusion | fp4, gemm, kernel-fusion |
| [#40109](../sources/prs/vllm/PR-40109.md) | [XPU] fix MoE triton backend in online fp8 quantization  | 2026-04-17 | kernel-fusion | fp8, kernel-fusion, moe |
| [#40131](../sources/prs/vllm/PR-40131.md) | [Bugfix] moe lora align kernel grid | 2026-04-17 |  | moe |
| [#40159](../sources/prs/vllm/PR-40159.md) | [MyPy] Enable mypy for `vllm/model_executor/layers/` | 2026-04-17 | kernel-fusion | attention, fp4, gemm |
| [#39959](../sources/prs/vllm/PR-39959.md) | Update flashinfer to 0.6.8 | 2026-04-16 | kernel-fusion | attention, kernel-fusion, moe |
| [#39988](../sources/prs/vllm/PR-39988.md) | [Bugfix] Fix turboquant FP8 cast failure for BF16 models on Ampere GPUs | 2026-04-16 |  | attention, fp8 |
| [#39999](../sources/prs/vllm/PR-39999.md) | [ROCm] Cast score correction bias tensor during model construction for DeepSeek/Kimi-K2 | 2026-04-16 | kernel-fusion | kernel-fusion, moe |
| [#40057](../sources/prs/vllm/PR-40057.md) | [Bugfix] Temporarily disable B200 fp4 MoE layer tests | 2026-04-16 |  | fp4, moe |
| [#39904](../sources/prs/vllm/PR-39904.md) | Add tuned triton fused_moe configs on H100 for gpt-oss | 2026-04-15 | kernel-fusion | kernel-fusion, moe |
| [#39930](../sources/prs/vllm/PR-39930.md) | [Attention][Spec Decode] Allow independent drafter attention backend selection | 2026-04-15 |  | attention, decode |
| [#39752](../sources/prs/vllm/PR-39752.md) | add warning when FP8 KV cache misses prefill query quantization | 2026-04-14 |  | attention, fp8, mla |
| [#39778](../sources/prs/vllm/PR-39778.md) | [Quantization][Autoround][Toolkit] Add W4A16 Support | 2026-04-14 |  | quantization |
| [#39780](../sources/prs/vllm/PR-39780.md) | [Bugfix] Reject empty tools array with HTTP 400 | 2026-04-14 |  | moe |
| [#39796](../sources/prs/vllm/PR-39796.md) | [Bugfix] add support for 'num_attention_groups' in ModelArchConfigConvertorBase for Step3p5 | 2026-04-14 |  | attention |
| [#39820](../sources/prs/vllm/PR-39820.md) | [Bug] Fix batch invariance nvfp4 support | 2026-04-14 |  | fp4, nvfp4 |
| [#39825](../sources/prs/vllm/PR-39825.md) | [Bugfix] Disable FlashInfer CUTLASS MoE on SM121 (DGX Spark) | 2026-04-14 | kernel-fusion | kernel-fusion, moe |
| [#39676](../sources/prs/vllm/PR-39676.md) | [XPU] properly handle q_descale on XPU as quant query input not supported | 2026-04-13 |  | attention |
| [#39688](../sources/prs/vllm/PR-39688.md) | [fix][MOE] Fix MOE experts `intermediate_size` dimension not being narrowed before weight loading | 2026-04-13 | kernel-fusion | kernel-fusion, moe |
| [#39707](../sources/prs/vllm/PR-39707.md) | [Bugfix] Fix mismatch between global and local attention heads in tensor-parallel mode for param2moe model | 2026-04-13 |  | attention, moe |
| [#39717](../sources/prs/vllm/PR-39717.md) | [Bugfix] Reject non-nvfp4 dtypes when using the flashinfer_nvlink_one_sided all2all backend | 2026-04-13 | kernel-fusion | fp4, kernel-fusion, moe |
| [#39724](../sources/prs/vllm/PR-39724.md) | [Bugfix][NIXL] Fix `_logical_to_kernel_block_ids` conversion for non-mamba models | 2026-04-13 |  | gemm |
| [#39596](../sources/prs/vllm/PR-39596.md) | [Mooncake] Fix mixed MLA+Eagle block-size validation | 2026-04-12 |  | mla |
| [#39604](../sources/prs/vllm/PR-39604.md) | [Quantization] [Refactor] Create special "GptOssMxfp4MoeMethod" | 2026-04-12 | kernel-fusion | fp4, kernel-fusion, moe |
| [#39612](../sources/prs/vllm/PR-39612.md) | [Migration] Migrate GGUF quantization support to plugin | 2026-04-12 | kernel-fusion | gemm, kernel-fusion, moe |
| [#39644](../sources/prs/vllm/PR-39644.md) | [Bugfix] [Tests] Enforce `out` tensor device in `kernel/moe/test_cutedsl_moe.py` | 2026-04-12 |  | moe |
| [#39510](../sources/prs/vllm/PR-39510.md) | [Kernel] Support TRTLLM GEN NVFP4 MoE for non-512-aligned hidden dims via weight padding | 2026-04-10 | kernel-fusion | fp4, kernel-fusion, moe |
| [#39542](../sources/prs/vllm/PR-39542.md) | [Bugfix] Fix tensor shape mismatch in sparse attention with speculative decoding | 2026-04-10 |  | attention |
| [#39418](../sources/prs/vllm/PR-39418.md) | [Bugfix][CT] Fix KV cache scale handling | 2026-04-09 |  | quantization |
| [#39458](../sources/prs/vllm/PR-39458.md) | [MLA] Optimize mla indexer prepare uniform decode for MTP > 1 | 2026-04-09 |  | attention, decode, mla |
| [#39315](../sources/prs/vllm/PR-39315.md) | [Bugfix] FlashInfer MXINT4 MoE crashes, missing do_finalize | 2026-04-08 |  | moe, quantization |
| [#39322](../sources/prs/vllm/PR-39322.md) | [Feature] Batch invariant nvfp4 linear support | 2026-04-08 |  | fp4, nvfp4, quantization |
| [#39336](../sources/prs/vllm/PR-39336.md) | [Attention] Improve attention benchmarks: configs and profiling | 2026-04-08 |  | attention, decode, mla |
| [#39353](../sources/prs/vllm/PR-39353.md) | [Model Runner V2] Fix flex attention kv blocks calculation issue | 2026-04-08 |  | attention |
| [#39361](../sources/prs/vllm/PR-39361.md) | Fix NUMA binding on non-CDMM Grace-Blackwell systems | 2026-04-08 |  | gemm |
| [#39136](../sources/prs/vllm/PR-39136.md) | [ROCm][Quantization][2/N] Refactor quark_moe w4a8 w/ oracle  | 2026-04-07 | kernel-fusion | fp4, kernel-fusion, moe |
| [#39141](../sources/prs/vllm/PR-39141.md) | [Perf] Update TRTLLM supported MoE routing methods | 2026-04-07 | kernel-fusion | fp4, fp8, kernel-fusion |
| [#39177](../sources/prs/vllm/PR-39177.md) | [ROCm][Perf] Expose AITER MoE sorting dispatch policy via env var | 2026-04-07 | kernel-fusion | kernel-fusion, moe |
| [#39183](../sources/prs/vllm/PR-39183.md) | perf(moe): add tuned fused_moe config for RTX PRO 6000 Blackwell Server Edition | 2026-04-07 | kernel-fusion | fp8, kernel-fusion, moe |
| [#39205](../sources/prs/vllm/PR-39205.md) | [Refactor] Move MXFP8 GEMM management into MxFp8LinearKernel | 2026-04-07 |  | fp8, gemm, quantization |
| [#39222](../sources/prs/vllm/PR-39222.md) | [Bugfix] Replace code that disabled shared expert overlap | 2026-04-07 | kernel-fusion | kernel-fusion, moe |
| [#39225](../sources/prs/vllm/PR-39225.md) | [Bug] Fix rocm sparse attn indexer issue | 2026-04-07 |  | attention, mla |
| [#39054](../sources/prs/vllm/PR-39054.md) | [Bug] Fix Trtllm Fp8 MoE Weight Shuffle Memory Fragamentation | 2026-04-06 |  | fp8, moe, quantization |
| [#39058](../sources/prs/vllm/PR-39058.md) | [Kernel] Implement CUDA kernel for ReLUSquaredActivation (relu^2) | 2026-04-06 |  | gemm |
| [#39064](../sources/prs/vllm/PR-39064.md) | [Bugfix] Fix GDN FLA kernel crashes with NULL_BLOCK_ID=0 CUDA graph padding | 2026-04-06 | kernel-fusion | kernel-fusion |
| [#39088](../sources/prs/vllm/PR-39088.md) | [XPU] Quick fix for TritonMLA to remove cuda hardcode | 2026-04-06 | kernel-fusion | attention, kernel-fusion, mla |
| [#39119](../sources/prs/vllm/PR-39119.md) | [ROCm] Align AiterFlashAttentionImpl attn_type check with backend | 2026-04-06 |  | attention |
| [#39129](../sources/prs/vllm/PR-39129.md) | [Refactor] Move NVFP4 GEMM management into NvFp4LinearKernel | 2026-04-06 |  | fp4, gemm, nvfp4 |
| [#39005](../sources/prs/vllm/PR-39005.md) | [MoE] Move DEEP_GEMM into experts/ subdirectory | 2026-04-05 | kernel-fusion | fp8, gemm, kernel-fusion |
| [#39007](../sources/prs/vllm/PR-39007.md) | [MoE] Move GPT OSS Triton kernel experts into fused_moe/experts/ | 2026-04-05 | kernel-fusion | fp4, kernel-fusion, moe |
| [#39009](../sources/prs/vllm/PR-39009.md) | [MoE] Move remaining PrepareAndFinalize to prepare finalize folder | 2026-04-05 | kernel-fusion | gemm, kernel-fusion, moe |
| [#39045](../sources/prs/vllm/PR-39045.md) | [Gemma4] Support quantized MoE  | 2026-04-05 |  | gemm, moe, quantization |
| [#38960](../sources/prs/vllm/PR-38960.md) | [MoE Refactor] Split up compressed_tensors_moe.py | 2026-04-04 |  | fp4, fp8, moe |
| [#38981](../sources/prs/vllm/PR-38981.md) | [Perf][GDN] Align TMA usage with upstream FLA | 2026-04-04 |  | tma |
| [#38989](../sources/prs/vllm/PR-38989.md) | [Bug] Fix routing bias dtype for trtllm per-block fp8 moe | 2026-04-04 | kernel-fusion | fp8, kernel-fusion, moe |
| [#38990](../sources/prs/vllm/PR-38990.md) | [Bugfix][MoE] Fix 6-8% decode regression: prefer multi-stream shared expert overlap | 2026-04-04 | kernel-fusion | decode, kernel-fusion, moe |
| [#38993](../sources/prs/vllm/PR-38993.md) | [Perf] Change Trtllm fp8 MoE to use Shuffled Weights and BlockMajorK Layout | 2026-04-04 | kernel-fusion | fp8, gemm, kernel-fusion |
| [#38995](../sources/prs/vllm/PR-38995.md) | [Quantization] - Layerwise reloading of Attention/KV quantized models | 2026-04-04 |  | attention, quantization |
| [#39002](../sources/prs/vllm/PR-39002.md) | [Bugfix] Fix FlashInfer crash with kv_cache_dtype_skip_layers | 2026-04-04 | kernel-fusion | attention, kernel-fusion |
| [#38859](../sources/prs/vllm/PR-38859.md) | [Bugfix] Re-enable Renormalize routing for TRT-LLM MoE experts | 2026-04-03 | kernel-fusion | fp8, kernel-fusion, moe |
| [#38865](../sources/prs/vllm/PR-38865.md) | [Refactor] Improve indexer decode path metadata preparation | 2026-04-03 |  | attention, decode, mla |
| [#38879](../sources/prs/vllm/PR-38879.md) | [Gemma4] Enable Fast Prefill Optimization | 2026-04-03 |  | gemm, prefill |
| [#38922](../sources/prs/vllm/PR-38922.md) | [Bugfix] Fix broken explicit unquantized kv cache dtype support | 2026-04-03 |  | attention, fp8, quantization |
| [#38770](../sources/prs/vllm/PR-38770.md) | [CPU] Support gelu act in cpu_fused_moe | 2026-04-02 | kernel-fusion | attention, kernel-fusion, moe |
| [#38791](../sources/prs/vllm/PR-38791.md) | [Bugfix] Fix test mocks after SM100 restriction in #38730 | 2026-04-02 |  | attention |
| [#38794](../sources/prs/vllm/PR-38794.md) | [Perf] Reduce H2D pageable memory copies | 2026-04-02 |  | attention |
| [#38810](../sources/prs/vllm/PR-38810.md) | [LMCache][MP] optimize save when mla enabled | 2026-04-02 |  | mla |
| [#38814](../sources/prs/vllm/PR-38814.md) | [FlashAttention] Symlink FA4 instead of copying when using `VLLM_FLASH_ATTN_SRC_DIR` | 2026-04-02 |  | attention |
| [#38815](../sources/prs/vllm/PR-38815.md) | [Quant] add CompressedTensorsW8A8Mxfp8 for linear and MoE layers | 2026-04-02 |  | fp8, moe, quantization |
| [#38819](../sources/prs/vllm/PR-38819.md) | [Attention][MLA] Re-enable FA4 as default MLA prefill backend | 2026-04-02 |  | attention, mla, prefill |
| [#38822](../sources/prs/vllm/PR-38822.md) | [Attention] Add head_dim=512 support for FlashInfer trtllm attention backend | 2026-04-02 |  | attention |
| [#38832](../sources/prs/vllm/PR-38832.md) | [Bugfix] Fix NVFP4+MTP crash: force unquantized mtp.fc for Qwen3.5 | 2026-04-02 |  | fp4, nvfp4, quantization |
| [#38835](../sources/prs/vllm/PR-38835.md) | [Attention] relax the head dim 512 and paged kv for sm90+FA4 | 2026-04-02 |  | attention |
| [#38682](../sources/prs/vllm/PR-38682.md) | [XPU] add  xpu backend implementation of mxfp8 quant | 2026-04-01 |  | fp8, quantization |
| [#38690](../sources/prs/vllm/PR-38690.md) | [FA4] Update flash-attention to latest upstream FA4 | 2026-04-01 |  | attention, flash-attention |
| [#38730](../sources/prs/vllm/PR-38730.md) | [Bugfix] Restrict TRTLLM attention to SM100, fixing GB300 (SM103) hang | 2026-04-01 |  | attention |
| [#38615](../sources/prs/vllm/PR-38615.md) | [ROCm] Fix aiter persistent mode mla with q/o nhead<16 for kimi-k2.5 tp8 | 2026-03-31 | persistent-kernel | attention, mla, persistent-kernel |
| [#38631](../sources/prs/vllm/PR-38631.md) | Fix MLA runs when use_inductor_graph_partition=True | 2026-03-31 |  | attention, mla |
| [#38491](../sources/prs/vllm/PR-38491.md) | [XPU] Fix spec-decode UTs under tests/v1/spec_decode | 2026-03-30 |  | decode |
| [#38502](../sources/prs/vllm/PR-38502.md) | [ROCm] Cap Triton paged attention block size to fix ROCm shared memory OOM | 2026-03-30 |  | attention, decode, prefill |
| [#38562](../sources/prs/vllm/PR-38562.md) | [Bugfix][MLA] Change default SM100 MLA prefill backend back to TRT-LLM | 2026-03-30 |  | attention, mla, prefill |
| [#38573](../sources/prs/vllm/PR-38573.md) | [Compile] Fix nvfp4 compile warning | 2026-03-30 |  | fp4, nvfp4, quantization |
| [#38460](../sources/prs/vllm/PR-38460.md) | [Perf] Batch KV cache swap copies via cuMemcpyBatchAsync | 2026-03-29 |  | gemm |
| [#38463](../sources/prs/vllm/PR-38463.md) | [Quantization] Consolidate experts_int8 with fp8 online quantization | 2026-03-29 | kernel-fusion | fp8, kernel-fusion, moe |
| [#38479](../sources/prs/vllm/PR-38479.md) | [Attention Backend] TurboQuant: 2-bit KV cache compression with 4x capacity | 2026-03-29 |  | attention, decode, quantization |
| [#38423](../sources/prs/vllm/PR-38423.md) | [NVIDIA] Bugfix NVFP4 DGX Spark and RTX50 | 2026-03-28 | kernel-fusion | fp4, fp8, kernel-fusion |
| [#38427](../sources/prs/vllm/PR-38427.md) | [Bugfix] Enable batch-invariant Triton matmul on all Ampere GPUs (SM 8x)  | 2026-03-28 |  | gemm |
| [#38442](../sources/prs/vllm/PR-38442.md) | [QeRL] Fix online quantized reloading | 2026-03-28 |  | fp8, quantization |
| [#38311](../sources/prs/vllm/PR-38311.md) | [Model Runner V2] Rebuild attention metadata before eagle decode full… | 2026-03-27 |  | attention, decode |
| [#38325](../sources/prs/vllm/PR-38325.md) | [Kernel] Add swapAB support for SM120 CUTLASS blockwise FP8 GEMM  | 2026-03-27 |  | fp8, gemm, quantization |
| [#38329](../sources/prs/vllm/PR-38329.md) | [MoE] Add RoutingMethodType.Simulated to TRT-LLM FP8/NVFP4 kernel allowlists | 2026-03-27 | kernel-fusion | fp4, fp8, kernel-fusion |
| [#38361](../sources/prs/vllm/PR-38361.md) | [GDN] Eliminate GPU->CPU sync in prepare_chunk_indices during prefill | 2026-03-27 |  | attention, prefill |
| [#38391](../sources/prs/vllm/PR-38391.md) | [CI Bugfix] Pre-download missing FlashInfer headers in Docker build | 2026-03-27 |  | gemm |
| [#38251](../sources/prs/vllm/PR-38251.md) | [Quantization] Add FlashInfer CuteDSL batched experts backend for NVFP4 MoE | 2026-03-26 | kernel-fusion | fp4, kernel-fusion, moe |
| [#38288](../sources/prs/vllm/PR-38288.md) | [Quant] Consolidate GPTQ: rename gptq_marlin.py to auto_gptq.py | 2026-03-26 | kernel-fusion | kernel-fusion, moe, quantization |
| [#38050](../sources/prs/vllm/PR-38050.md) | [MoE Kernel] Flashinfer nvfp4 cutedsl moe kernel integration | 2026-03-25 | kernel-fusion | fp4, kernel-fusion, moe |
| [#38083](../sources/prs/vllm/PR-38083.md) | [Bugfix] Fix DeepGemm E8M0 accuracy degradation for Qwen3.5 FP8 on Blackwell | 2026-03-25 |  | fp4, fp8, gemm |
| [#38148](../sources/prs/vllm/PR-38148.md) | Fix NaN from stale FP4 scale padding in create_fp4_scale_tensor | 2026-03-25 |  | fp4 |
| [#37940](../sources/prs/vllm/PR-37940.md) | [NIXL][BUG] Fix Triton heterogeneous TP | 2026-03-24 |  | attention |
| [#37948](../sources/prs/vllm/PR-37948.md) | [Perf] triton bilinear_pos_embed kernel for ViT | 2026-03-24 |  | gemm |
| [#37880](../sources/prs/vllm/PR-37880.md) | [Feature] Support per-draft-model MoE backend via `--speculative-config` | 2026-03-23 |  | decode, moe |
| [#37887](../sources/prs/vllm/PR-37887.md) | [ROCm][perf] fix Aiter sparse MLA with MTP>1 | 2026-03-23 |  | decode, mla |
| [#37759](../sources/prs/vllm/PR-37759.md) | [MoE] Move FlashInfer CuteDSL experts into fused_moe/experts/ | 2026-03-21 | kernel-fusion | fp4, kernel-fusion, moe |
| [#37695](../sources/prs/vllm/PR-37695.md) | [Perf] Use torch compile to fuse pack topk in trtllm moe | 2026-03-20 | kernel-fusion | fp4, fp8, kernel-fusion |
| [#37718](../sources/prs/vllm/PR-37718.md) | [Bug] Fix fp8 deepgemm batch invariant | 2026-03-20 |  | fp8, gemm, quantization |
| [#37719](../sources/prs/vllm/PR-37719.md) | [Test] Only Run MLA model when user explicitly set for batch invariance | 2026-03-20 |  | mla |
| [#37725](../sources/prs/vllm/PR-37725.md) | [Bugfix] Preserve CUDA arch suffix (a/f) for SM12x — fixes NVFP4 NaN on desktop Blackwell | 2026-03-20 |  | fp4, nvfp4 |
| [#37502](../sources/prs/vllm/PR-37502.md) | [Bugfix] Fix marlin nvfp4 rescaling | 2026-03-19 |  | fp4, nvfp4, quantization |
| [#37503](../sources/prs/vllm/PR-37503.md) | [4/n] Migrate FP4/W4A8 CUTLASS kernels to torch stable ABI | 2026-03-19 | epilogue-fusion, kernel-fusion | epilogue-fusion, fp4, kernel-fusion |
| [#37519](../sources/prs/vllm/PR-37519.md) | refactor: abstract deepgemm support into platform | 2026-03-19 |  | gemm |
| [#37536](../sources/prs/vllm/PR-37536.md) | Fix KV Offloading + MLA AssertionError by using num_kv_heads=1 in cpu… | 2026-03-19 |  | mla |
| [#37539](../sources/prs/vllm/PR-37539.md) | [Performance] Remove unnecessary zero-fill of MLA decode output tensor in Aiter backend | 2026-03-19 |  | attention, decode, mla |
| [#37547](../sources/prs/vllm/PR-37547.md) | [Bugfix][ROCm] Fix lru_cache on paged_mqa_logits_module | 2026-03-19 |  | attention, mla |
| [#37565](../sources/prs/vllm/PR-37565.md) | [Bugfix] Disable --calculate-kv-scales for hybrid GDN/Mamba+Attention… | 2026-03-19 |  | attention |
| [#37605](../sources/prs/vllm/PR-37605.md) | [Bugfix] Disable monolithic TRTLLM MoE for Renormalize routing (#37591) | 2026-03-19 | kernel-fusion | fp8, kernel-fusion, moe |
| [#37606](../sources/prs/vllm/PR-37606.md) | [ROCm][Bugfix] fix cache block size mismatch for aiter unified attention | 2026-03-19 |  | attention |
| [#37364](../sources/prs/vllm/PR-37364.md) | [Model Runner V2] fix draft attention metadata generation | 2026-03-18 |  | attention, decode |
| [#37373](../sources/prs/vllm/PR-37373.md) | [torch.compile] Refactor Attention Quant Fusion Pass and Remove Boilerplate | 2026-03-18 | kernel-fusion | attention, kernel-fusion |
| [#37421](../sources/prs/vllm/PR-37421.md) | [Perf][Kernel] Persistent TopK scheduler: unified CUDAGraph-safe kernel with dynamic per-row dispatch - DeepSeek-V3.2 DSA decode | 2026-03-18 | persistent-kernel | attention, decode, mla |
| [#37465](../sources/prs/vllm/PR-37465.md) | [Bugfix] Remove assertion for NVFP4 scale dynamic range | 2026-03-18 |  | fp4, nvfp4, quantization |
| [#37475](../sources/prs/vllm/PR-37475.md) | [BugFix] Allow qk_nope_head_dim=192 in FlashInfer MLA backend checks | 2026-03-18 |  | attention, mla |
| [#37252](../sources/prs/vllm/PR-37252.md) | [Perf] Set Flashinfer sparse MLA as default backend for FP8 kv cache | 2026-03-17 |  | attention, fp8, mla |
| [#37303](../sources/prs/vllm/PR-37303.md) | [Attention] Support distinguishing between short extends and decodes | 2026-03-17 |  | attention, decode, prefill |
| [#37320](../sources/prs/vllm/PR-37320.md) | [Kernel] Add non-gated support for NVFP4 CUTLASS MoE | 2026-03-17 | kernel-fusion | fp4, kernel-fusion, moe |
| [#37322](../sources/prs/vllm/PR-37322.md) | [Bugfix] Fix EP weight filter breaking EPLB and NVFP4 accuracy | 2026-03-17 |  | fp4, nvfp4 |
| [#37143](../sources/prs/vllm/PR-37143.md) | [XPU] support MLA model on Intel GPU | 2026-03-16 |  | attention, fp8, mla |
| [#37199](../sources/prs/vllm/PR-37199.md) | [Misc] Add `float16` to `CacheDType` | 2026-03-16 |  | attention, mla |
| [#37205](../sources/prs/vllm/PR-37205.md) | [Kernel] Add gpt-oss Router GEMM kernel | 2026-03-16 | kernel-fusion | gemm, kernel-fusion, moe |
| [#37214](../sources/prs/vllm/PR-37214.md) | Fix minimax m2.5 nvfp4 kv scales weight loading | 2026-03-16 |  | fp4, nvfp4 |
| [#37217](../sources/prs/vllm/PR-37217.md) | [MoE/EPLB] Fix FlashInfer nvfp4 experts + EPLB correctness | 2026-03-16 | kernel-fusion | fp4, kernel-fusion, moe |
| [#37228](../sources/prs/vllm/PR-37228.md) | [ROCM][Bugfix] Use correct stride in cp_mha_gather_cache_kernel for hybrid model (#37228) | 2026-03-16 |  | attention |
| [#37231](../sources/prs/vllm/PR-37231.md) | [Bugfix] Expand quantization method support in perf metrics | 2026-03-16 |  | quantization |
| [#37233](../sources/prs/vllm/PR-37233.md) | [UX] Add flashinfer-cubin as CUDA default dep | 2026-03-16 |  | gemm |
| [#37090](../sources/prs/vllm/PR-37090.md) | [Bugfix] Disable cross-layer KV cache for MLA attention backends | 2026-03-15 |  | attention, mla |
| [#37115](../sources/prs/vllm/PR-37115.md) | [Benchmark] Improvements to attention benchmark script | 2026-03-15 |  | attention, decode, mla |
| [#37128](../sources/prs/vllm/PR-37128.md) | [MoE Refactor] Mxfp4 oracle rebased | 2026-03-15 | kernel-fusion | fp4, kernel-fusion, moe |
| [#37054](../sources/prs/vllm/PR-37054.md) | [Bugfix] Fix KV scales inconsistency in fp8 MLA & FlashInfer kv_cache_dtype "auto" leading to gibberish | 2026-03-14 |  | attention, fp8, mla |
| [#36976](../sources/prs/vllm/PR-36976.md) | [Bugfix][LoRA] Fix  Qwen35 LoRA | 2026-03-13 |  | moe |
| [#36982](../sources/prs/vllm/PR-36982.md) | [MTP][Sparse MLA] Take advantage of native MTP support in indexer when possible | 2026-03-13 |  | attention, mla |
| [#36845](../sources/prs/vllm/PR-36845.md) | [ROCm] Fix KV copy methods and auto-select attention backend for ROCm | 2026-03-12 |  | attention, decode |
| [#36846](../sources/prs/vllm/PR-36846.md) | [ROCm] Validate block_size for explicitly selected attention backends | 2026-03-12 |  | attention |
| [#36847](../sources/prs/vllm/PR-36847.md) | [Feat][Spec Decode] DFlash | 2026-03-12 |  | attention, decode |
| [#36876](../sources/prs/vllm/PR-36876.md) | [Bugfix] Fix FlashInfer GDN warmup ValueError on SM90 GPUs | 2026-03-12 |  | gemm |
| [#36931](../sources/prs/vllm/PR-36931.md) | [Feat][Bugfix] Enable additional dimension for Flashinfer MLA and fix routing dtype | 2026-03-12 |  | attention, mla |
| [#36725](../sources/prs/vllm/PR-36725.md) | [Bug][MoE] Fix TRTLLM NVFP4 Routing Kernel Precision | 2026-03-11 | kernel-fusion | fp4, kernel-fusion, moe |
| [#36728](../sources/prs/vllm/PR-36728.md) | [Bug][MoE] Strengthen _supports_current_device() checks in the TRTLLM FP8, NVFP4, and FlashInfer CuteDSL MoE experts | 2026-03-11 | kernel-fusion | fp4, fp8, kernel-fusion |
| [#36768](../sources/prs/vllm/PR-36768.md) | Update Flashinfer to 0.6.6 | 2026-03-11 |  | gemm |
| [#36574](../sources/prs/vllm/PR-36574.md) | [ROCm] Utilize persistent MLA kernel from AITER | 2026-03-10 | persistent-kernel | attention, mla, persistent-kernel |
| [#36599](../sources/prs/vllm/PR-36599.md) | [Bugfix] Warm up Triton autotuner for GDN layers during V1 profiling | 2026-03-10 |  | gemm |
| [#36647](../sources/prs/vllm/PR-36647.md) | [GDN] add a config for gdn kernel selection | 2026-03-10 |  | gemm |
| [#36673](../sources/prs/vllm/PR-36673.md) | [Misc][Attention] Clean up unused method in `CPU_ATTN` | 2026-03-10 |  | attention |
| [#36674](../sources/prs/vllm/PR-36674.md) | [Bug] Fix FlashInfer MNNVL socket collisions under concurrent vLLM jobs | 2026-03-10 |  | gemm |
| [#36681](../sources/prs/vllm/PR-36681.md) | [ROCm][Perf] Allow MTP lens > 1 in Sparse MLA | 2026-03-10 |  | decode, mla |
| [#36684](../sources/prs/vllm/PR-36684.md) | fix(kv-cache): increase hybrid attention grouping threshold from 1.25 to 1.5 | 2026-03-10 |  | attention |
| [#36702](../sources/prs/vllm/PR-36702.md) | [ROCm] Attention selector reordering | 2026-03-10 |  | attention, decode, prefill |
| [#36723](../sources/prs/vllm/PR-36723.md) | [DSV3.2][MTP] Optimize Indexer MTP handling | 2026-03-10 |  | attention, mla |
| [#36458](../sources/prs/vllm/PR-36458.md) | [XPU] Support block fp8 moe by fallback to TritonExpert on XPU | 2026-03-09 | kernel-fusion | fp8, kernel-fusion, moe |
| [#36466](../sources/prs/vllm/PR-36466.md) | feat(attention): extract KV-cache update from FlashAttentionDiffKV ba… | 2026-03-09 |  | attention |
| [#36518](../sources/prs/vllm/PR-36518.md) | [Kernel] Fuse FP8 output quantization into merge_attn_states | 2026-03-09 | kernel-fusion | attention, fp8, kernel-fusion |
| [#36519](../sources/prs/vllm/PR-36519.md) | [Bugfix][Sparse MLA] report indexer CG support properly | 2026-03-09 |  | attention, mla |
| [#36559](../sources/prs/vllm/PR-36559.md) | [Kernel] Add swap AB optimization to fused_moe_kernel | 2026-03-09 | kernel-fusion | kernel-fusion, moe |
| [#36307](../sources/prs/vllm/PR-36307.md) | [Perf] Add TRTLLM FP8 MoE Modular Kernel | 2026-03-07 | kernel-fusion | fp8, kernel-fusion, moe |
| [#36318](../sources/prs/vllm/PR-36318.md) | Disable cascade attention by default | 2026-03-07 |  | attention |
| [#36361](../sources/prs/vllm/PR-36361.md) | Kimi k2.5 MLA based eagle3 | 2026-03-07 |  | decode, mla |
| [#36205](../sources/prs/vllm/PR-36205.md) | [mla] Support fused FP8/NVFP4 output quantization in MLA attention (#35792) | 2026-03-06 | kernel-fusion | attention, fp4, fp8 |
| [#36209](../sources/prs/vllm/PR-36209.md) | docs: fix wrong cc in int8.md | 2026-03-06 |  | quantization |
| [#36282](../sources/prs/vllm/PR-36282.md) | mla: don't update kv cache on dummy forwards | 2026-03-06 |  | attention, mla |
| [#36129](../sources/prs/vllm/PR-36129.md) | [LMCache] Pass TP size in lookup for MLA multi-reader locking | 2026-03-05 |  | mla |
| [#36146](../sources/prs/vllm/PR-36146.md) | [Bugfix] Disable FlashInfer TRTLLM BF16 path for non-gated MoE | 2026-03-05 | kernel-fusion | kernel-fusion, moe |
| [#36161](../sources/prs/vllm/PR-36161.md) | Add 320 dimension size support to MLA | 2026-03-05 |  | attention, mla |
| [#36162](../sources/prs/vllm/PR-36162.md) | [Mamba] Flashinfer selective_state_update | 2026-03-05 |  | fp4, nvfp4 |
| [#36178](../sources/prs/vllm/PR-36178.md) | [Bugfix][MLA] Add logits size budget to sparse indexer prefill chunking | 2026-03-05 |  | attention, mla, prefill |
| [#35986](../sources/prs/vllm/PR-35986.md) | Add support for ModelOpt MXFP8 MoE models | 2026-03-04 | kernel-fusion | fp8, kernel-fusion, moe |
| [#36017](../sources/prs/vllm/PR-36017.md) | [Bugfix] Fix passing of activation_type to trtllm fused MoE NVFP4 and FP8 | 2026-03-04 | kernel-fusion | fp4, fp8, kernel-fusion |
| [#36022](../sources/prs/vllm/PR-36022.md) | [Kernel] Add FlashInfer MoE A2A Kernel | 2026-03-04 | kernel-fusion | gemm, kernel-fusion, moe |
| [#36042](../sources/prs/vllm/PR-36042.md) | Fix CUDA graph decode capture crash in AITER FlashAttention | 2026-03-04 |  | attention, decode |
| [#36059](../sources/prs/vllm/PR-36059.md) | [BugFix] Fallback from FA4->FA2 for Batch Invariance | 2026-03-04 |  | attention |
| [#35849](../sources/prs/vllm/PR-35849.md) | [Bugfix] Fix score layer quantization for sequence classification models  - Qwen3 (VL) Reranker | 2026-03-03 |  | quantization |
| [#35850](../sources/prs/vllm/PR-35850.md) | [ROCm] Support MLA with nhead<16 and FP8 KV cache for TP=8 (Kimi K2.5/Linear) | 2026-03-03 |  | attention, fp8, mla |
| [#35891](../sources/prs/vllm/PR-35891.md) | [Perf] Support FP8 KV cache for Flashinfer MLA Sparse | 2026-03-03 |  | attention, fp8, mla |
| [#35927](../sources/prs/vllm/PR-35927.md) | [MoE] Move PF Methods to Folder | 2026-03-03 | kernel-fusion | gemm, kernel-fusion, moe |
| [#35733](../sources/prs/vllm/PR-35733.md) | [NVFP4] Support NVFP4 dense models from `modelopt` and `compressed-tensors` on AMD Instinct MI300, MI355X and Hopper through emulation | 2026-03-02 |  | fp4, nvfp4, quantization |
| [#35744](../sources/prs/vllm/PR-35744.md) | Fix routed experts capture for hybrid models (Mamba + Attention) | 2026-03-02 |  | attention |
| [#35751](../sources/prs/vllm/PR-35751.md) | [MoE][Perf] Wrap DSV3 QKVAProj GEMM in custom op for torch.compile | 2026-03-02 |  | gemm, moe |
| [#35753](../sources/prs/vllm/PR-35753.md) | [Mamba] Add stochastic rounding support | 2026-03-02 |  | gemm |
| [#35777](../sources/prs/vllm/PR-35777.md) | [Kernel] Add fused_sigmoid_gating_delta_rule_update kernel for Qwen3 Next | 2026-03-02 | kernel-fusion | kernel-fusion |
| [#35422](../sources/prs/vllm/PR-35422.md) | [Performance] Extract KV cache update op from flashinfer forward | 2026-02-26 |  | attention |
| [#35430](../sources/prs/vllm/PR-35430.md) | [Bugfix] Fix KV Scale loading for MLA Models | 2026-02-26 |  | mla, quantization |
| [#35448](../sources/prs/vllm/PR-35448.md) | [Quant][Feature] Support online MXFP8 quantization for MoE and dense models | 2026-02-26 | kernel-fusion | fp8, kernel-fusion, moe |
| [#35290](../sources/prs/vllm/PR-35290.md) | [Attention][Perf] Optimize cp_gather_and_upconvert_fp8_kv_cache - DeepSeek-v3.2 | 2026-02-25 |  | attention, fp8 |
| [#35156](../sources/prs/vllm/PR-35156.md) | [BUGFIX][Qwen3.5] Hardcode `mlp.gate` as not quantizable  | 2026-02-24 |  | quantization |
| [#35157](../sources/prs/vllm/PR-35157.md) | [Linear Attention] fix bug for linear attention + prefix caching + reset_prefix_cache | 2026-02-24 |  | attention |
| [#35210](../sources/prs/vllm/PR-35210.md) | [BugFix] Fix fp4 quant kernel on CUDA 12.8 | 2026-02-24 | kernel-fusion | fp4, kernel-fusion, nvfp4 |
| [#35219](../sources/prs/vllm/PR-35219.md) | [BUGFIX][Mamba][Qwen3.5] Zero freed SSM cache blocks on GPU | 2026-02-24 |  | attention |
| [#35075](../sources/prs/vllm/PR-35075.md) | [Bug][DSV3.2] Always prepare metadata for DeepGEMM Sparse Attention | 2026-02-23 |  | attention, gemm, mla |
| [#35121](../sources/prs/vllm/PR-35121.md) | [Performance] Cublas Bf16 Gate with Fp32 Output | 2026-02-23 | kernel-fusion | gemm, kernel-fusion, moe |
| [#35123](../sources/prs/vllm/PR-35123.md) | [Bugfix] Fix DSV3 kernels breaking _C and _moe_C on unsupported arches | 2026-02-23 | kernel-fusion | gemm, kernel-fusion, moe |
| [#35036](../sources/prs/vllm/PR-35036.md) | [Model Runner V2] Support attention group | 2026-02-22 |  | attention, decode |
| [#35047](../sources/prs/vllm/PR-35047.md) | add mixed precision support for modelopt | 2026-02-22 |  | quantization |
| [#35053](../sources/prs/vllm/PR-35053.md) | Integrate flashinfer mm_mxfp8 in ModelOpt MXFP8 | 2026-02-22 |  | fp8, quantization |
| [#34871](../sources/prs/vllm/PR-34871.md) | [Bugfix] Fix GDN attention crash with mixed decode/spec-decode batches | 2026-02-19 |  | attention, decode |
| [#34900](../sources/prs/vllm/PR-34900.md) | [Model Bash][DSR1] Add selective dynamic shape marking for CustomOp | 2026-02-19 |  | attention, mla |
| [#34917](../sources/prs/vllm/PR-34917.md) | [Attention][Perf][Kernel] Replace torch.cat with vectorized CUDA kernel MLA query concat - DeepSeek-V3.2 | 2026-02-19 |  | attention, mla |
| [#34924](../sources/prs/vllm/PR-34924.md) | [Perf] Enable FlashInfer DeepGEMM swapAB on SM90 by default | 2026-02-19 | kernel-fusion | gemm, kernel-fusion |
| [#34791](../sources/prs/vllm/PR-34791.md) | [Bugfix] Gate 256-bit instructions to CUDA 12.9+ | 2026-02-18 |  | gemm |
| [#34687](../sources/prs/vllm/PR-34687.md) | [Update] Use FlashInfer fast_decode_plan directly instead of replication | 2026-02-17 |  | attention, decode |
| [#34695](../sources/prs/vllm/PR-34695.md) | [Bugfix] Fix MLA attention crash with AWQ/GPTQ quantized models | 2026-02-17 |  | attention, mla, quantization |
| [#34709](../sources/prs/vllm/PR-34709.md) | [ROCm] Enable wvSplitK skinny GEMM kernel for RDNA4/gfx1x decode | 2026-02-17 |  | decode, gemm, quantization |
| [#34718](../sources/prs/vllm/PR-34718.md) | [torch.compile] Turn on silu+fp4 quant fusion by default for O1+ | 2026-02-17 | kernel-fusion | fp4, kernel-fusion |
| [#34725](../sources/prs/vllm/PR-34725.md) | [Bugfix] Fix NVFP4 TRTLLM MoE non-gated support; add gsm8k for Nemotron-3-Nano FP8+NVFP4 | 2026-02-17 |  | fp4, fp8, moe |
| [#34732](../sources/prs/vllm/PR-34732.md) | [Attention] Use FA4 for MLA prefill | 2026-02-17 |  | attention, mla, prefill |
| [#34597](../sources/prs/vllm/PR-34597.md) | [Kernel] Add FP8 KV cache support to Triton MLA decode attention | 2026-02-16 |  | attention, decode, fp8 |
| [#34570](../sources/prs/vllm/PR-34570.md) | [ROCm][AITER] Fix aiter paged_attention_v1 decode for sliding window and head_size < 64 | 2026-02-15 |  | attention, decode |
| [#34577](../sources/prs/vllm/PR-34577.md) | [Bugfix] Rescale NVFP4 weight scales to fix BF16 dequant underflow | 2026-02-15 |  | fp4, nvfp4, quantization |
| [#34552](../sources/prs/vllm/PR-34552.md) | [BugFix] Add support for MTP num_speculative_tokens > 1 with sparse MLA | 2026-02-14 |  | attention, decode, mla |
| [#34556](../sources/prs/vllm/PR-34556.md) | [Quantization] add humming quantization kernel | 2026-02-14 | kernel-fusion | kernel-fusion, moe, quantization |
| [#34476](../sources/prs/vllm/PR-34476.md) | [BUGFIX] Fix accuracy regression for NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 with TP>1 | 2026-02-13 |  | fp4, nvfp4 |
| [#34478](../sources/prs/vllm/PR-34478.md) | [Model] Add NVFP4 quantization support for Step3.5-Flash | 2026-02-13 | kernel-fusion | fp4, kernel-fusion, moe |
| [#34494](../sources/prs/vllm/PR-34494.md) | [Bugfix] Handle num_expert_group=None in flashinfer block-scale FP8 MoE | 2026-02-13 | kernel-fusion | block-scale, fp8, kernel-fusion |
| [#34434](../sources/prs/vllm/PR-34434.md) | [CPU][Perf] Accelerate Attention head for s390x using vector intrinsics | 2026-02-12 |  | attention |
| [#34448](../sources/prs/vllm/PR-34448.md) | [Kernel] Integrate SM100 MXFP8 blockscaled grouped MM and quant kernels | 2026-02-12 |  | fp8, moe |
| [#34471](../sources/prs/vllm/PR-34471.md) | [Llama4,Quantization] Simplify and generalize logic for Q/K permutations in quantized self-attn layers  | 2026-02-12 |  | quantization |
| [#34298](../sources/prs/vllm/PR-34298.md) | [ModelBash][DSR1 NVFp4] Avoid Bf16 Bias Cast | 2026-02-11 |  | fp4, moe, nvfp4 |
| [#34302](../sources/prs/vllm/PR-34302.md) | [ModelBash][DSV3] Add TRTLLM DSV3 Router GEMM kernel (6% B1 Speedup) | 2026-02-11 |  | gemm, moe |
| [#34374](../sources/prs/vllm/PR-34374.md) | [Bugfix] Enforce DeepGEMM when using sparse_attn_indexer on CUDA | 2026-02-11 |  | gemm |
| [#34378](../sources/prs/vllm/PR-34378.md) | Use paged_attention_v1 for sliding window decode in rocm_aiter_fa | 2026-02-11 |  | attention, decode |
| [#34158](../sources/prs/vllm/PR-34158.md) | [Bugfix] Relax TRTLLM KV cache contiguity assertion for cross-layer layout | 2026-02-09 |  | attention |
| [#34187](../sources/prs/vllm/PR-34187.md) | [Bugfix] Fix DP Attention Padding in Dummy Run | 2026-02-09 |  | attention |
| [#34052](../sources/prs/vllm/PR-34052.md) | fix(cpu): fix mla_decode compilation on x86 without AVX512 | 2026-02-07 |  | decode, mla |
| [#33972](../sources/prs/vllm/PR-33972.md) | [Bugfix]fix output Nan/Inf in marlin if dtype=float16 | 2026-02-06 |  | fp4, moe, quantization |
| [#33932](../sources/prs/vllm/PR-33932.md) | [Bugfix] Fix DSV3.2 NVFP4 | 2026-02-05 |  | attention, fp4, mla |
| [#33942](../sources/prs/vllm/PR-33942.md) | Adding support to Sarvam's MoE models | 2026-02-05 |  | moe |
| [#33749](../sources/prs/vllm/PR-33749.md) | [ROCm][AITER] Fix AITER import regression for explicit backend selection | 2026-02-04 |  | attention, decode |
| [#33637](../sources/prs/vllm/PR-33637.md) | [Bugfix] fix DeepSeek R1 with CUTLASS MLA Broken on B200 | 2026-02-03 |  | attention, mla |
| [#33695](../sources/prs/vllm/PR-33695.md) | enable skipping of SW attention layers when using FP8 KV cache | 2026-02-03 |  | attention, fp8, quantization |
| [#33529](../sources/prs/vllm/PR-33529.md) | Triton MLA perf fixes | 2026-02-02 |  | attention, decode, mla |
| [#33540](../sources/prs/vllm/PR-33540.md) | [Feature][Core] Support Fabric detection to adapt the MNNVL protocol for the GB series | 2026-02-02 |  | gemm |
| [#33556](../sources/prs/vllm/PR-33556.md) | [PluggableLayer][3/N] Apply PluggableLayer to moe-related layers. | 2026-02-02 | kernel-fusion | kernel-fusion, moe |
| [#33579](../sources/prs/vllm/PR-33579.md) | [Bugfix] Fix sparse MLA metadata building | 2026-02-02 |  | attention, mla |
| [#33506](../sources/prs/vllm/PR-33506.md) | [Kernel] Support Flashinfer trtllm fused MoE non gated FP8 & NVFP4 | 2026-02-01 | kernel-fusion | fp4, fp8, kernel-fusion |
| [#33517](../sources/prs/vllm/PR-33517.md) | [Kernel] Add enable_sm120_or_later for SM121 (DGX Spark) CUTLASS support | 2026-02-01 |  | fp8, quantization |
| [#33417](../sources/prs/vllm/PR-33417.md) | fix: Add SM120 (RTX Blackwell) support for FlashInfer CUTLASS NVFP4 MoE kernels | 2026-01-30 | kernel-fusion | fp4, kernel-fusion, moe |
| [#33285](../sources/prs/vllm/PR-33285.md) | [Bugfix] Register fp8 cutlass_group_gemm as supported for only SM90+SM100 | 2026-01-28 |  | fp8, gemm |
| [#33291](../sources/prs/vllm/PR-33291.md) | [PERF] Change GDN Attention State Layout from [N, HV, K, V] to [N, HV, V, K] | 2026-01-28 | kernel-fusion | attention, kernel-fusion |
| [#33174](../sources/prs/vllm/PR-33174.md) | Add support for Mistral Large 3 inference with Flashinfer MoE | 2026-01-27 | kernel-fusion | fp8, kernel-fusion, moe |
| [#33177](../sources/prs/vllm/PR-33177.md) | [Attention] Use `has_flashinfer` helper | 2026-01-27 |  | attention, mla |
| [#33192](../sources/prs/vllm/PR-33192.md) | [Bugfix] Disable TRTLLM attention when KV transfer is enabled | 2026-01-27 |  | attention |
| [#33076](../sources/prs/vllm/PR-33076.md) | Support compress-tensors with nvfp4 or fp8 weights and modelopt with nvfp4 weights on Turing | 2026-01-26 |  | fp4, fp8, nvfp4 |
| [#33112](../sources/prs/vllm/PR-33112.md) | Fix IndexError with encoder-decoder models when using Custom Paged Attention | 2026-01-26 |  | attention, decode, prefill |
| [#33122](../sources/prs/vllm/PR-33122.md) | [CPU][Feat] Enable KleidiAI accelerated int4 dynamic quant with BF16 activations on Arm CPUs | 2026-01-26 |  | quantization |
| [#33022](../sources/prs/vllm/PR-33022.md) | [Kernel] Apply 256bit LDG/STG To Activation Kernels | 2026-01-25 |  | gemm |
| [#32993](../sources/prs/vllm/PR-32993.md) | [Feature] Support CPU Offloading without Pytorch Pinned Memory that leads to doubled allocation | 2026-01-24 |  | gemm |
| [#32914](../sources/prs/vllm/PR-32914.md) | [ROCm][perf] Shuffle KV cache to use paged_attention_common | 2026-01-23 |  | attention |
| [#32954](../sources/prs/vllm/PR-32954.md) | [NVIDIA] [feat] Integrate flashinfer Trtllmgen bf16 moe | 2026-01-23 | kernel-fusion | kernel-fusion, moe, quantization |
| [#32974](../sources/prs/vllm/PR-32974.md) | [Attention] FA4 integration | 2026-01-23 |  | attention, mla |
| [#32846](../sources/prs/vllm/PR-32846.md) | [Kernel] use flashinfer for gdn prefill | 2026-01-22 |  | prefill |
| [#32873](../sources/prs/vllm/PR-32873.md) | [Performance] Tune Mamba selective scan kernel for B200 | 2026-01-22 |  | gemm |
| [#32877](../sources/prs/vllm/PR-32877.md) | [Bugfix][Hardware][AMD] Fix ROCM_AITER_FA speculative decoding support | 2026-01-22 |  | attention |
| [#32886](../sources/prs/vllm/PR-32886.md) | [Bugfix] Fix FP8 MoE EP Weight Loading for ModelOpt Llama4 | 2026-01-22 |  | fp8, moe |
| [#32887](../sources/prs/vllm/PR-32887.md) | [Spec Decode] Unified Parallel Drafting | 2026-01-22 |  | attention, decode |
| [#32740](../sources/prs/vllm/PR-32740.md) | [Kernel] [Helion] [1/N] Add Helion ConfigManager | 2026-01-21 |  | gemm |
| [#32795](../sources/prs/vllm/PR-32795.md) | [Bugfix][Attention] Explicitly report support for kv_cache_dtype bfloat16 | 2026-01-21 |  | attention, mla, quantization |
| [#32614](../sources/prs/vllm/PR-32614.md) | fix: Add glm4_moe_lite to MLA detection | 2026-01-19 |  | attention, mla, moe |
| [#32520](../sources/prs/vllm/PR-32520.md) | [Perf][Kernel] Optimize FP4 quantization kernels (SM100F) | 2026-01-17 | kernel-fusion | fp4, kernel-fusion, moe |
| [#32437](../sources/prs/vllm/PR-32437.md) | [Hardware][SM100] Add TRTLLM Kernel for INT4 W4A16 Kernel. | 2026-01-16 | kernel-fusion | kernel-fusion, moe, quantization |
| [#32361](../sources/prs/vllm/PR-32361.md) | [BugFix] Fix DeepSeek-V3.1 + DeepGEMM incompatible scale shapes | 2026-01-15 |  | gemm, quantization |
| [#32385](../sources/prs/vllm/PR-32385.md) | [Model] Molmo2: Enable quantized weight mapping for vision backbone | 2026-01-15 |  | quantization |
| [#32336](../sources/prs/vllm/PR-32336.md) | [Bugfix][ROCm][performance] Resolve the performance regression issue of the Qwen3-Next-80B-A3B-Thinking under rocm_atten | 2026-01-14 |  | attention |
| [#32263](../sources/prs/vllm/PR-32263.md) | [cpu][performance] CPU Paged Attention NEON BFMMLA BF16 Implementation | 2026-01-13 |  | attention, mla |
| [#32008](../sources/prs/vllm/PR-32008.md) | [MISC] Add strict contiguity check for FlashInfer attention tensors | 2026-01-09 |  | attention |
| [#32060](../sources/prs/vllm/PR-32060.md) | [4/N][Attention] Move MLA common to model_executor | 2026-01-09 |  | attention, decode, mla |
| [#32064](../sources/prs/vllm/PR-32064.md) | [5/N][Attention] Finish eliminating `vllm/attention` folder | 2026-01-09 | kernel-fusion, pipeline-stages | attention, fp4, fp8 |
| [#31916](../sources/prs/vllm/PR-31916.md) | [1/N][Attention] Restructure attention: move files | 2026-01-07 | kernel-fusion, pipeline-stages | attention, decode, fp8 |
| [#31777](../sources/prs/vllm/PR-31777.md) | [LoRA]Disable linear LoRA  kernel PDL | 2026-01-06 | kernel-fusion | kernel-fusion, moe |
| [#31828](../sources/prs/vllm/PR-31828.md) | [Perf] Add opt-in SM100 Oink RMSNorm custom-op path | 2026-01-06 |  | gemm |
| [#31837](../sources/prs/vllm/PR-31837.md) | [Perf] Fuse stride preparation for NVFP4 cutlass_moe | 2026-01-06 |  | fp4, moe, nvfp4 |
| [#31720](../sources/prs/vllm/PR-31720.md) | [cpu][bench] Add CPU paged attention benchmarks | 2026-01-05 |  | attention |
| [#31742](../sources/prs/vllm/PR-31742.md) | [Bugfix] Fix Broken ModelOpt NVFP4 MoE | 2026-01-05 | kernel-fusion | fp4, fp8, kernel-fusion |
| [#31635](../sources/prs/vllm/PR-31635.md) | Decouple page_size_bytes calculation in AttentionSpec for TPU/RPA Compatibility. | 2026-01-03 |  | attention |
| [#31523](../sources/prs/vllm/PR-31523.md) | [ROCm][Bugfix] Fix accuracy issue on fmoe when `VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS` enabled | 2025-12-30 | kernel-fusion | kernel-fusion, moe |
| [#31528](../sources/prs/vllm/PR-31528.md) | [FIX] Add NO_MUL activation support for modular kernel path | 2025-12-30 | kernel-fusion | gemm, kernel-fusion, moe |
| [#31531](../sources/prs/vllm/PR-31531.md) | Use the same memory for workspace13 and fused_output. | 2025-12-30 | kernel-fusion | kernel-fusion, moe |
| [#31534](../sources/prs/vllm/PR-31534.md) | [Fix] Align fused moe lora_b shape with peft | 2025-12-30 | kernel-fusion | kernel-fusion, moe |
| [#31465](../sources/prs/vllm/PR-31465.md) | fixed mypy warnings for files vllm/v1/attention with TEMPORARY workaround | 2025-12-29 |  | attention, mla |
| [#31502](../sources/prs/vllm/PR-31502.md) | [Bugfix][ROCm] Fix Static Quant Issue | 2025-12-29 | kernel-fusion | fp8, kernel-fusion, moe |
| [#31453](../sources/prs/vllm/PR-31453.md) | [BugFix]  add select_gemm_impl on CompressedTensorsWNA16MoEMethod to support LoRA | 2025-12-28 |  | gemm, moe, quantization |
| [#31380](../sources/prs/vllm/PR-31380.md) | [Bugfix][ROCm]Fix Qwen3-Next-80B-A3B-Thinking inference and optimize non-standard block size (544) support under rocm_atten | 2025-12-26 |  | attention, decode, prefill |
| [#31339](../sources/prs/vllm/PR-31339.md) | [Misc] Fix Qwen2-MoE shared_expert_gate | 2025-12-25 |  | moe |
| [#31282](../sources/prs/vllm/PR-31282.md) | [Bugfix][Hardware][AMD] Fix last_page_len calculation in AITER MLA decode | 2025-12-24 |  | attention, decode, mla |
| [#31286](../sources/prs/vllm/PR-31286.md) | fix(rocm): add early return in get_flash_attn_version for ROCm | 2025-12-24 |  | attention |
| [#31317](../sources/prs/vllm/PR-31317.md) | pin lora_b moe weights on cpu | 2025-12-24 | kernel-fusion | kernel-fusion, moe |
| [#31195](../sources/prs/vllm/PR-31195.md) | [SM100] Resubmit FMHA FP8 prefill for MLA | 2025-12-23 |  | attention, flash-attention, fp8 |
| [#31246](../sources/prs/vllm/PR-31246.md) | [Kernel] Add topk_sigmoid kernel | 2025-12-23 | kernel-fusion | kernel-fusion, moe, tma |
| [#31104](../sources/prs/vllm/PR-31104.md) | [BugFix] LoRA: Support loading base_layer of experts | 2025-12-22 | kernel-fusion | kernel-fusion, moe |
| [#31106](../sources/prs/vllm/PR-31106.md) | [Bugfix][Hardware][AMD] Consolidate FP8 min/max values helper function | 2025-12-22 |  | fp8, gemm, quantization |
| [#31115](../sources/prs/vllm/PR-31115.md) | [Misc] Fix grammar errors in comments and messages | 2025-12-22 |  | attention, quantization |
| [#31161](../sources/prs/vllm/PR-31161.md) | [Bugfix] Fix MoE LoRA bin/pt loading | 2025-12-22 |  | moe |
| [#31171](../sources/prs/vllm/PR-31171.md) | [perf] Integrate flashinfer concat_mla_k | 2025-12-22 |  | attention, mla |
| [#31177](../sources/prs/vllm/PR-31177.md) | [Bugfix][Hardware][AMD] Fix exception types in AITER MLA FP8 check | 2025-12-22 |  | fp8, mla |
| [#31099](../sources/prs/vllm/PR-31099.md) |  [FIX] Always support TP > 4 for FP4 Gemm | 2025-12-21 |  | fp4, gemm, nvfp4 |
| [#31055](../sources/prs/vllm/PR-31055.md) | [Bugfix] Fix GLM-4 MoE router logits dtype for data parallel chunking | 2025-12-20 | kernel-fusion | kernel-fusion, moe |
| [#31003](../sources/prs/vllm/PR-31003.md) | [Mics] add pcp basic support to MoE model | 2025-12-19 | kernel-fusion | kernel-fusion, moe |
| [#30957](../sources/prs/vllm/PR-30957.md) | [Feature]: Support NVIDIA ModelOpt HF FP8 variants FP8_PER_CHANNEL_PER_TOKEN and FP8_PB_WO  in vLLM | 2025-12-18 |  | fp8, quantization |
| [#30974](../sources/prs/vllm/PR-30974.md) | [Bugfix] Fix incorrect tiles creation for mm prefix triton attention | 2025-12-18 |  | attention |
| [#30976](../sources/prs/vllm/PR-30976.md) | Use aiter triton fused_add_rmsnorm_pad for gpt-oss | 2025-12-18 | kernel-fusion | kernel-fusion |
| [#30842](../sources/prs/vllm/PR-30842.md) | [Kernels][FI] Skip trtllm attention when num_kv_heads=1 | 2025-12-17 |  | attention |
| [#30881](../sources/prs/vllm/PR-30881.md) | [Compressed-Tensors] Simplify NVFP4 Conditions, enable marlin support for NVFP4A16 MoEs | 2025-12-17 |  | fp4, moe, nvfp4 |
| [#30885](../sources/prs/vllm/PR-30885.md) | [Kernel][Performance] Enable smaller Scaling Factor tiling for NVFP4 small-batch decoding | 2025-12-17 | pipeline-stages | fp4, nvfp4, pipeline-stages |
| [#30887](../sources/prs/vllm/PR-30887.md) | [Bugfix] [Kernel] Triton attention kernels: mask out V blocks that fall outside sliding window | 2025-12-17 |  | attention |
| [#30897](../sources/prs/vllm/PR-30897.md) | [NVFP4][Perf] Tune NVFP4 input quant kernel for small batch size | 2025-12-17 | kernel-fusion | fp4, kernel-fusion, nvfp4 |
| [#30729](../sources/prs/vllm/PR-30729.md) | [Perf] enable flashinfer rotary_embedding custom ops in DeepSeek rotary | 2025-12-16 |  | gemm |
| [#30731](../sources/prs/vllm/PR-30731.md) | [Bugfix] Fix broken ViT attention selection for Blackwell device | 2025-12-16 |  | attention |
| [#30746](../sources/prs/vllm/PR-30746.md) | [SM100] Enable fp8 compute for prefill MLA | 2025-12-16 |  | attention, fp4, fp8 |
| [#30802](../sources/prs/vllm/PR-30802.md) | Add support for LoRA adapters in Nemotron-H models | 2025-12-16 | kernel-fusion | kernel-fusion, moe |
| [#30687](../sources/prs/vllm/PR-30687.md) | Triton Attention: Support cross-layers blocks | 2025-12-15 |  | attention |
| [#30692](../sources/prs/vllm/PR-30692.md) | OffloadingConnector: Support kernel_block_size != block_size | 2025-12-15 |  | attention |
| [#30709](../sources/prs/vllm/PR-30709.md) | [Misc][LLaMa4] Compile LLaMa Vision Encoder | 2025-12-15 |  | attention |
| [#30711](../sources/prs/vllm/PR-30711.md) | Update note comment for flashinfer attention warmup | 2025-12-15 |  | attention |
| [#30716](../sources/prs/vllm/PR-30716.md) | fused_moe_lora PDL improvements | 2025-12-15 | kernel-fusion | kernel-fusion, moe |
| [#30647](../sources/prs/vllm/PR-30647.md) | [Perf] Eliminate padding and slicing op for GPT-OSS with Flashinfer MXFP4 MXFP8 MoE | 2025-12-14 | kernel-fusion | fp4, fp8, kernel-fusion |
| [#30585](../sources/prs/vllm/PR-30585.md) | [Bugfix] Fix Triton FusedMoE LoRA | 2025-12-13 | kernel-fusion | kernel-fusion, moe |
| [#30528](../sources/prs/vllm/PR-30528.md) | [Perf] Set split_k to 1 for triton_kernels | 2025-12-12 |  | fp4, quantization |
| [#30575](../sources/prs/vllm/PR-30575.md) | [Bugfix] Pass FA version in `MultiHeadAttention` | 2025-12-12 |  | attention |
| [#30484](../sources/prs/vllm/PR-30484.md) | [Feature] Add SM103 (Blackwell Ultra) Support to vLLM | 2025-12-11 | kernel-fusion | attention, decode, fp4 |
| [#30399](../sources/prs/vllm/PR-30399.md) | [BugFix] Fix `AttributeError: 'MergedColumnParallelLinear' object has no attribute 'weight_scale'` | 2025-12-10 |  | gemm |
| [#30408](../sources/prs/vllm/PR-30408.md) | fix(gguf): Disable bfloat16 for GGUF on blackwell device | 2025-12-10 |  | quantization |
| [#30430](../sources/prs/vllm/PR-30430.md) | [ROCm][Bugfix] Add MLACommonMetadata to allowed attention types for speculative decoding | 2025-12-10 |  | attention, decode, mla |
| [#30286](../sources/prs/vllm/PR-30286.md) | [LoRA] Support Quantized Adapters | 2025-12-09 | kernel-fusion | fp8, kernel-fusion, moe |
| [#30307](../sources/prs/vllm/PR-30307.md) | [Model][Quantization] Fix / Add GGUF support for Qwen2 MoE models | 2025-12-09 |  | moe, quantization |
| [#30308](../sources/prs/vllm/PR-30308.md) | [bugfix][quantization] fix quark qwen3 kv_cache quantization | 2025-12-09 |  | moe, quantization |
| [#30314](../sources/prs/vllm/PR-30314.md) | [fix] fix SM check for Flashinfer TRTLLM MOE | 2025-12-09 |  | moe, quantization |
| [#30329](../sources/prs/vllm/PR-30329.md) | [Feature][CPU Backend]: Optimize ARM vectorization backend | 2025-12-09 |  | decode, mla |
| [#30336](../sources/prs/vllm/PR-30336.md) | [Bugfix] Fix fp8 DeepGemm compilation issues | 2025-12-09 |  | fp8, gemm, quantization |
| [#30357](../sources/prs/vllm/PR-30357.md) | [ROCm][Quantization] GPT OSS Upstream MoE wmxfp4_afp8 with static scales | 2025-12-09 | kernel-fusion | fp4, fp8, kernel-fusion |
| [#30243](../sources/prs/vllm/PR-30243.md) | [LoRA]  Reduce the loading time of MoE LoRA | 2025-12-08 |  | moe |
| [#30254](../sources/prs/vllm/PR-30254.md) | gptq marlin quantization support for fused moe with lora | 2025-12-08 | kernel-fusion | kernel-fusion, moe, quantization |
| [#30267](../sources/prs/vllm/PR-30267.md) | [Bugfix] Fix DeepGEMM after #29546  | 2025-12-08 |  | fp8, gemm, quantization |
| [#30203](../sources/prs/vllm/PR-30203.md) | Add latent MoE support | 2025-12-07 |  | moe |
| [#30210](../sources/prs/vllm/PR-30210.md) | [Bugfix]: Fix glm46 awq marlin moe wna16 compatibility | 2025-12-07 | kernel-fusion | kernel-fusion, moe, quantization |
| [#30164](../sources/prs/vllm/PR-30164.md) | Nvidia ModelOpt workaround for issue 28072 | 2025-12-06 |  | quantization |
| [#30116](../sources/prs/vllm/PR-30116.md) | [Model][Quantization] Restore MoE + GGUF models support (incl. Qwen3 MoE) by allowing Sideload Parameters | 2025-12-05 |  | moe, quantization |
| [#30118](../sources/prs/vllm/PR-30118.md) | [Model][Quantization] Override HF defaults to GGUF ones (incl. Qwen3 MoE) | 2025-12-05 |  | moe, quantization |
| [#30141](../sources/prs/vllm/PR-30141.md) | Add llmcompressor fp8 kv-cache quant (per-tensor and per-attn_head) | 2025-12-05 |  | attention, fp8, quantization |
| [#30014](../sources/prs/vllm/PR-30014.md) | [Perf] Do FP4 quant before All gather on flashinfer trtllmgen MOE  | 2025-12-04 | kernel-fusion | fp4, kernel-fusion, moe |
| [#30068](../sources/prs/vllm/PR-30068.md) | [CPU][Perf] Add fast vectorized exp impl from Arm Optimized Routines | 2025-12-04 |  | gemm |
| [#30071](../sources/prs/vllm/PR-30071.md) | [Quantization] Support Quark int4-fp8 w4a8 for MoE | 2025-12-04 |  | fp8, moe, quantization |
| [#29933](../sources/prs/vllm/PR-29933.md) | [BugFix] Fix DBO assert `assert B_block_table == B_q` | 2025-12-03 |  | attention, decode |
| [#29935](../sources/prs/vllm/PR-29935.md) | [moe] Use enable_chunking func (to support disabling chunking) | 2025-12-03 | kernel-fusion | kernel-fusion, moe |
| [#29936](../sources/prs/vllm/PR-29936.md) | [moe] Allow disabling DP chunking | 2025-12-03 | kernel-fusion | kernel-fusion, moe |
| [#29960](../sources/prs/vllm/PR-29960.md) | [Bugfix] Fix flashinfer ar+norm kernel not available issue | 2025-12-03 |  | gemm |
| [#30005](../sources/prs/vllm/PR-30005.md) | [ROCm] add fallback for aiter fp8 decode mla | 2025-12-03 |  | decode, fp8, mla |
| [#29845](../sources/prs/vllm/PR-29845.md) | [SpecDecode] Simplified alternative padded-speculation acceptance rate fix | 2025-12-02 |  | attention, decode, mla |
| [#29867](../sources/prs/vllm/PR-29867.md) | [Quantization] fix: overflow with static per-tensor scaling | 2025-12-02 |  | attention, mla, quantization |
| [#29887](../sources/prs/vllm/PR-29887.md) | [ROCm][Perf] Enable shuffle kv cache layout and assembly paged attention kernel for `AiterFlashAttentionBackend` | 2025-12-02 |  | attention |
| [#29890](../sources/prs/vllm/PR-29890.md) | [Bugfix] Fix FP8 MoE LoRA | 2025-12-02 |  | fp8, moe, quantization |
| [#29901](../sources/prs/vllm/PR-29901.md) | [Kernel][Quantization][MoE] add marlin kernel support for turing (sm75) | 2025-12-02 |  | fp8, moe, quantization |
| [#29773](../sources/prs/vllm/PR-29773.md) | [ROCm] [Fused Moe EP] Use binary expert mask for aiter fused moe kernel | 2025-12-01 | kernel-fusion | kernel-fusion, moe, quantization |
| [#29775](../sources/prs/vllm/PR-29775.md) | [ROCm][MXFP4] Infer w4a4 quant method in rocm aiter fused moe | 2025-12-01 | kernel-fusion | fp4, kernel-fusion, moe |
| [#29795](../sources/prs/vllm/PR-29795.md) | [Perf] Improve fp8 quant in mla; replace ReduceSum with ReduceScatterSum | 2025-12-01 |  | attention, fp8, mla |
| [#29804](../sources/prs/vllm/PR-29804.md) | [EPLB] Support EPLB w/ NVFP4 | 2025-12-01 | kernel-fusion | fp4, kernel-fusion, moe |
| [#29816](../sources/prs/vllm/PR-29816.md) | [Bugfix][Model] Support LoRA on Qwen3 Output Embedding | 2025-12-01 |  | moe |
| [#29742](../sources/prs/vllm/PR-29742.md) | [Bugfix] Fix mismatched nvfp4 gemm output shape | 2025-11-30 |  | fp4, gemm, nvfp4 |
| [#29748](../sources/prs/vllm/PR-29748.md) | [MoE-FP8-modelopt] Add FlashInfer alignment padding for intermediate dimensions | 2025-11-30 |  | fp8, moe, quantization |
| [#29757](../sources/prs/vllm/PR-29757.md) | Add Mistral Large 3 and Ministral 3 | 2025-11-30 | kernel-fusion | decode, fp8, kernel-fusion |
| [#29710](../sources/prs/vllm/PR-29710.md) | [perf] Use direct copy (broadcast) instead of cat for k_nope/k_pe in MLA prefill | 2025-11-29 |  | attention, mla, prefill |
| [#29732](../sources/prs/vllm/PR-29732.md) | [Quantization] Enable compressed-tensors AWQ for Turing GPU | 2025-11-29 |  | quantization |
| [#29631](../sources/prs/vllm/PR-29631.md) | [Bugfix] Defunctionalize TRTLLM AR+Norm op for avoiding extra clone kernel before it | 2025-11-28 |  | gemm |
| [#29642](../sources/prs/vllm/PR-29642.md) | [Kernel][MoE] optimize `moe_align_block_size` | 2025-11-28 | kernel-fusion | kernel-fusion, moe |
| [#29644](../sources/prs/vllm/PR-29644.md) | [Attention] Make `split_decodes_and_prefills(..., require_uniform=True)` support padding | 2025-11-28 |  | attention, decode, prefill |
| [#29691](../sources/prs/vllm/PR-29691.md) | [Kernel]Support W4A8 Grouped GEMM on Hopper | 2025-11-28 | kernel-fusion | fp8, gemm, kernel-fusion |
| [#29627](../sources/prs/vllm/PR-29627.md) | [Attention] Cache attention metadata builds across hybrid KV-cache groups | 2025-11-27 |  | attention |
| [#29439](../sources/prs/vllm/PR-29439.md) | [Bugfix] Fix grouped_topk pytorch impl when num_experts can't be grouped properly | 2025-11-25 | kernel-fusion | kernel-fusion, moe |
| [#29339](../sources/prs/vllm/PR-29339.md) | [Bugfix] Only use triton_kernels for MXFP4 on SM90 and SM100 | 2025-11-24 |  | fp4, quantization |
| [#29346](../sources/prs/vllm/PR-29346.md) | [Perf] Disable DeepGEMM MoE by default when TP=8 is used | 2025-11-24 |  | fp8, gemm, moe |
| [#29354](../sources/prs/vllm/PR-29354.md) | Add unpermute-aware fused MoE path and small-batch fallback | 2025-11-24 | kernel-fusion | kernel-fusion, moe |
| [#29257](../sources/prs/vllm/PR-29257.md) | Lora MoE Align Improvements | 2025-11-23 |  | moe |
| [#29213](../sources/prs/vllm/PR-29213.md) | [Perf][Kernels] Enable FlashInfer DeepGEMM swapAB on SM90 (for W8A8 Linear Op) | 2025-11-22 |  | fp8, gemm, quantization |
| [#29222](../sources/prs/vllm/PR-29222.md) | [LoRA] Optimize 3D MoE logic | 2025-11-22 | kernel-fusion | kernel-fusion, moe |
| [#29240](../sources/prs/vllm/PR-29240.md) | chore: add RTX_PRO_6000 GLM4.6-FP8 kernel tuning | 2025-11-22 | kernel-fusion | fp8, kernel-fusion, moe |
| [#29242](../sources/prs/vllm/PR-29242.md) | [Kernel] Add NVFP4 MoE CUTLASS support for SM120 | 2025-11-22 |  | fp4, moe, nvfp4 |
| [#29125](../sources/prs/vllm/PR-29125.md) | [Feature] Batch invariant: Enable `TRITON_MLA` without prefix-caching | 2025-11-20 |  | attention, mla |
| [#28990](../sources/prs/vllm/PR-28990.md) | [BugFix] Fix async-scheduling + FlashAttn MLA | 2025-11-19 |  | attention, mla |
| [#29004](../sources/prs/vllm/PR-29004.md) | [Feat] Support non-gated activations in NVFP4 modelopt path | 2025-11-19 | kernel-fusion | fp4, kernel-fusion, moe |
| [#29039](../sources/prs/vllm/PR-29039.md) | [DeepSeek + LMCache Multiprocess] handle MLA for deepseek model + LMCache Multiprocess connector | 2025-11-19 |  | mla |
| [#28892](../sources/prs/vllm/PR-28892.md) | Add TRTLLM MoE NVFP4 kernel to CompressedTensorsW4A4MoeMethod | 2025-11-18 |  | fp4, moe, nvfp4 |
| [#28938](../sources/prs/vllm/PR-28938.md) | [NVIDIA] Guard SM100 CUTLASS MoE macro to SM100 builds v2 | 2025-11-18 |  | moe |
| [#28840](../sources/prs/vllm/PR-28840.md) | bugfix: correct attn output with base 2 or e | 2025-11-17 |  | attention, mla |
| [#28841](../sources/prs/vllm/PR-28841.md) | [Bugfix] Fix GPT-OSS AR+NORM fusion | 2025-11-17 | kernel-fusion, pipeline-stages | kernel-fusion, moe, pipeline-stages |
| [#28878](../sources/prs/vllm/PR-28878.md) | [Bugfix] Make compressed-tensors MoEs respect ignored layers | 2025-11-17 | kernel-fusion, pipeline-stages | kernel-fusion, moe, pipeline-stages |
| [#28816](../sources/prs/vllm/PR-28816.md) | [Bugfix] Fix GPT-OSS on AMD after #28603 | 2025-11-16 |  | fp4, quantization |
| [#28775](../sources/prs/vllm/PR-28775.md) | [Model] Add support for openPangu moe model | 2025-11-15 |  | attention, moe |
| [#28718](../sources/prs/vllm/PR-28718.md) | [Feature] Prefill Context Parallel (PCP) basic support | 2025-11-14 | kernel-fusion | attention, kernel-fusion, mla |
| [#28660](../sources/prs/vllm/PR-28660.md) | [Attention][Bugfix] Fix FA sink support | 2025-11-13 |  | attention |
| [#28677](../sources/prs/vllm/PR-28677.md) | [Bugfix][Nixl] Fix kernel physical<>logical block_size issue  | 2025-11-13 |  | gemm |
| [#28687](../sources/prs/vllm/PR-28687.md) | [Performance] Reduce DeepGEMM N dim restriction from 128 to 64 multiplier  | 2025-11-13 | pipeline-stages | fp8, gemm, pipeline-stages |
| [#28561](../sources/prs/vllm/PR-28561.md) | [Bugfix] Fix SM100 gpt-oss regression due to faulty attn sink support | 2025-11-12 |  | attention |
| [#28376](../sources/prs/vllm/PR-28376.md) | [ROCm] Support for Whisper v1 with Aiter Unified Attention and Aiter Flash Attention | 2025-11-10 |  | attention |
| [#28377](../sources/prs/vllm/PR-28377.md) | [Bugfix][EPLB] Disabled shared expert overlap when EPLB is enabled | 2025-11-10 | kernel-fusion | kernel-fusion, moe |
| [#28358](../sources/prs/vllm/PR-28358.md) | [Performance][B200] silu_mul_quant: pack scales in int32 | 2025-11-09 | kernel-fusion | fp8, gemm, kernel-fusion |
| [#28346](../sources/prs/vllm/PR-28346.md) | fix cross attention | 2025-11-08 |  | attention |
| [#28284](../sources/prs/vllm/PR-28284.md) | [Feature] Support recording expert indices for rollout router replay | 2025-11-07 | kernel-fusion | kernel-fusion, moe |
| [#28101](../sources/prs/vllm/PR-28101.md) | [Model] Consolidate Deepseek-MoE implementation with DeepSeek-v2 | 2025-11-05 |  | moe |
| [#28124](../sources/prs/vllm/PR-28124.md) | [Perf][DeepSeek] Add sigmoid+bias fusion to fused_grouped_topk from TRTLLM | 2025-11-05 | kernel-fusion | kernel-fusion, moe |
| [#28133](../sources/prs/vllm/PR-28133.md) | [Mamba] - Consolidate Mambas Attention Logic | 2025-11-05 |  | attention |
| [#28166](../sources/prs/vllm/PR-28166.md) | [flashinfer] fix FI all2all with FI cutlass moe | 2025-11-05 | kernel-fusion | kernel-fusion, moe |
| [#28032](../sources/prs/vllm/PR-28032.md) | [ROCm][MLA] enable fp8 MLA decode on ROCm | 2025-11-04 |  | attention, decode, fp8 |
| [#27952](../sources/prs/vllm/PR-27952.md) | Update Flashinfer from `v0.4.1` to `v0.5.2` | 2025-11-03 |  | attention |
| [#27990](../sources/prs/vllm/PR-27990.md) | [flashinfer][fix] do not check nvcc availability when using pre-downloaded cubins | 2025-11-03 |  | gemm |
| [#27994](../sources/prs/vllm/PR-27994.md) | [FlashInfer] Avoid FlashInfer block_size 16 + head_size 256 on blackwell | 2025-11-03 |  | attention |
| [#27931](../sources/prs/vllm/PR-27931.md) | [Kernel] Optimize rms_norm kernel | 2025-11-01 |  | gemm |
| [#27856](../sources/prs/vllm/PR-27856.md) | [Feature] Extend batch invariant torch.compile to B200 | 2025-10-31 |  | gemm |
| [#27883](../sources/prs/vllm/PR-27883.md) | [Performance] Fused blockwise quant RMS norm | 2025-10-31 | kernel-fusion | fp8, gemm, kernel-fusion |
| [#27884](../sources/prs/vllm/PR-27884.md) | [Bug] Batch invariant: Fix flash attn MLA `RuntimeError: scheduler_metadata must have shape (metadata_size)` | 2025-10-31 |  | attention, mla |
| [#27897](../sources/prs/vllm/PR-27897.md) | [Performance][B200] Fix deepgemm prologue | 2025-10-31 | kernel-fusion | fp8, gemm, kernel-fusion |
| [#27715](../sources/prs/vllm/PR-27715.md) | [AMD] Use Decoupled Kernel Block Size to Support AITER MLA block_size=1 | 2025-10-29 |  | attention, mla |
| [#27660](../sources/prs/vllm/PR-27660.md) | [Feature] Batch invariant torch.compile | 2025-10-28 |  | fp8, quantization |
| [#27532](../sources/prs/vllm/PR-27532.md) | [Attention] Use sparse prefill kernel for fp8 kv-cache in DeepSeek-v3.2 | 2025-10-26 | kernel-fusion | attention, fp4, fp8 |
| [#27492](../sources/prs/vllm/PR-27492.md) | [Performance] Support FP8 flashinfer TRTLLM MOE on Qwen3 and Qwen-3next | 2025-10-24 | kernel-fusion | fp8, kernel-fusion, moe |
| [#27363](../sources/prs/vllm/PR-27363.md) | Prefer FlashAttention MLA as default over FlashMLA | 2025-10-22 |  | attention, mla |
| [#27367](../sources/prs/vllm/PR-27367.md) | [Misc] Make reorder batch also separate extends | 2025-10-22 |  | attention |
| [#27232](../sources/prs/vllm/PR-27232.md) | [Bugfix] Ensure calculated KV scales are applied in attention. | 2025-10-21 | pipeline-stages | attention, pipeline-stages |
| [#27255](../sources/prs/vllm/PR-27255.md) | Bugfix: Cutlass FP8 FusedMoE bad scaling factors | 2025-10-21 | kernel-fusion | fp8, kernel-fusion, moe |
| [#27261](../sources/prs/vllm/PR-27261.md) | Feature: Support Relu2 in FusedMoE fp8 cutlass path | 2025-10-21 | kernel-fusion | fp8, kernel-fusion, moe |
| [#27284](../sources/prs/vllm/PR-27284.md) | [Perf] SM100 - add swap AB optimization to CUTLASS FP8 GEMM | 2025-10-21 |  | fp8, gemm, quantization |
| [#27187](../sources/prs/vllm/PR-27187.md) | [ROCM] Enable CompressedTensorsWNA16 | 2025-10-20 |  | moe, quantization |
| [#27190](../sources/prs/vllm/PR-27190.md) | [BUGFIX][ROCM] ViT FlashAttention on ROCm (no GFX9) and contiguous on qwen3vl ROCm TORCH_SDPA | 2025-10-20 |  | attention |
| [#27223](../sources/prs/vllm/PR-27223.md) | Flashinfer_CUTLASS_MOE fuses quantization for TP | 2025-10-20 | kernel-fusion | fp4, kernel-fusion, moe |
| [#27229](../sources/prs/vllm/PR-27229.md) | [Feature] Batch Invariant for R1 TP 8 on Blackwell | 2025-10-20 |  | gemm |
| [#27146](../sources/prs/vllm/PR-27146.md) | [torch.compile] Enable silu_mul_fp8_quant fusion without custom ops enabled | 2025-10-18 | kernel-fusion | fp8, kernel-fusion |
| [#27127](../sources/prs/vllm/PR-27127.md) | [Feature] Batch Invariant: Support DeepGEMM and Blackwell | 2025-10-17 |  | fp8, gemm, quantization |
| [#26859](../sources/prs/vllm/PR-26859.md) | Disable FlashInfer sampler by default | 2025-10-15 |  | gemm |
| [#26846](../sources/prs/vllm/PR-26846.md) | [Attention] Tune CUTLASS MLA num_splits | 2025-10-14 |  | attention, mla |
| [#26669](../sources/prs/vllm/PR-26669.md) | support flashinfer_fp4 moe for 5090 gpu | 2025-10-13 |  | fp4, moe, quantization |
| [#26714](../sources/prs/vllm/PR-26714.md) | [NVIDIA] [Perf] Update to leverage flashinfer trtllm FP4 MOE throughput kernel | 2025-10-13 | kernel-fusion | fp4, kernel-fusion, moe |
| [#26729](../sources/prs/vllm/PR-26729.md) | [Bugfix] Fix gpt-oss w4a8 DP/EP on B200 | 2025-10-13 | kernel-fusion | fp4, fp8, kernel-fusion |
| [#26534](../sources/prs/vllm/PR-26534.md) | Move query quantization to attention layer for Flashinfer & Triton. | 2025-10-09 | kernel-fusion | attention, kernel-fusion, quantization |
| [#26535](../sources/prs/vllm/PR-26535.md) | [Bugfix] Convert untraceable GroupShape to list for AMD impl | 2025-10-09 |  | fp8, quantization |
| [#26545](../sources/prs/vllm/PR-26545.md) | [ROCM] MoE fp4 CK kernel | 2025-10-09 | kernel-fusion | fp4, kernel-fusion, moe |
| [#26440](../sources/prs/vllm/PR-26440.md) | [Performance] Dual stream execution of "shared_experts" and "selected_experts" inside FusedMoE | 2025-10-08 | kernel-fusion | kernel-fusion, moe |
| [#26322](../sources/prs/vllm/PR-26322.md) | [Bug] Fix Shape Validation for Fallback while Enabling E8M0 for DeepGEMM | 2025-10-06 | kernel-fusion | gemm, kernel-fusion, moe |
| [#26098](../sources/prs/vllm/PR-26098.md) | Fix undefined symbol: cutlass_moe_mm_sm100 | 2025-10-02 |  | moe, quantization |
| [#26107](../sources/prs/vllm/PR-26107.md) | [NVIDIA] Add support for cudnn fp4 gemm via flashinfer | 2025-10-02 |  | fp4, gemm, nvfp4 |
| [#26135](../sources/prs/vllm/PR-26135.md) | [ModelOpt] Load w13/w2_input_scale for all experts, nvfp4 | 2025-10-02 | kernel-fusion | fp4, kernel-fusion, moe |
| [#25935](../sources/prs/vllm/PR-25935.md) | Fix INT8 quantization error on Blackwell GPUs (SM100+) | 2025-09-30 |  | quantization |
| [#25947](../sources/prs/vllm/PR-25947.md) | [Bugfix] Enable padded FP4 quantization | 2025-09-30 |  | fp4, quantization |
| [#25954](../sources/prs/vllm/PR-25954.md) | [Performance] Split FlashAttn attention and cache update | 2025-09-30 |  | attention, decode |
| [#25968](../sources/prs/vllm/PR-25968.md) | [Quantization/NVFP4] Speed up TRTLLM NVFP4 MOE weight loading and fix K/V scale loading for MLA Attn | 2025-09-30 |  | fp4, mla, moe |
| [#25984](../sources/prs/vllm/PR-25984.md) | [Spec Decode] Enable efficient speculative decoding with FlashInfer-MLA | 2025-09-30 |  | attention, decode, mla |
| [#25987](../sources/prs/vllm/PR-25987.md) | [Bugfix] Allow skipping MoE in NVFP4 (fix for MTP) | 2025-09-30 | kernel-fusion | fp4, kernel-fusion, moe |
| [#25990](../sources/prs/vllm/PR-25990.md) | [MoE] Nvfp4 Masked Gemm: Add flashinfer grouped_gemm_nt_masked | 2025-09-30 | kernel-fusion, pipeline-stages | fp4, gemm, grouped-gemm |
| [#25843](../sources/prs/vllm/PR-25843.md) | Update launch_bounds_utils.h for correct compile on Multiple Cuda Arch - PTXAS out of range Warning | 2025-09-28 |  | gemm |
| [#25774](../sources/prs/vllm/PR-25774.md) | Fuse RoPE and MLA KV-cache write | 2025-09-26 | kernel-fusion | kernel-fusion, mla |
| [#25674](../sources/prs/vllm/PR-25674.md) | [Flashinfer][gpt-oss] Support FP8-qkv Flashinfer TRTLLM Sinks Attention | 2025-09-25 |  | attention, fp8 |
| [#25609](../sources/prs/vllm/PR-25609.md) | Enable Fbgemm NVFP4 on Dense models | 2025-09-24 |  | fp4, gemm, nvfp4 |
| [#25478](../sources/prs/vllm/PR-25478.md) | [BugFix] Fix MLA assert with CUTLASS MLA | 2025-09-23 |  | attention, mla |
| [#25503](../sources/prs/vllm/PR-25503.md) | feat: BF16 FlashInfer Fused Cutlass MOE for Hopper and Blackwell Expert Parallel | 2025-09-23 | kernel-fusion | kernel-fusion, moe |
| [#25509](../sources/prs/vllm/PR-25509.md) | [Bugfix] [B200] cutlass_mla - ensure kv_split == 1 for batch size > 1 | 2025-09-23 |  | attention, mla |
| [#25193](../sources/prs/vllm/PR-25193.md) | [Compile] Fix Compile Warning for Ignoring `MIN_BLOCK_PER_SM` | 2025-09-18 | kernel-fusion | fp4, kernel-fusion, nvfp4 |
| [#25201](../sources/prs/vllm/PR-25201.md) | [ROCm] Small functional changes for gptoss | 2025-09-18 |  | fp4, quantization |
| [#25049](../sources/prs/vllm/PR-25049.md) | [Attention][DCP] Support DCP with query length > 1 (MTP) with FA3 | 2025-09-17 |  | attention, decode, mla |
| [#25106](../sources/prs/vllm/PR-25106.md) | [Bug] Fix `returned_lse` not Defined issue | 2025-09-17 |  | attention, mla |
| [#25107](../sources/prs/vllm/PR-25107.md) | Disable failing GPT-OSS Eval (Blackwell) for now | 2025-09-17 | pipeline-stages | pipeline-stages |
| [#24966](../sources/prs/vllm/PR-24966.md) | [Bugfix][B200] Fix `cutlass_mla` hang | 2025-09-16 |  | attention, mla |
| [#24864](../sources/prs/vllm/PR-24864.md) | [DCP] Support Decode Context Parallel (DCP) for GQA with FlashAttention | 2025-09-15 |  | attention, decode |
| [#24833](../sources/prs/vllm/PR-24833.md) | [Bugfix] Fix accuracy issue for silu_mul + nvfp4 quant fusion kernel | 2025-09-14 | kernel-fusion, pipeline-stages | fp4, kernel-fusion, nvfp4 |
| [#24722](../sources/prs/vllm/PR-24722.md) | [Kernel][Quantization] add w4a8 support for marlin kernel | 2025-09-12 | kernel-fusion | fp4, fp8, gemm |
| [#24727](../sources/prs/vllm/PR-24727.md) | [Model] Support Qwen3-VL Model Series | 2025-09-12 |  | moe |
| [#24774](../sources/prs/vllm/PR-24774.md) | [Bug] Fix `is_flashmla_supported` Check Error | 2025-09-12 |  | attention, mla |
| [#23696](../sources/prs/vllm/PR-23696.md) | [Kernel][tcgen05] nvfp4 fused tcgen05 moe | 2025-09-11 | kernel-fusion, fine-grained-quantization | tcgen05, nvfp4, moe |
| [#24666](../sources/prs/vllm/PR-24666.md) | [Performance] Move apply_w8a8_block_fp8_linear to an op class | 2025-09-11 |  | fp8, gemm, quantization |
| [#24673](../sources/prs/vllm/PR-24673.md) | [NVIDIA] Blackwell Family | 2025-09-11 |  | fp8, quantization |
| [#24521](../sources/prs/vllm/PR-24521.md) | [Feature] Disallow FlashMLA on Blackwell | 2025-09-09 |  | attention, mla |
| [#24440](../sources/prs/vllm/PR-24440.md) | [Transform] [Quantization] Add QuTLASS support to vLLM | 2025-09-08 | pipeline-stages | fp4, nvfp4, pipeline-stages |
| [#24385](../sources/prs/vllm/PR-24385.md) | [Kernel] Support decode context parallelism on Blackwell with CUTLASS MLA | 2025-09-07 |  | attention, decode, mla |
| [#24248](../sources/prs/vllm/PR-24248.md) | [PERF] Allreduce fusion. Support torch native matching. Tuning of the thresholds | 2025-09-04 | kernel-fusion, pipeline-stages | kernel-fusion, moe, pipeline-stages |
| [#23978](../sources/prs/vllm/PR-23978.md) | Feature/vit attention unification# 23880 | 2025-08-30 |  | attention |
| [#23991](../sources/prs/vllm/PR-23991.md) | [Model] Add LongCat-Flash  | 2025-08-30 | kernel-fusion | decode, fp4, fp8 |
| [#23994](../sources/prs/vllm/PR-23994.md) | [BUGFIX] GPTQ quantization compatibility for Qwen3 MOE models (AutoGPTQ and AutoRound-GPTQ) | 2025-08-30 |  | moe, quantization |
| [#23929](../sources/prs/vllm/PR-23929.md) | [BUGFIX ] fix undefined silu_and_mul_nvfp4_quant | 2025-08-29 |  | fp4, nvfp4 |
| [#23972](../sources/prs/vllm/PR-23972.md) | [Kernel] Faster pre-processing time for W4A8 | 2025-08-29 |  | quantization |
| [#23791](../sources/prs/vllm/PR-23791.md) | [Kernel] cuda kernels for upcoming decode context parallel feature | 2025-08-28 |  | attention, decode |
| [#23798](../sources/prs/vllm/PR-23798.md) | [Misc] add reorder_batch AttentionMetadataBuilder | 2025-08-28 |  | attention |
| [#23809](../sources/prs/vllm/PR-23809.md) | [fix]: add Arm 4bit fused moe support | 2025-08-28 | kernel-fusion | kernel-fusion, moe, quantization |
| [#23819](../sources/prs/vllm/PR-23819.md) | [Model][gpt-oss] Support DP+EP for GPT-OSS with FlashInfer trtllm-gen MoE | 2025-08-28 | kernel-fusion | fp4, kernel-fusion, moe |
| [#23727](../sources/prs/vllm/PR-23727.md) | [Bugfix][Misc] Fix silu_and_mul_nvfp4_quant issue and extract common utils for nvfp4 kernel source files | 2025-08-27 | kernel-fusion, pipeline-stages | fp4, kernel-fusion, moe |
| [#23732](../sources/prs/vllm/PR-23732.md) | [FlashInfer] Cache hyper params in metadata builder | 2025-08-27 |  | attention |
| [#23734](../sources/prs/vllm/PR-23734.md) | [Feature] Support Decode Context Parallel (DCP) for MLA | 2025-08-27 | pipeline-stages | attention, decode, mla |
| [#23737](../sources/prs/vllm/PR-23737.md) | [BugFix][FlashInfer] Fix potential race condition for paged_kv_indptr_cpu | 2025-08-27 |  | attention |
| [#23745](../sources/prs/vllm/PR-23745.md) | [Feat][EPLB] A novel static EPLB placement strategy for MoE models. | 2025-08-27 | kernel-fusion | kernel-fusion, moe |
| [#23608](../sources/prs/vllm/PR-23608.md) | DP/EP Support for gpt-oss with deepep-ht comm kernel on SM100 | 2025-08-26 | kernel-fusion | fp4, fp8, kernel-fusion |
| [#23647](../sources/prs/vllm/PR-23647.md) | [Flashinfer] Support Flashinfer TRTLLM FP8-qkv BF16/FP16-out Attention Kernel | 2025-08-26 | kernel-fusion | attention, decode, fp8 |
| [#23659](../sources/prs/vllm/PR-23659.md) | [Bugfix] Fix Marlin NVFP4 for modelopt | 2025-08-26 |  | fp4, nvfp4, quantization |
| [#23660](../sources/prs/vllm/PR-23660.md) | [Compile] Fix Compile Warning for `w4a8_mm_entry.cu` | 2025-08-26 |  | quantization |
| [#23664](../sources/prs/vllm/PR-23664.md) | [v1] Add cross-attention KV cache support for encoder-decoder models | 2025-08-26 |  | attention, decode |
| [#23666](../sources/prs/vllm/PR-23666.md) | [Feature] Add Hopper DeepGEMM E8M0 for DeepSeekV3.1 scale_fmt | 2025-08-26 | kernel-fusion | fp8, gemm, kernel-fusion |
| [#23671](../sources/prs/vllm/PR-23671.md) | [NVIDIA] Support SiluMul + NVFP4 quant fusion | 2025-08-26 | kernel-fusion, pipeline-stages | fp4, kernel-fusion, nvfp4 |
| [#23693](../sources/prs/vllm/PR-23693.md) | [Core/DBO][1/N] Add Dual-Batch Overlap mechanism to VLLM | 2025-08-26 | kernel-fusion | attention, decode, kernel-fusion |
| [#23536](../sources/prs/vllm/PR-23536.md) | [V1][P/D]P2pNcclConnector supports flashinfer | 2025-08-25 |  | clc |
| [#23537](../sources/prs/vllm/PR-23537.md) | Update Flashinfer to  0.2.14.post1 | 2025-08-25 | kernel-fusion | fp4, kernel-fusion, quantization |
| [#23585](../sources/prs/vllm/PR-23585.md) | [Misc] Simplify FlashInfer attention metadata | 2025-08-25 |  | attention |
| [#23478](../sources/prs/vllm/PR-23478.md) | fix incompatibililty with non cuda platform for nvfp4 | 2025-08-24 | kernel-fusion | fp4, kernel-fusion, nvfp4 |
| [#23485](../sources/prs/vllm/PR-23485.md) | fix(v1/kv_cache): resolve async KV transfer bug in cascade attention | 2025-08-24 |  | attention |
| [#23490](../sources/prs/vllm/PR-23490.md) | [Bugfix] Fix Qwen3 MoE GPTQ inference | 2025-08-24 |  | moe |
| [#23465](../sources/prs/vllm/PR-23465.md) | [Attention][FA3] Update FA3 to include new swizzle optimization | 2025-08-23 | swizzling | attention, mla, swizzling |
| [#23424](../sources/prs/vllm/PR-23424.md) | [Bugfix] Fixing division by zero in triton_attn if query_heads/kv_heads > 16  | 2025-08-22 |  | attention |
| [#23439](../sources/prs/vllm/PR-23439.md) | [Perf] Warmup FlashInfer attention during startup | 2025-08-22 |  | attention |
| [#23264](../sources/prs/vllm/PR-23264.md) | [ROCm][Aiter] Add triton fp8 bmm kernel for mla | 2025-08-20 |  | attention, fp8, mla |
| [#23265](../sources/prs/vllm/PR-23265.md) | [Perf] Small optimizations for silu_mul_fp8_quant_deep_gemm | 2025-08-20 | kernel-fusion | fp8, gemm, kernel-fusion |
| [#23273](../sources/prs/vllm/PR-23273.md) | [Kernels] Overlap shared experts with send/recv | 2025-08-20 | kernel-fusion | fp4, fp8, kernel-fusion |
| [#23274](../sources/prs/vllm/PR-23274.md) | [Kernel] Add fused grouped_topk kernel for MoE | 2025-08-20 | kernel-fusion | kernel-fusion, moe |
| [#23280](../sources/prs/vllm/PR-23280.md) | [Perf] Use upstream CUTLASS for SM90 Block FP8 kernel | 2025-08-20 |  | fp8, gemm, quantization |
| [#23287](../sources/prs/vllm/PR-23287.md) | [Compile] Fix Compile Warning SM100 Cutlass MLA | 2025-08-20 |  | attention, mla |
| [#23294](../sources/prs/vllm/PR-23294.md) | [Bug] Fix R1 Accuracy 0 Bug | 2025-08-20 |  | fp8, quantization |
| [#23297](../sources/prs/vllm/PR-23297.md) | [Attention] Allow V1 flash_attn to support cross-attention | 2025-08-20 |  | attention |
| [#23140](../sources/prs/vllm/PR-23140.md) | Fix nvfp4 swizzling | 2025-08-19 | swizzling | fp4, nvfp4, quantization |
| [#23146](../sources/prs/vllm/PR-23146.md) | [CPU] add cpu fused moe pytorch native implementation | 2025-08-19 | kernel-fusion | kernel-fusion, moe |
| [#23148](../sources/prs/vllm/PR-23148.md) | [XPU][Feature] fp8 online quantization support for XPU | 2025-08-19 |  | fp8, quantization |
| [#23174](../sources/prs/vllm/PR-23174.md) | Optimize input preparation for FlashInfer [2/N] | 2025-08-19 |  | attention |
| [#23185](../sources/prs/vllm/PR-23185.md) | [Attention] Optimize make_local_attention_virtual_batches for Flash Attention | 2025-08-19 |  | attention |
| [#23198](../sources/prs/vllm/PR-23198.md) | [kernel] Support W4A8 on Hopper | 2025-08-19 |  | fp8, quantization |
| [#23207](../sources/prs/vllm/PR-23207.md) | [Misc][qwen2_5_vl][torch.compile] Enable `supports_torch_compile` on generic nn.Module and demonstrate speedup on Qwen Vision model | 2025-08-19 |  | attention |
| [#23214](../sources/prs/vllm/PR-23214.md) | [Core] Always use tensor cores for Flashinfer Decode Wrapper | 2025-08-19 |  | attention, decode |
| [#23122](../sources/prs/vllm/PR-23122.md) | [Misc] Add @tdoublep as a maintainer of hybrid model and Triton-attention related code | 2025-08-18 |  | attention |
| [#23123](../sources/prs/vllm/PR-23123.md) | Add routed_scaling_factor to MoE grouped topk | 2025-08-18 | kernel-fusion | fp4, fp8, kernel-fusion |
| [#23125](../sources/prs/vllm/PR-23125.md) | [Bugfix] Fix accuracy issue when using flashinfer cutlass moe, TP=1 and modelopt. | 2025-08-18 | kernel-fusion | kernel-fusion, moe, quantization |
| [#23129](../sources/prs/vllm/PR-23129.md) | Update to flashinfer-python==0.2.12 and disable AOT compile for non-release image | 2025-08-18 | pipeline-stages | pipeline-stages |
| [#23137](../sources/prs/vllm/PR-23137.md) | [Log] Warning Once for Cutlass MLA  | 2025-08-18 |  | attention, mla |
| [#23045](../sources/prs/vllm/PR-23045.md) | [Kernel] CUTLASS MoE FP8: Integrate cuda moe permute/unpermute | 2025-08-17 | kernel-fusion | fp8, gemm, grouped-gemm |
| [#23046](../sources/prs/vllm/PR-23046.md) | [V1] address post issues related to #20059 (part 1); cascade attention reenable by default | 2025-08-17 |  | attention |
| [#23016](../sources/prs/vllm/PR-23016.md) | [Bugfix gpt-oss] Fix float32 convert for flashinfer sink support | 2025-08-16 |  | attention |
| [#23031](../sources/prs/vllm/PR-23031.md) | [Bugfix] fix qwen3 moe fp8 accuracy issue | 2025-08-16 |  | fp8, moe, quantization |
| [#23036](../sources/prs/vllm/PR-23036.md) | [Core] Support weight_loader_v2 for `UnquantizedLinearMethod` | 2025-08-16 |  | quantization |
| [#22991](../sources/prs/vllm/PR-22991.md) | [Fix] enable swap_ab for pplx problem size computation | 2025-08-15 |  | moe, quantization |
| [#23008](../sources/prs/vllm/PR-23008.md) | Use Blackwell FlashInfer MXFP4 MoE by default if available  | 2025-08-15 | kernel-fusion | fp4, kernel-fusion, moe |
| [#22887](../sources/prs/vllm/PR-22887.md) | [XPU] support data parallel for MoE models on XPU | 2025-08-14 | kernel-fusion | kernel-fusion, moe |
| [#22895](../sources/prs/vllm/PR-22895.md) | [Kernel] Added flashinfer fp8 per-tensor gemms | 2025-08-14 | kernel-fusion, pipeline-stages | fp8, gemm, kernel-fusion |
| [#22934](../sources/prs/vllm/PR-22934.md) | [Bugfix] Fix DeepSeek MTP | 2025-08-14 |  | moe |
| [#22738](../sources/prs/vllm/PR-22738.md) | [Bugfix] Fix default enable for CUTLASS MLA on SM100 | 2025-08-13 |  | tcgen05, mla, gemm |
| [#22785](../sources/prs/vllm/PR-22785.md) | Fix GGUF loader for Qwen3 MoE. | 2025-08-13 |  | moe |
| [#22797](../sources/prs/vllm/PR-22797.md) | [FIXBUG] Add return_success parameter to moe_wna16_weight_loader function | 2025-08-13 |  | moe, quantization |
| [#22832](../sources/prs/vllm/PR-22832.md) | [Model] Modify the gate implementation of glm4_moe | 2025-08-13 |  | moe |
| [#22703](../sources/prs/vllm/PR-22703.md) | [NVIDIA] Support Flashinfer TRTLLM FP8-q/kv NVFP4-out Attention Kernel | 2025-08-12 | kernel-fusion | attention, decode, fp4 |
| [#22758](../sources/prs/vllm/PR-22758.md) | fp8 kv cache support fix for torch.compile | 2025-08-12 |  | attention, fp8, quantization |
| [#22613](../sources/prs/vllm/PR-22613.md) | Upgrade FlashInfer to v0.2.11 | 2025-08-11 |  | gemm |
| [#22637](../sources/prs/vllm/PR-22637.md) | [Bugfix] Fix ModernBert load & Enable sliding window attention for bidirectional attention. | 2025-08-11 |  | attention |
| [#22672](../sources/prs/vllm/PR-22672.md) | Support multiple attention groups for KV sharing | 2025-08-11 |  | attention |
| [#22674](../sources/prs/vllm/PR-22674.md) | [Quantization] Expand compressed-tensors MoE matching logic to support NFP4 + FP8 MoEs | 2025-08-11 |  | fp4, fp8, moe |
| [#22678](../sources/prs/vllm/PR-22678.md) | Force TRTLLM attention for gpt-oss on SM100 | 2025-08-11 |  | attention |
| [#22478](../sources/prs/vllm/PR-22478.md) | [Attention] FA3 Attention Sinks Perf Boost | 2025-08-08 |  | attention |
| [#22511](../sources/prs/vllm/PR-22511.md) | Fix Llama4 FlashInfer FP4 MoE issues | 2025-08-08 | kernel-fusion | fp4, kernel-fusion, moe |
| [#22514](../sources/prs/vllm/PR-22514.md) | [Model] Add Ernie4.5 VL Model Support | 2025-08-08 |  | moe |
| [#22527](../sources/prs/vllm/PR-22527.md) | Quantization: support FP4 quantized models on AMD CDNA2/CDNA3 GPUs | 2025-08-08 |  | fp4, quantization |
| [#22535](../sources/prs/vllm/PR-22535.md) | Fix torch version check for SM100 mxfp4  | 2025-08-08 | kernel-fusion | fp4, kernel-fusion, moe |
| [#22421](../sources/prs/vllm/PR-22421.md) | [gpt-oss] triton kernel mxfp4 | 2025-08-07 | kernel-fusion | fp4, kernel-fusion, moe |
| [#22426](../sources/prs/vllm/PR-22426.md) | [bugfix] Fix Llama3/4 issues caused by FlashInfer 0.2.10 | 2025-08-07 |  | attention, quantization |
| [#22468](../sources/prs/vllm/PR-22468.md) | [Quantization]: Support compressed-tensors mixed-precision model loading | 2025-08-07 |  | quantization |
| [#22313](../sources/prs/vllm/PR-22313.md) | Upgrade FA3 for attention sink | 2025-08-06 |  | attention |
| [#22314](../sources/prs/vllm/PR-22314.md) | [Bugfix] Add proper comparison for package versions | 2025-08-06 |  | attention, decode, quantization |
| [#22329](../sources/prs/vllm/PR-22329.md) | [ROCm] Add attention sink to use_rocm_custom_paged_attention | 2025-08-06 |  | attention |
| [#22339](../sources/prs/vllm/PR-22339.md) | [gpt-oss] flashinfer mxfp4 | 2025-08-06 | kernel-fusion | fp4, kernel-fusion, moe |
| [#22368](../sources/prs/vllm/PR-22368.md) | [BugFix] Fix triton compile error in `kernel_unified_attention_2/3d` caused by attention sinks | 2025-08-06 |  | attention |
| [#22389](../sources/prs/vllm/PR-22389.md) | Update `flashinfer-python==0.2.10` | 2025-08-06 |  | gemm |
| [#22399](../sources/prs/vllm/PR-22399.md) | [Bug] Fix B200 DeepGEMM E8M0 Accuracy Issue | 2025-08-06 |  | fp8, gemm, quantization |
| [#22222](../sources/prs/vllm/PR-22222.md) | Fp8 paged attention update | 2025-08-05 |  | attention, fp8 |
| [#22255](../sources/prs/vllm/PR-22255.md) | [bugfix] fix blackwell deepep installation | 2025-08-05 |  | gemm |
| [#22260](../sources/prs/vllm/PR-22260.md) | [Bugfix] Fix MoE BNB version | 2025-08-05 |  | moe, quantization |
| [#22273](../sources/prs/vllm/PR-22273.md) | Support encoder_only attention for FlexAttention | 2025-08-05 |  | attention |
| [#22278](../sources/prs/vllm/PR-22278.md) | [Bugfix] Fix 3D input passed into cutlass_scaled_mm | 2025-08-05 |  | gemm |
| [#22154](../sources/prs/vllm/PR-22154.md) | [fix] fix correct assertion syntax error in attention utils. | 2025-08-03 |  | attention |
| [#22131](../sources/prs/vllm/PR-22131.md) | [Kernel] Add support for block FP8 on SM120 (NVIDIA 5090 and RTX PRO 6000) | 2025-08-02 |  | fp8, quantization |
| [#22095](../sources/prs/vllm/PR-22095.md) | [NVIDIA] Support Flashinfer TRT-LLM Prefill Attention Kernel | 2025-08-01 | pipeline-stages | attention, decode, pipeline-stages |
| [#22100](../sources/prs/vllm/PR-22100.md) | [EPLB] Support ernie4.5-moe | 2025-08-01 |  | moe |
| [#22017](../sources/prs/vllm/PR-22017.md) | [BUGFIX] KeyError 'layers.14.mlp.gate.g_idx' for Qwen3-MoE with GPTQ on ROCm | 2025-07-31 |  | moe |
| [#21893](../sources/prs/vllm/PR-21893.md) | [Bugfix] Check NVIDIA artifactory is accessible before using flashinfer cubin kernels | 2025-07-30 |  | attention, mla |
| [#21924](../sources/prs/vllm/PR-21924.md) | [Qwen3] Enable dual-chunk-attention support for Qwen3 models. | 2025-07-30 |  | attention, moe |
| [#21963](../sources/prs/vllm/PR-21963.md) | Fix Flashinfer CUTLASS MOE Allgather | 2025-07-30 | kernel-fusion | kernel-fusion, moe |
| [#21701](../sources/prs/vllm/PR-21701.md) | update flashinfer to v0.2.9rc2 | 2025-07-28 |  | gemm |
| [#21716](../sources/prs/vllm/PR-21716.md) | [NVIDIA] Support Flashinfer TRTLLM FP8-q/kv/out Attention Kernel | 2025-07-28 | kernel-fusion, pipeline-stages | attention, decode, fp8 |
| [#21733](../sources/prs/vllm/PR-21733.md) | feat: Add Support GPTQ Quantization MOE on ROCM vllm serve | 2025-07-28 | kernel-fusion | kernel-fusion, moe, quantization |
| [#21759](../sources/prs/vllm/PR-21759.md) | [Logs] Change flashinfer sampler logs to once | 2025-07-28 |  | gemm |
| [#21761](../sources/prs/vllm/PR-21761.md) | [Perf] Disable chunked local attention by default with llama4 | 2025-07-28 |  | attention |
| [#21691](../sources/prs/vllm/PR-21691.md) | [BugFix] Fix IMA FlashMLA full cuda-graph and DP + Update FlashMLA | 2025-07-27 |  | attention, mla |
| [#21643](../sources/prs/vllm/PR-21643.md) | [xpu]support moe models on XPU platform | 2025-07-26 | kernel-fusion | kernel-fusion, moe |
| [#21664](../sources/prs/vllm/PR-21664.md) | support `torch.compile` for bailing moe | 2025-07-26 |  | moe |
| [#21588](../sources/prs/vllm/PR-21588.md) | [Attention] Support multiple attention metadata builders per kv_cache_spec  + proper local attention no hybrid kv cache fix | 2025-07-25 |  | attention, decode |
| [#21590](../sources/prs/vllm/PR-21590.md) | Override attention metadata for fast prefill in some KV sharing setups | 2025-07-25 |  | attention, gemm, prefill |
| [#21639](../sources/prs/vllm/PR-21639.md) | [Feature] Add Flashinfer MoE Support for Compressed Tensor NVFP4 | 2025-07-25 |  | fp4, moe, nvfp4 |
| [#21485](../sources/prs/vllm/PR-21485.md) | update flashinfer to v0.2.9rc1 | 2025-07-24 |  | attention |
| [#21497](../sources/prs/vllm/PR-21497.md) | [MoE] More balanced expert sharding | 2025-07-24 | kernel-fusion | kernel-fusion, moe |
| [#21499](../sources/prs/vllm/PR-21499.md) | [NVIDIA] Fix Llama4 Scout FP4 functionality issues | 2025-07-24 | kernel-fusion | fp4, kernel-fusion, moe |
| [#21548](../sources/prs/vllm/PR-21548.md) | Enable 4bit bnb prequant MOE | 2025-07-24 |  | moe |
| [#21419](../sources/prs/vllm/PR-21419.md) | [V1] Fix local chunked attention always disabled | 2025-07-23 |  | attention |
| [#21420](../sources/prs/vllm/PR-21420.md) | [Bugfix][CUDA] fixes CUDA FP8 kv cache dtype supported | 2025-07-23 |  | fp8 |
| [#21428](../sources/prs/vllm/PR-21428.md) | [BugFix] Fix shared storage connector load kv only load attention layer | 2025-07-23 |  | attention |
| [#21465](../sources/prs/vllm/PR-21465.md) | [Bug] Fix Compressed Tensor NVFP4 `cutlass_fp4_group_mm` illegal memory access | 2025-07-23 |  | fp4, moe, nvfp4 |
| [#21340](../sources/prs/vllm/PR-21340.md) | [TPU][Bugfix] fix moe layer | 2025-07-22 | kernel-fusion | kernel-fusion, moe |
| [#21370](../sources/prs/vllm/PR-21370.md) | [Quantization] Enable BNB support for more MoE models | 2025-07-22 |  | moe, quantization |
| [#21408](../sources/prs/vllm/PR-21408.md) | Update flashinfer CUTLASS NVFP4 MoE Kernel to use per expert global scaling factor | 2025-07-22 | kernel-fusion | fp4, kernel-fusion, moe |
| [#21411](../sources/prs/vllm/PR-21411.md) | [NVIDIA] Explicitly disable shuffled weights for flashinfer blockscale moe fp8 kernels | 2025-07-22 | kernel-fusion | fp8, kernel-fusion, moe |
| [#21412](../sources/prs/vllm/PR-21412.md) | [v1][attention] Support Hybrid Allocator + FlashInfer | 2025-07-22 |  | attention, decode, mla |
| [#21416](../sources/prs/vllm/PR-21416.md) | Updates to Flex + VLLm integration | 2025-07-22 |  | attention |
| [#21309](../sources/prs/vllm/PR-21309.md) | Support CUTLASS NVFP4 (w4a4) for Blackwell Geforce GPUs (SM120) | 2025-07-21 |  | fp4, moe, nvfp4 |
| [#21325](../sources/prs/vllm/PR-21325.md) | Fix Flashinfer Allreduce+Norm enable disable calculation based on `fi_allreduce_fusion_max_token_num` | 2025-07-21 | kernel-fusion | kernel-fusion |
| [#21331](../sources/prs/vllm/PR-21331.md) | Support Tensorrt-LLM MoE fp4 for low-latency | 2025-07-21 | kernel-fusion | fp4, kernel-fusion, moe |
| [#21249](../sources/prs/vllm/PR-21249.md) | [v1] - Mamba1 Attention Metadata | 2025-07-20 |  | attention, moe |
| [#21270](../sources/prs/vllm/PR-21270.md) | Support encoder-only models without KV-Cache | 2025-07-20 |  | attention, decode |
| [#21229](../sources/prs/vllm/PR-21229.md) | [Feature][Kernel]FusedMoE LoRA | 2025-07-19 | kernel-fusion, pipeline-stages | kernel-fusion, moe, pipeline-stages |
| [#21153](../sources/prs/vllm/PR-21153.md) | [Attention][DBO] Add support for "splitting" the CommonAttentionMetadata | 2025-07-18 |  | attention |
| [#21166](../sources/prs/vllm/PR-21166.md) | [Feature][OCP MX] Support mxfp6 and mixed mxfp6-mxfp4 | 2025-07-18 | kernel-fusion | fp4, fp8, kernel-fusion |
| [#21187](../sources/prs/vllm/PR-21187.md) | [Bug] DeepGemm: Fix TypeError: per_block_cast_to_fp8() missing 1 required positional argument: 'use_ue8m0' for SM100 | 2025-07-18 |  | fp8, gemm |
| [#21188](../sources/prs/vllm/PR-21188.md) | [Attention] Clean up iRoPE in V1 | 2025-07-18 |  | attention |
| [#21197](../sources/prs/vllm/PR-21197.md) | [Kernel] Enable Hybrid Model Support in Triton Unified Attention Kernel | 2025-07-18 |  | attention |
| [#21088](../sources/prs/vllm/PR-21088.md) | [v1] Add Whisper model support (encoder-decoder) | 2025-07-17 | pipeline-stages | attention, decode, pipeline-stages |
| [#21116](../sources/prs/vllm/PR-21116.md) | [perf] Add fused MLA QKV + strided layernorm | 2025-07-17 | kernel-fusion | fp8, kernel-fusion, mla |
| [#21121](../sources/prs/vllm/PR-21121.md) | [Bugfix] Allocate less memory in non-batched CUTLASS MoE | 2025-07-17 | kernel-fusion | kernel-fusion, moe |
| [#21126](../sources/prs/vllm/PR-21126.md) | [Perf] Use FlashInfer RoPE for RotaryEmbedding.forward_cuda when available | 2025-07-17 |  | gemm |
| [#21137](../sources/prs/vllm/PR-21137.md) | [Attention] Optimize FlashInfer MetadataBuilder Build call | 2025-07-17 |  | attention |
| [#21069](../sources/prs/vllm/PR-21069.md) | Add FlashInfer allreduce RMSNorm Quant fusion | 2025-07-16 | kernel-fusion, pipeline-stages | kernel-fusion, pipeline-stages |
| [#21077](../sources/prs/vllm/PR-21077.md) | [Bugfix] Voxtral on Blackwell GPUs (RTX 50 series) | 2025-07-16 |  | attention |
| [#21078](../sources/prs/vllm/PR-21078.md) | [Kernel] Flashinfer MLA (trtllm-gen) decode kernel integration | 2025-07-16 | pipeline-stages | attention, decode, mla |
| [#21083](../sources/prs/vllm/PR-21083.md) | [Perf] Cuda Kernel for Per Token Group Quant | 2025-07-16 |  | fp8, quantization |
| [#20998](../sources/prs/vllm/PR-20998.md) | [Bugfix] Fix Mistral3 support on SM100/SM120 | 2025-07-15 |  | gemm |
| [#21003](../sources/prs/vllm/PR-21003.md) | Support mnnvl all2allv from Flashinfer | 2025-07-15 | kernel-fusion | fp4, kernel-fusion, moe |
| [#20903](../sources/prs/vllm/PR-20903.md) | [Kernel] DeepGemm MoE : Integrate triton permute / unpermute kernels  | 2025-07-14 | kernel-fusion | gemm, kernel-fusion, moe |
| [#20911](../sources/prs/vllm/PR-20911.md) | [Perf] Add swap_ab to SM90 FP8 non-block CUTLASS moe grouped gemm | 2025-07-14 |  | fp8, gemm, moe |
| [#20930](../sources/prs/vllm/PR-20930.md) | [Model] Pooling models default to using chunked prefill & prefix caching if supported. | 2025-07-14 |  | gemm, prefill |
| [#20932](../sources/prs/vllm/PR-20932.md) | [Misc] Qwen MoE model supports LoRA | 2025-07-14 |  | moe |
| [#20934](../sources/prs/vllm/PR-20934.md) | [Bugfix] Switch bailout logic for kv-cache-dtype with SM100 Flashinfer | 2025-07-14 |  | gemm |
| [#20936](../sources/prs/vllm/PR-20936.md) | Fall back if flashinfer comm module not found | 2025-07-14 | kernel-fusion | kernel-fusion |
| [#20811](../sources/prs/vllm/PR-20811.md) | [v1][core] Support for attention free models | 2025-07-11 |  | attention |
| [#20815](../sources/prs/vllm/PR-20815.md) | [Feature][EPLB] Add eplb support for Qwen3 | 2025-07-11 |  | moe |
| [#20825](../sources/prs/vllm/PR-20825.md) | [Bugfix] Fix a couple PPLX+CUTLASS MoE bugs | 2025-07-11 | kernel-fusion | kernel-fusion, moe, quantization |
| [#20833](../sources/prs/vllm/PR-20833.md) | [Bug] Fix DeepGemm for EP low latency case | 2025-07-11 | kernel-fusion | gemm, kernel-fusion, moe |
| [#20841](../sources/prs/vllm/PR-20841.md) | [Perf] Use Triton instead of Torch for DeepGEMM Per Token Group Quant | 2025-07-11 | kernel-fusion | fp8, gemm, kernel-fusion |
| [#20736](../sources/prs/vllm/PR-20736.md) | GLM-4.5 Model Support | 2025-07-10 |  | moe |
| [#20762](../sources/prs/vllm/PR-20762.md) | [Performance] Performance improvements in non-blockwise fp8 CUTLASS MoE | 2025-07-10 | kernel-fusion | fp8, gemm, grouped-gemm |
| [#20769](../sources/prs/vllm/PR-20769.md) | SM100 Cutlass MLA decode with unrestricted num_heads (< 128) for DeepSeek TP | 2025-07-10 | tile-scheduling | attention, decode, flash-attention |
| [#20781](../sources/prs/vllm/PR-20781.md) | [fix]: disable cutlass block scaled group gemm for EP | 2025-07-10 | kernel-fusion | gemm, kernel-fusion, moe |
| [#20691](../sources/prs/vllm/PR-20691.md) | Integration SM100 FlashInfer fused allreduce RMSNorm | 2025-07-09 | kernel-fusion | kernel-fusion |
| [#20640](../sources/prs/vllm/PR-20640.md) | [feat] enable SM100 CUTLASS block scaled group gemm for smaller batch sizes | 2025-07-08 | kernel-fusion | gemm, kernel-fusion, moe |
| [#20509](../sources/prs/vllm/PR-20509.md) | [Bugfix] Fix missing per_act_token parameter in compressed_tensors_moe | 2025-07-05 | kernel-fusion | kernel-fusion, moe |
| [#20500](../sources/prs/vllm/PR-20500.md) | [Perf] Reuse workspace for FP8+FP4 Marlin MoE | 2025-07-04 |  | fp4, fp8, moe |
| [#20447](../sources/prs/vllm/PR-20447.md) | [feat]: add SM100 support for cutlass FP8 groupGEMM | 2025-07-03 |  | fp8, gemm, moe |
| [#20453](../sources/prs/vllm/PR-20453.md) | Support Llama 4 for cutlass_moe_fp4 | 2025-07-03 | kernel-fusion | fp4, kernel-fusion, moe |
| [#20457](../sources/prs/vllm/PR-20457.md) | Support Llama 4 for fused_marlin_moe | 2025-07-03 | kernel-fusion | fp8, kernel-fusion, moe |
| [#20358](../sources/prs/vllm/PR-20358.md) | Update PyTorch to 2.8.0 | 2025-07-02 | pipeline-stages | attention, pipeline-stages |
| [#20396](../sources/prs/vllm/PR-20396.md) | [Kernel] SM90 CUTLASS FP8 GEMM: add support for swap AB + kernel tuning | 2025-07-02 |  | fp8, gemm, quantization |
| [#20308](../sources/prs/vllm/PR-20308.md) | [Kernel] Optimize Prefill Attention in Unified Triton Attention Kernel | 2025-07-01 |  | attention, prefill |
| [#20324](../sources/prs/vllm/PR-20324.md) | [Kernel][Bugfix] Fixup some warnings in nvfp4_blockwise_moe when CUDA < 12.8 | 2025-07-01 |  | fp4, moe, nvfp4 |
| [#20332](../sources/prs/vllm/PR-20332.md) | [Misc] DP : Add ExpertTokensMetadata | 2025-07-01 | kernel-fusion | gemm, kernel-fusion, moe |
| [#20270](../sources/prs/vllm/PR-20270.md) | [V1] [ROCm] Enable EP with AITER Fused MoE | 2025-06-30 | kernel-fusion | fp8, kernel-fusion, moe |
| [#20166](../sources/prs/vllm/PR-20166.md) | [Bugfix] Fix topk_ids indices_type for CUTLASS w8a8 FP8 MoE | 2025-06-27 | kernel-fusion | fp8, kernel-fusion, moe |
| [#20167](../sources/prs/vllm/PR-20167.md) | [Bugfix] Fix Maverick correctness by filling zero to cache space in cutlass_moe | 2025-06-27 | kernel-fusion | kernel-fusion, moe |
| [#20189](../sources/prs/vllm/PR-20189.md) | [Nixl] Heterogeneous TP support FlashInfer | 2025-06-27 |  | gemm |
| [#20141](../sources/prs/vllm/PR-20141.md) | [Bugfix] Fix some narrowing conversion warnings | 2025-06-26 |  | attention, fp4, mla |
| [#20142](../sources/prs/vllm/PR-20142.md) | Replace `multiply_add` with `homogeneous_multiply_add` to Address Clang Template Parameter Issue | 2025-06-26 | epilogue-fusion | epilogue-fusion |
| [#20152](../sources/prs/vllm/PR-20152.md) | [Bugfix] Mark 'hidden_states' as mutable in moe_forward registration. | 2025-06-26 | kernel-fusion | kernel-fusion, moe |
| [#20086](../sources/prs/vllm/PR-20086.md) | [Bugfix] Build moe_data for both sm100 and sm90 | 2025-06-25 |  | moe, quantization |
| [#20087](../sources/prs/vllm/PR-20087.md) |  [Feature] Integrate SM100 DeepGEMM support | 2025-06-25 | kernel-fusion | fp8, gemm, kernel-fusion |
| [#20101](../sources/prs/vllm/PR-20101.md) | Add ModelOpt Qwen3 nvfp4 support | 2025-06-25 |  | fp4, moe, nvfp4 |
| [#20016](../sources/prs/vllm/PR-20016.md) | Enable V1 for Hybrid SSM/Attention Models | 2025-06-24 |  | attention, moe |
| [#20034](../sources/prs/vllm/PR-20034.md) | [Attention] MLA - Flashinfer Ragged Prefill | 2025-06-24 |  | attention, mla, prefill |
| [#19990](../sources/prs/vllm/PR-19990.md) | [Quantization] Add compressed-tensors NVFP4 MoE Support | 2025-06-23 | kernel-fusion | fp4, kernel-fusion, moe |
| [#19879](../sources/prs/vllm/PR-19879.md) | [Quantization] Add compressed-tensors emulations support for NVFP4 | 2025-06-19 |  | fp4, nvfp4, quantization |
| [#19781](../sources/prs/vllm/PR-19781.md) | Fix FA2 fallback for Blackwell V1 | 2025-06-18 |  | gemm |
| [#19820](../sources/prs/vllm/PR-19820.md) | [Feature] Integrate new deepgemm | 2025-06-18 | kernel-fusion | fp8, gemm, kernel-fusion |
| [#19822](../sources/prs/vllm/PR-19822.md) | [Bugfix] Enable PP with AITER+V1 | 2025-06-18 |  | attention, mla |
| [#19757](../sources/prs/vllm/PR-19757.md) | [feat]: CUTLASS block scaled group gemm for SM100 | 2025-06-17 | kernel-fusion | fp8, gemm, grouped-gemm |
| [#19566](../sources/prs/vllm/PR-19566.md) | [Perf] Further tunings for SM100 FP8 CUTLASS kernel | 2025-06-15 | tile-scheduling | tcgen05, fp8, gemm |
| [#19667](../sources/prs/vllm/PR-19667.md) | [Kernels] Use empty for modular MoE workspaces | 2025-06-15 | kernel-fusion | kernel-fusion, moe |
| [#19648](../sources/prs/vllm/PR-19648.md) | Only build CUTLASS MoE kernels on Hopper | 2025-06-14 |  | moe |
| [#19561](../sources/prs/vllm/PR-19561.md) | [Bugfix] Don't attempt to use triton if no driver is active | 2025-06-12 |  | gemm |
| [#19492](../sources/prs/vllm/PR-19492.md) | [Bugfix][V1] Allow manual FlashAttention for Blackwell | 2025-06-11 |  | attention |
| [#19500](../sources/prs/vllm/PR-19500.md) | [Hardware][NVIDIA][kernel] Fp4 MOE quant kernel optimization | 2025-06-11 |  | fp4, moe, nvfp4 |
| [#19346](../sources/prs/vllm/PR-19346.md) | [Kernel] Apply torch.Tag.needs_fixed_stride_order only for torch==2.6.0 | 2025-06-09 | kernel-fusion | attention, kernel-fusion, mla |
| [#19351](../sources/prs/vllm/PR-19351.md) | [Core] Support Local Chunked Attention for Hybrid KV Cache | 2025-06-09 |  | attention |
| [#19118](../sources/prs/vllm/PR-19118.md) | [V1] Use FlashInfer by default on Blackwell GPUs | 2025-06-04 |  | gemm |
| [#19168](../sources/prs/vllm/PR-19168.md) | [Kernels] Add activation chunking logic to FusedMoEModularKernel | 2025-06-04 | kernel-fusion | fp8, gemm, kernel-fusion |
| [#19085](../sources/prs/vllm/PR-19085.md) | [Kernel] Support deep_gemm for linear methods | 2025-06-03 |  | fp8, gemm, quantization |
| [#19110](../sources/prs/vllm/PR-19110.md) | [Hardware][NVIDIA] FP4 MoE kernel optimization | 2025-06-03 | kernel-fusion | fp4, kernel-fusion, moe |
| [#18990](../sources/prs/vllm/PR-18990.md) | [ROCm] [AITER] [Bugfix] Patch for AITER commit `648764942e552a8bb5fe16026703716a81f05374` | 2025-05-31 | kernel-fusion | kernel-fusion, moe |
| [#18807](../sources/prs/vllm/PR-18807.md) | [BugFix] FA2 MLA Accuracy Issue | 2025-05-28 |  | attention, mla |
| [#18833](../sources/prs/vllm/PR-18833.md) | [P/D] Heterogeneous TP | 2025-05-28 |  | attention |
| [#18864](../sources/prs/vllm/PR-18864.md) | [Kernel] Enable fp8 support for pplx and BatchedTritonExperts. | 2025-05-28 | kernel-fusion | fp8, gemm, kernel-fusion |
| [#18762](../sources/prs/vllm/PR-18762.md) | [Kernel] Integrate CUTLASS MoE kernel with PPLX | 2025-05-27 | kernel-fusion | fp8, gemm, grouped-gemm |
| [#18778](../sources/prs/vllm/PR-18778.md) | [Perf] Tunings for SM100 FP8 CUTLASS kernel | 2025-05-27 |  | fp8, quantization |
| [#18596](../sources/prs/vllm/PR-18596.md) | [Hardware][AMD] integrate aiter chunked prefill into vllm | 2025-05-23 |  | attention, prefill |
| [#18564](../sources/prs/vllm/PR-18564.md) | Sm100 blockwise fp8 swap ab | 2025-05-22 |  | fp8, quantization |
| [#18465](../sources/prs/vllm/PR-18465.md) | [V1] Support `LLM.apply_model` | 2025-05-21 |  | fp4, fp8, moe |
| [#18440](../sources/prs/vllm/PR-18440.md) | [Bug] Fix moe_sum signature | 2025-05-20 |  | moe |
| [#18343](../sources/prs/vllm/PR-18343.md) | [Feature] Expert Parallelism Load Balancer (EPLB) | 2025-05-19 | kernel-fusion, pipeline-stages | fp8, kernel-fusion, moe |
| [#18316](../sources/prs/vllm/PR-18316.md) | [Build] Supports CUDA 12.6 and 11.8 after Blackwell Update | 2025-05-18 | pipeline-stages | pipeline-stages |
| [#18321](../sources/prs/vllm/PR-18321.md) | [Model]: Fused MoE for nomic-embed-text-v2-moe | 2025-05-18 | kernel-fusion | kernel-fusion, moe |
| [#18312](../sources/prs/vllm/PR-18312.md) | [Quantization] Add compressed-tensors NVFP4 support | 2025-05-17 |  | fp4, nvfp4, quantization |
| [#18046](../sources/prs/vllm/PR-18046.md) | [Kernel] Have rotary embeddings support tensors | 2025-05-13 |  | gemm |
| [#18049](../sources/prs/vllm/PR-18049.md) | Fix Broken macro for cutlass moe | 2025-05-13 |  | moe, quantization |
| [#18093](../sources/prs/vllm/PR-18093.md) | [Bugfix][ROCm] Use `chunked_prefill_paged_decode` as fallback for V1 attention on ROCm | 2025-05-13 |  | attention, decode, prefill |
| [#18000](../sources/prs/vllm/PR-18000.md) | Use NVFP4 Marlin for CompressedTensorsW4A16Fp4 | 2025-05-12 |  | fp4, nvfp4, quantization |
| [#17961](../sources/prs/vllm/PR-17961.md) | [BUG] [ROCm] [MLA] Fix variable name bug due to change in variable name in PR #17483 | 2025-05-11 |  | attention, mla |
| [#17945](../sources/prs/vllm/PR-17945.md) | [v1] Support multiple KV cache groups in GPU model runner | 2025-05-10 |  | attention, mla |
| [#17880](../sources/prs/vllm/PR-17880.md) | [Bugfix][ROCm] Fix AITER MLA V1 | 2025-05-09 |  | attention, mla |
| [#17912](../sources/prs/vllm/PR-17912.md) | [BugFix][AMD] Compatible patch for AITER lib after 04/20 | 2025-05-09 | kernel-fusion | attention, kernel-fusion, mla |
| [#17914](../sources/prs/vllm/PR-17914.md) | [Misc] Add compressed-tensors NVFP4A16 emulation support | 2025-05-09 |  | fp4, nvfp4, quantization |
| [#17918](../sources/prs/vllm/PR-17918.md) | use ceil_div in cutlass block scaling shape check | 2025-05-09 |  | fp8, quantization |
| [#17871](../sources/prs/vllm/PR-17871.md) | fix amd triton mla path | 2025-05-08 |  | attention, mla |
| [#17687](../sources/prs/vllm/PR-17687.md) | [Kernel] fp4 marlin kernel | 2025-05-06 | kernel-fusion | fp4, fp8, gemm |
| [#17668](../sources/prs/vllm/PR-17668.md) | [Attention] MLA move rotary embedding to cuda-graph region | 2025-05-05 |  | attention, mla |
| [#17523](../sources/prs/vllm/PR-17523.md) | [FEAT][ROCm]: Support AITER MLA on V1 Engine | 2025-05-01 | kernel-fusion | attention, kernel-fusion, mla |
| [#17483](../sources/prs/vllm/PR-17483.md) | [v1] Pass BlockTable and KVCacheSpec to AttentionMetadataBuilders | 2025-04-30 |  | attention, mla |
| [#17484](../sources/prs/vllm/PR-17484.md) | [Attention] MLA move o_proj q_proj into cuda-graph region | 2025-04-30 |  | attention, mla |
| [#17494](../sources/prs/vllm/PR-17494.md) | [BugFix] Fix mla cpu - missing 3 required positional arguments | 2025-04-30 |  | attention, mla |
| [#17394](../sources/prs/vllm/PR-17394.md) | [v1] AttentionMetadata for each layer | 2025-04-29 |  | attention, decode, mla |
| [#17414](../sources/prs/vllm/PR-17414.md) | Fix noisy warning for uncalibrated q_scale/p_scale | 2025-04-29 |  | quantization |
| [#17280](../sources/prs/vllm/PR-17280.md) | [NVIDIA] Support Cutlass w8a8 FP8 for Blackwell Geforce GPUs (sm120) | 2025-04-28 |  | fp8, quantization |
| [#17283](../sources/prs/vllm/PR-17283.md) | [BugFix] Fix cascade attention - RuntimeError: scheduler_metadata must have shape (metadata_size) | 2025-04-28 |  | attention |
| [#17289](../sources/prs/vllm/PR-17289.md) | [Misc][ROCm] Exclude `cutlass_mla_decode` for ROCm build | 2025-04-28 |  | decode, mla |
| [#16032](../sources/prs/vllm/PR-16032.md) | [NVIDIA] Support Cutlass MLA for Blackwell GPUs | 2025-04-27 | warp-specialization, persistent-kernel | tcgen05, mla, moe |
| [#17267](../sources/prs/vllm/PR-17267.md) | [BugFix] Fix vllm_flash_attn install issues | 2025-04-27 |  | attention, mla |
| [#17222](../sources/prs/vllm/PR-17222.md) | [Bugfix] Get a specific type of layer from forward context | 2025-04-26 |  | attention |
| [#17180](../sources/prs/vllm/PR-17180.md) | [Bugfix] gemma[2,3] interleaved attention when sliding window is disabled | 2025-04-25 |  | attention, gemm |
| [#17091](../sources/prs/vllm/PR-17091.md) | [Bugfix] Add contiguous call inside rope kernel wrapper | 2025-04-24 |  | attention, mla |
| [#17110](../sources/prs/vllm/PR-17110.md) | [FEAT] [ROCm]: Add AITER CK 2 Stages MoE support | 2025-04-24 | kernel-fusion | fp8, kernel-fusion, moe |
| [#17139](../sources/prs/vllm/PR-17139.md) | [ROCm][FP8][Kernel] FP8 quantization fused into Custom Paged Attention | 2025-04-24 | kernel-fusion | attention, fp8, kernel-fusion |
| [#17082](../sources/prs/vllm/PR-17082.md) | Fix `numel()` downcast in vllm/csrc/moe/moe_align_sum_kernels.cu +2 | 2025-04-23 |  | moe, tma |
| [#17004](../sources/prs/vllm/PR-17004.md) | [ROCm][Kernel][V1] Enable AMD Radeon GPU Custom Paged Attention on v1 | 2025-04-22 |  | attention, decode, prefill |
| [#16902](../sources/prs/vllm/PR-16902.md) | [Bugfix] Triton FA function takes no keyword arguments | 2025-04-21 |  | attention, mla |
| [#16946](../sources/prs/vllm/PR-16946.md) | Update Qwen1.5-MoE-W4A16-compressed-tensors.yaml | 2025-04-21 |  | moe |
| [#16828](../sources/prs/vllm/PR-16828.md) | [Kernel] Unified Triton kernel that doesn't distinguish between prefill + decode | 2025-04-18 |  | attention, decode, prefill |
| [#16850](../sources/prs/vllm/PR-16850.md) | [Kernel] some optimizations for dense marlin and moe marlin | 2025-04-18 | kernel-fusion | fp8, gemm, kernel-fusion |
| [#16854](../sources/prs/vllm/PR-16854.md) | [Bugfix] Fix moe weight losing all extra attrs after `process_weights_after_loading`. | 2025-04-18 | kernel-fusion | kernel-fusion, moe, quantization |
| [#16859](../sources/prs/vllm/PR-16859.md) | Update PyTorch to 2.7.0 | 2025-04-18 | pipeline-stages | attention, pipeline-stages |
| [#16861](../sources/prs/vllm/PR-16861.md) | [Kernel] Add expert_map support to Cutlass FP8 MOE | 2025-04-18 | kernel-fusion | fp8, kernel-fusion, moe |
| [#16864](../sources/prs/vllm/PR-16864.md) | [Attention] FA3 decode perf improvement - single mma warp group support for head dim 128 | 2025-04-18 |  | attention, decode |
| [#16752](../sources/prs/vllm/PR-16752.md) | [FEAT] [ROCm]: AITER Fused MOE V1 Support | 2025-04-17 | kernel-fusion | kernel-fusion, moe, quantization |
| [#16756](../sources/prs/vllm/PR-16756.md) | [torch.compile][ROCm] Fuse quantization onto attention using a torch.compile pass | 2025-04-17 | kernel-fusion, pipeline-stages | attention, kernel-fusion, mla |
| [#16760](../sources/prs/vllm/PR-16760.md) | [misc] ignore marlin_moe_wna16 local gen codes | 2025-04-17 |  | moe |
| [#16780](../sources/prs/vllm/PR-16780.md) | [Kernel] GGUF MoeVec kernel | 2025-04-17 |  | moe, quantization |
| [#16801](../sources/prs/vllm/PR-16801.md) | [BugFix] Accuracy fix for llama4 int4 - improperly casted scales | 2025-04-17 | kernel-fusion | kernel-fusion, moe |
| [#16727](../sources/prs/vllm/PR-16727.md) | [ROCm] Add aiter tkw1 kernel for Llama4 fp8 | 2025-04-16 | kernel-fusion | fp8, kernel-fusion, moe |
| [#16745](../sources/prs/vllm/PR-16745.md) | Support W8A8 INT8 MoE for compressed-tensors | 2025-04-16 |  | moe, quantization |
| [#16673](../sources/prs/vllm/PR-16673.md) | [MLA] Simplification to batch P/D reordering | 2025-04-15 |  | attention, mla |
| [#16674](../sources/prs/vllm/PR-16674.md) | [ROCM] enable aiter fused moe kernel for llama4 bf16 checkpoints | 2025-04-15 | kernel-fusion | kernel-fusion, moe |
| [#16684](../sources/prs/vllm/PR-16684.md) | [V1] V1 FlashInfer Attention | 2025-04-15 |  | attention, mla |
| [#16605](../sources/prs/vllm/PR-16605.md) | Allocate kv_cache with stride order | 2025-04-14 |  | attention |
| [#16537](../sources/prs/vllm/PR-16537.md) | Enable PTPC FP8 for CompressedTensorsW8A8Fp8MoEMethod (triton fused_moe) | 2025-04-12 | kernel-fusion | fp8, kernel-fusion, moe |
| [#16439](../sources/prs/vllm/PR-16439.md) | [Llama4] Enable attention temperature tuning by default for long context (>32k) | 2025-04-11 |  | attention |
| [#16457](../sources/prs/vllm/PR-16457.md) | [Perf]Optimize rotary_emb implementation to use Triton operator for improved inference performance | 2025-04-11 |  | gemm |
| [#16362](../sources/prs/vllm/PR-16362.md) | [Hardware/NVIDIA/Kernel] [Functional Enablement] [1/N] Enable nvidia/DeepSeek-R1-FP4 Model | 2025-04-09 | kernel-fusion | fp4, kernel-fusion, moe |
| [#16366](../sources/prs/vllm/PR-16366.md) | [Kernel] Support W8A8 channel-wise weights and per-token activations in triton fused_moe_kernel | 2025-04-09 | kernel-fusion | fp8, kernel-fusion, moe |
| [#16263](../sources/prs/vllm/PR-16263.md) | [Hardware][AMD] Improve OAM device ID + llama4 Maverick MOE tuning | 2025-04-08 | kernel-fusion | kernel-fusion, moe |
| [#16173](../sources/prs/vllm/PR-16173.md) | [Kernel] support merge_attn_states CUDA kernel, 3x speedup | 2025-04-07 |  | attention, mla |
| [#16198](../sources/prs/vllm/PR-16198.md) | [Bug] [ROCm] Fix Llama 4 Enablement Bug on ROCm: V0 ROCmFlashAttentionImpl and Triton Fused MoE bugs | 2025-04-07 | kernel-fusion | attention, kernel-fusion, moe |
| [#16203](../sources/prs/vllm/PR-16203.md) | [Model] use AutoWeightsLoader for phimoe,qwen2_moe,qwen3_moe | 2025-04-07 |  | moe |
| [#16113](../sources/prs/vllm/PR-16113.md) | Upstream Llama4 Support to Main | 2025-04-06 | kernel-fusion, pipeline-stages | attention, decode, fp8 |
| [#16071](../sources/prs/vllm/PR-16071.md) | [Kernel][Bugfix] Re-fuse triton moe weight application | 2025-04-04 | kernel-fusion | kernel-fusion, moe |
| [#16078](../sources/prs/vllm/PR-16078.md) | Add FlexAttention to V1 | 2025-04-04 |  | attention |
| [#16034](../sources/prs/vllm/PR-16034.md) | [ROCM] Add gfx950 to the custom attention archs | 2025-04-03 |  | attention |
| [#16038](../sources/prs/vllm/PR-16038.md) | [Kernel] Use moe_wna16 kernel for compressed tensors wna16 moe models | 2025-04-03 | kernel-fusion | kernel-fusion, moe, quantization |
| [#15945](../sources/prs/vllm/PR-15945.md) | [Hardware][Gaudi][BugFix] fix arguments of hpu fused moe | 2025-04-02 | kernel-fusion | kernel-fusion, moe |
| [#15946](../sources/prs/vllm/PR-15946.md) | [Bugfix] fix use_atomic_add support of marlin kernel when using v1 engine | 2025-04-02 |  | quantization |
| [#15956](../sources/prs/vllm/PR-15956.md) | Modularize fused experts and integrate PPLX kernels | 2025-04-02 | kernel-fusion | attention, fp8, gemm |
| [#15961](../sources/prs/vllm/PR-15961.md) | Add support to modelopt quantization of Mixtral model | 2025-04-02 |  | quantization |
| [#15848](../sources/prs/vllm/PR-15848.md) | [Bugfix] Fix cache block size calculation for CPU MLA | 2025-04-01 |  | mla |
| [#15893](../sources/prs/vllm/PR-15893.md) | [FEAT][ROCm]: Support AITER MLA | 2025-04-01 |  | attention, mla |
| [#15834](../sources/prs/vllm/PR-15834.md) | [V1] TPU - Fix fused MOE | 2025-03-31 | kernel-fusion | kernel-fusion, moe |
| [#15720](../sources/prs/vllm/PR-15720.md) | [ROCM][KERNEL] Paged attention for V1 | 2025-03-28 |  | attention, decode, prefill |
| [#15732](../sources/prs/vllm/PR-15732.md) | [TPU] Support sliding window and logit soft capping in the paged attention kernel for TPU. | 2025-03-28 |  | attention |
| [#15587](../sources/prs/vllm/PR-15587.md) | [Quantization] Fp8 Channelwise Dynamic Per Token GroupedGEMM | 2025-03-27 | kernel-fusion | fp8, gemm, kernel-fusion |
| [#15511](../sources/prs/vllm/PR-15511.md) | Use Cache Hinting for fused_moe kernel | 2025-03-26 | kernel-fusion | kernel-fusion, moe |
| [#15515](../sources/prs/vllm/PR-15515.md) | [moe][quant] add weight name case for offset | 2025-03-26 | kernel-fusion | kernel-fusion, moe |
| [#15433](../sources/prs/vllm/PR-15433.md) | [FEAT] [ROCm] Add AITER int8 scaled gemm kernel | 2025-03-25 |  | gemm, quantization |
| [#15456](../sources/prs/vllm/PR-15456.md) | [Kernel] Fix conflicting macro names for gguf kernels | 2025-03-25 |  | moe, quantization |
| [#15492](../sources/prs/vllm/PR-15492.md) | [BugFix] Fix nightly MLA failure (FA2 + MLA chunked prefill, i.e. V1, producing bad results) | 2025-03-25 |  | attention, mla, prefill |
| [#15354](../sources/prs/vllm/PR-15354.md) | [V1] Fully Transparent Implementation of CPU Offloading | 2025-03-23 |  | quantization |
| [#15319](../sources/prs/vllm/PR-15319.md) | Fix non-contiguous input passed to Marlin kernel | 2025-03-22 |  | quantization |
| [#15273](../sources/prs/vllm/PR-15273.md) | [Misc] Add attention mask pre-computation optimization back to Qwen2.5-VL | 2025-03-21 |  | attention |
| [#15289](../sources/prs/vllm/PR-15289.md) | [Model] Add Qwen3 and Qwen3MoE | 2025-03-21 |  | moe |
| [#15200](../sources/prs/vllm/PR-15200.md) | [Bugfix] Fix incorrect qwen2.5-vl attention mask pre-computation | 2025-03-20 |  | attention, decode |
| [#15211](../sources/prs/vllm/PR-15211.md) | [Bugfix] Fix use_cascade_attention handling for Alibi-based models on vllm/v1 | 2025-03-20 |  | attention |
| [#15001](../sources/prs/vllm/PR-15001.md) | [FEAT][ROCm] Integrate Paged Attention Kernel from AITER | 2025-03-18 |  | attention |
| [#14921](../sources/prs/vllm/PR-14921.md) | [V1] Default MLA to V1 | 2025-03-17 |  | mla |
| [#14967](../sources/prs/vllm/PR-14967.md) | [FEAT][ROCm] Integrate Fused MoE Kernels from AITER | 2025-03-17 | kernel-fusion | decode, fp8, kernel-fusion |
| [#14968](../sources/prs/vllm/PR-14968.md) | [FEAT] [ROCm]: Add AITER Block-Scaled GEMM Feature | 2025-03-17 |  | block-scale, fp8, gemm |
| [#14896](../sources/prs/vllm/PR-14896.md) | [V1][BugFix] Detect interleaved sliding window attention | 2025-03-16 |  | attention |
| [#14842](../sources/prs/vllm/PR-14842.md) | [Attention] Get rid of mla cache alignment | 2025-03-14 |  | attention, mla |
| [#14744](../sources/prs/vllm/PR-14744.md) | [Kernel][CPU] CPU MLA | 2025-03-13 |  | attention, decode, mla |
| [#14770](../sources/prs/vllm/PR-14770.md) | [Attention] MLA get rid of materialization | 2025-03-13 |  | attention, fp8, mla |
| [#14658](../sources/prs/vllm/PR-14658.md) | [Kernel] allow non-contiguous input for marlin kernel | 2025-03-12 |  | gemm, quantization |
| [#14660](../sources/prs/vllm/PR-14660.md) | [Model] Add support for Gemma 3 | 2025-03-12 |  | gemm |
| [#14667](../sources/prs/vllm/PR-14667.md) | [Bugfix][Kernel][CPU] Fix num_tokens in CPU rotary embedding kernel | 2025-03-12 |  | gemm |
| [#14681](../sources/prs/vllm/PR-14681.md) | [Bugfix][IPEX] Add `VLLM_CPU_MOE_PREPACK` to allow disabling MoE prepack when CPU does not support it | 2025-03-12 | kernel-fusion | kernel-fusion, moe |
| [#14613](../sources/prs/vllm/PR-14613.md) | [Kernel] GGUF MoE kernel | 2025-03-11 |  | moe, quantization |
| [#14540](../sources/prs/vllm/PR-14540.md) | [Perf] Improve MLA on V1 | 2025-03-10 |  | attention, mla |
| [#14555](../sources/prs/vllm/PR-14555.md) | [BugFix][TritonMLA] Process weights after model loading for GGUF | 2025-03-10 |  | mla |
| [#14568](../sources/prs/vllm/PR-14568.md) | permute/unpermute kernel for moe optimization | 2025-03-10 | kernel-fusion | fp8, gemm, grouped-gemm |
| [#14570](../sources/prs/vllm/PR-14570.md) | [Attention] Flash Attention 3 - fp8 | 2025-03-10 |  | attention, fp8, mla |
| [#14572](../sources/prs/vllm/PR-14572.md) | [BugFix/Build] Fix sparse kernels not getting built on hopper | 2025-03-10 |  | gemm |
| [#14578](../sources/prs/vllm/PR-14578.md) | [Quantization][FP8] Adding support for fp8 gemm layer input in fp8 | 2025-03-10 |  | fp8, gemm, quantization |
| [#14476](../sources/prs/vllm/PR-14476.md) | [Bugfix] DeepSeek Accuracy | 2025-03-08 |  | attention, mla |
| [#14396](../sources/prs/vllm/PR-14396.md) | [BugFix] Illegal Memory Access in the blockwise cutlass fp8 GEMMs | 2025-03-07 |  | fp8, gemm, tma |
| [#14447](../sources/prs/vllm/PR-14447.md) | [Kernel] moe wna16 marlin kernel | 2025-03-07 | kernel-fusion | kernel-fusion, moe, quantization |
| [#14454](../sources/prs/vllm/PR-14454.md) | [ROCm][Kernel] MoE weights padding | 2025-03-07 | kernel-fusion | fp8, kernel-fusion, moe |
| [#14327](../sources/prs/vllm/PR-14327.md) | fix minor miscalled method | 2025-03-06 |  | gemm |
| [#14354](../sources/prs/vllm/PR-14354.md) | [Build/BugFix] Fix hopper 12.8 build | 2025-03-06 |  | quantization |
| [#14383](../sources/prs/vllm/PR-14383.md) | Add cutlass support for blackwell fp8 blockwise gemm | 2025-03-06 |  | fp8, gemm, quantization |
| [#14384](../sources/prs/vllm/PR-14384.md) | [Perf] Reduce MLA CPU overheads in V1 | 2025-03-06 |  | attention, mla |
| [#14245](../sources/prs/vllm/PR-14245.md) | dynamic distpatch of fp8 kernels | 2025-03-05 | kernel-fusion | attention, fp8, gemm |
| [#14253](../sources/prs/vllm/PR-14253.md) | [BugFix] MLA + V1, illegal memory access and accuracy issues | 2025-03-05 |  | attention, mla |
| [#14255](../sources/prs/vllm/PR-14255.md) | [BugFix] Fix prefix caching V0 MLA | 2025-03-05 |  | attention, mla |
| [#14258](../sources/prs/vllm/PR-14258.md) | [Attention] FlashAttn MLA | 2025-03-05 |  | attention, mla |
| [#14276](../sources/prs/vllm/PR-14276.md) | [Misc] Add Qwen2MoeForCausalLM moe tuning support  | 2025-03-05 |  | moe |
| [#14310](../sources/prs/vllm/PR-14310.md) | [Hardware][TPU]Enable ragged paged attention kernel and resolve recompilation issue | 2025-03-05 |  | attention |
| [#14313](../sources/prs/vllm/PR-14313.md) | [Bug] Fix Attention when ignored in by quant_method | 2025-03-05 |  | attention |
| [#14316](../sources/prs/vllm/PR-14316.md) | [ROCm] Enable chunked prefill/paged attention in MLA on ROCm | 2025-03-05 |  | attention, mla, prefill |
| [#13798](../sources/prs/vllm/PR-13798.md) | add tcgen05 support for tcgen05 fp8 gemm | 2025-03-04 | tile-scheduling | tcgen05, fp8, gemm |
| [#14227](../sources/prs/vllm/PR-14227.md) | [V1][TPU] Support V1 Sampler for ragged attention | 2025-03-04 |  | attention |
| [#14244](../sources/prs/vllm/PR-14244.md) | [Hardware] Update the flash attn tag to support Blackwell | 2025-03-04 |  | attention |
| [#14138](../sources/prs/vllm/PR-14138.md) | [Kernel] optimize performance of gptq marlin kernel when n is small | 2025-03-03 |  | gemm, quantization |
| [#14155](../sources/prs/vllm/PR-14155.md) | [v1] Add comments to the new ragged paged attention Pallas kernel | 2025-03-03 |  | attention |
| [#14158](../sources/prs/vllm/PR-14158.md) | [V1][TPU] TPU multimodal model support for ragged attention | 2025-03-03 |  | attention |
| [#14097](../sources/prs/vllm/PR-14097.md) | [V1] Implement sliding window attention in kv_cache_manager | 2025-03-02 |  | attention |
| [#14068](../sources/prs/vllm/PR-14068.md) | [core] moe fp8 block quant tuning support | 2025-03-01 | kernel-fusion | fp8, kernel-fusion, moe |
| [#13972](../sources/prs/vllm/PR-13972.md) | [Kernel] CUTLASS grouped gemm fp8 MoE kernel | 2025-02-27 | epilogue-fusion, kernel-fusion | epilogue-fusion, fp8, gemm |
| [#13974](../sources/prs/vllm/PR-13974.md) | [Misc] Print FusedMoE detail info | 2025-02-27 | kernel-fusion | kernel-fusion, moe |
| [#13867](../sources/prs/vllm/PR-13867.md) | [Attention] Flash MLA for V1 | 2025-02-26 |  | attention, mla |
| [#13897](../sources/prs/vllm/PR-13897.md) | Fix mla prefill context performance | 2025-02-26 |  | attention, mla, prefill |
| [#13931](../sources/prs/vllm/PR-13931.md) | [V1] EP/TP MoE + DP Attention | 2025-02-26 | kernel-fusion | attention, kernel-fusion, moe |
| [#13844](../sources/prs/vllm/PR-13844.md) | [ROCm] Disable chunked prefill/prefix caching when running MLA on non-cuda platforms | 2025-02-25 |  | attention, mla, prefill |
| [#13747](../sources/prs/vllm/PR-13747.md) | [Kernel] FlashMLA integration | 2025-02-24 |  | attention, mla |
| [#13769](../sources/prs/vllm/PR-13769.md) | Fix CompressedTensorsWNA16MoE with grouped scales | 2025-02-24 |  | moe, quantization |
| [#13772](../sources/prs/vllm/PR-13772.md) | Fix precommit fail in fused_moe intermediate_cache2 chunking | 2025-02-24 | kernel-fusion | kernel-fusion, moe |
| [#13784](../sources/prs/vllm/PR-13784.md) | [Bugfix][Quantization] Fix FP8 + EP | 2025-02-24 | kernel-fusion | fp8, kernel-fusion, moe |
| [#13789](../sources/prs/vllm/PR-13789.md) | [Attention] MLA support for V1 | 2025-02-24 |  | attention, mla |
| [#13718](../sources/prs/vllm/PR-13718.md) | [core] Perf improvement for DSv3 on AMD GPUs | 2025-02-23 | kernel-fusion | attention, decode, fp8 |
| [#13725](../sources/prs/vllm/PR-13725.md) | [Bugfix] Support MLA for CompressedTensorsWNA16 | 2025-02-23 |  | attention, mla |
| [#13726](../sources/prs/vllm/PR-13726.md) | [V1] V1 Enablement Oracle  | 2025-02-23 | pipeline-stages | attention, decode, fp8 |
| [#13571](../sources/prs/vllm/PR-13571.md) | [NVIDIA] Support nvfp4 tcgen05 gemm | 2025-02-22 | fine-grained-quantization | tcgen05, nvfp4, fp4 |
| [#13693](../sources/prs/vllm/PR-13693.md) | [BugFix]  Illegal memory access for MoE On H20 | 2025-02-22 | kernel-fusion | kernel-fusion, moe |
| [#13650](../sources/prs/vllm/PR-13650.md) | [ROCM] fix native attention function call | 2025-02-21 |  | attention |
| [#13577](../sources/prs/vllm/PR-13577.md) | [ROCm][MoE] mi300 mixtral8x7B perf for specific BS | 2025-02-20 | kernel-fusion | kernel-fusion, moe |
| [#13620](../sources/prs/vllm/PR-13620.md) | [Bugfix] Fix max_num_batched_tokens for MLA | 2025-02-20 |  | mla |
| [#13625](../sources/prs/vllm/PR-13625.md) | [Kernel] Optimize moe intermediate_cache usage | 2025-02-20 | kernel-fusion | kernel-fusion, moe |
| [#13321](../sources/prs/vllm/PR-13321.md) | [Kernel] moe wna16 cuda kernel | 2025-02-15 | kernel-fusion | kernel-fusion, moe |
| [#13310](../sources/prs/vllm/PR-13310.md) | [Bugfix] Massage MLA's usage of flash attn for RoCM | 2025-02-14 |  | attention, mla |
| [#13181](../sources/prs/vllm/PR-13181.md) | Expand MLA to support most types of quantization | 2025-02-13 |  | attention, mla, quantization |
| [#13236](../sources/prs/vllm/PR-13236.md) | [Quant][Perf] Use moe_wna16 kernel by default for MoEs with many experts | 2025-02-13 |  | moe, quantization |
| [#13167](../sources/prs/vllm/PR-13167.md) | [Model] Deepseek GGUF support  | 2025-02-12 | kernel-fusion | kernel-fusion, moe, quantization |
| [#12978](../sources/prs/vllm/PR-12978.md) | [Kernel]Add streamK for block-quantized CUTLASS kernels | 2025-02-09 |  | fp8, gemm, quantization |
| [#12796](../sources/prs/vllm/PR-12796.md) | [Bugfix] Better FP8 supported defaults | 2025-02-06 |  | fp8, quantization |
| [#12807](../sources/prs/vllm/PR-12807.md) | [Attention] Use FA3 for MLA on Hopper | 2025-02-06 |  | attention, mla |
| [#12850](../sources/prs/vllm/PR-12850.md) | Optimize moe_align_block_size for deepseek_v3 | 2025-02-06 | kernel-fusion | kernel-fusion, moe |
| [#12757](../sources/prs/vllm/PR-12757.md) | [Misc] Update w2 scale loading for GPTQMarlinMoE | 2025-02-05 | kernel-fusion | kernel-fusion, moe, quantization |
| [#12777](../sources/prs/vllm/PR-12777.md) | [Kernel] Make rotary_embedding ops more flexible with input shape | 2025-02-05 |  | attention, mla |
| [#12784](../sources/prs/vllm/PR-12784.md) | [NVIDIA] Support nvfp4 quantization | 2025-02-05 |  | fp4, nvfp4, quantization |
| [#12729](../sources/prs/vllm/PR-12729.md) | [VLM] Add MLA with pure RoPE support for deepseek-vl2 models | 2025-02-04 |  | attention, mla |
| [#12676](../sources/prs/vllm/PR-12676.md) | [Perf] Mem align KV caches for CUDA devices (MLA perf improvement) | 2025-02-03 |  | attention, decode, mla |
| [#12695](../sources/prs/vllm/PR-12695.md) | [Core][AMD] Migrate fully transparent sleep mode to ROCm platform | 2025-02-03 |  | gemm |
| [#12696](../sources/prs/vllm/PR-12696.md) | [Bugfix][Kernel] Fix per-token/per-channel quantization for Hopper scaled mm | 2025-02-03 |  | quantization |
| [#12704](../sources/prs/vllm/PR-12704.md) | Squelch MLA warning for Compressed-Tensors Models | 2025-02-03 |  | mla |
| [#12662](../sources/prs/vllm/PR-12662.md) | [AMD][ROCm] Enable DeepSeek model on ROCm | 2025-02-02 |  | attention, fp8, mla |
| [#12637](../sources/prs/vllm/PR-12637.md) | Apply torch.compile to fused_moe/grouped_topk | 2025-02-01 | kernel-fusion | kernel-fusion, moe |
| [#12639](../sources/prs/vllm/PR-12639.md) | [Attention] MLA with chunked prefill | 2025-02-01 |  | attention, fp8, mla |
| [#12642](../sources/prs/vllm/PR-12642.md) | Disable chunked prefill and/or prefix caching when MLA is enabled  | 2025-02-01 |  | mla, prefill |
| [#12601](../sources/prs/vllm/PR-12601.md) | [Attention] Deepseek v3 MLA support with FP8 compute | 2025-01-31 |  | attention, fp8, mla |
| [#12574](../sources/prs/vllm/PR-12574.md) | [Kernel] port sgl moe_align_block_size kernels | 2025-01-30 | kernel-fusion | kernel-fusion, moe |
| [#12583](../sources/prs/vllm/PR-12583.md) | Expert Parallelism (EP) Support for DeepSeek Models | 2025-01-30 | kernel-fusion | fp8, kernel-fusion, moe |
| [#12587](../sources/prs/vllm/PR-12587.md) | [Kernel][Quantization] Integrate block-quantized CUTLASS kernels for DeepSeekV3 | 2025-01-30 |  | fp8, quantization |
| [#12558](../sources/prs/vllm/PR-12558.md) | [Misc][MoE] add Deepseek-V3 moe tuning support | 2025-01-29 |  | moe |
| [#12528](../sources/prs/vllm/PR-12528.md) | [Attention] MLA decode optimizations | 2025-01-28 |  | attention, decode, mla |
| [#12417](../sources/prs/vllm/PR-12417.md) | [Bugfix] Disable w16a16 2of4 sparse CompressedTensors24 | 2025-01-24 |  | quantization |
| [#12339](../sources/prs/vllm/PR-12339.md) | [Build] Only build 9.0a for scaled_mm and sparse kernels | 2025-01-23 |  | gemm |
| [#12348](../sources/prs/vllm/PR-12348.md) | [ROCm] Faster Custom Paged Attention kernels | 2025-01-23 |  | attention |
| [#12303](../sources/prs/vllm/PR-12303.md) | [Hardware][Gaudi][Feature] Enable Dynamic MoE for Mixtral | 2025-01-22 | kernel-fusion | kernel-fusion, moe |
| [#12325](../sources/prs/vllm/PR-12325.md) | [Core] Optimizing cross-attention `QKVParallelLinear` computation | 2025-01-22 |  | attention |
| [#12282](../sources/prs/vllm/PR-12282.md) | [AMD][Quantization] Add TritonScaledMMLinearKernel since int8 is broken for AMD | 2025-01-21 |  | quantization |
| [#12185](../sources/prs/vllm/PR-12185.md) | [Kernel] add triton fused moe kernel for gptq/awq | 2025-01-18 | kernel-fusion | kernel-fusion, moe, quantization |
| [#12093](../sources/prs/vllm/PR-12093.md) | [Kernel] Flash Attention 3 Support | 2025-01-15 |  | attention |
| [#12097](../sources/prs/vllm/PR-12097.md) | Add: Support for Sparse24Bitmask Compressed Models | 2025-01-15 |  | fp8, quantization, tma |
| [#12049](../sources/prs/vllm/PR-12049.md) | [ROCm][MoE] moe tuning support for rocm | 2025-01-14 |  | moe |
| [#11868](../sources/prs/vllm/PR-11868.md) | [Kernel] Update `cutlass_scaled_mm` to support 2d group (blockwise) scaling | 2025-01-08 |  | fp8, gemm, quantization |
| [#11743](../sources/prs/vllm/PR-11743.md) | [Core] Support fully transparent sleep mode | 2025-01-05 | pipeline-stages | pipeline-stages |
| [#3](../sources/prs/vllm/PR-3.md) | Implement `single_query_cached_kv_attention` kernel | 2023-03-01 |  | attention |

