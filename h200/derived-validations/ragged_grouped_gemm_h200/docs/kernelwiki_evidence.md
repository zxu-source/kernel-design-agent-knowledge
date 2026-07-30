# KernelWiki evidence: BF16 Ragged Grouped GEMM on H200

## Validation and retrieval record

On 2026-07-17, from `../skills/KernelWiki/`:

```text
python3 scripts/validate.py
Validated 2271 files (2222 source IDs collected)
Validated 365 asset bundles (verbatim=340, extracted=13, derived=12, orphan-source-files=0)
Validated 6 candidate ledgers
All files valid.
```

The directory inventory at validation time was 49 synthesis pages under `wiki/`,
4,439 files under `sources/`, and 365 `PROVENANCE.yaml` asset bundles.  The
validator's 2,271 count is the full Markdown corpus, not merely `wiki/` pages.

Required retrievals were executed with `--limit 10 --compact`:

1. `Hopper grouped GEMM variable M expert CUTLASS`
2. `FlashInfer fused MoE router GEMM grouped GEMM`
3. `Hopper SM90 TMA WGMMA persistent grouped GEMM`
4. `vLLM SGLang MoE expert grouped GEMM BF16`
5. `DeepGEMM grouped GEMM variable M scheduling`

Relevant pages were opened with `scripts/get_page.py`; the provenance files
below were read.  Key bundled files/diffs were inspected with `rg`, not copied.

## Knowledge-to-design mapping

| Source page / PR | Upstream repository | Observed technical fact | Adopt? | Concrete application here |
| --- | --- | --- | --- | --- |
| `kernel-grouped-gemm` | KernelWiki synthesis; sources include DeepGEMM/CUTLASS | Grouped MoE GEMM fixes K/N while M varies; contiguous layout packs A/C and uses cumulative offsets; small M and load imbalance are caveats. | Yes | Use packed A/C plus `a_offsets`/`c_offsets`, skip zero-M groups, and account useful FLOPs from real M only. |
| `pr-cutlass-3091`, `artifacts/prs/cutlass/PR-3091/PROVENANCE.yaml` | NVIDIA/cutlass | Hopper CuTeDSL grouped GEMM extends dense persistent GEMM with a group-aware static persistent scheduler, per-group TMA maps, DMA/MMA warp groups; its stated grouped constraints include BF16/FP16, FP32 accumulator, and 16-byte contiguous alignment. | Partly | Adopt the *scheduling concept*: a flattened tile list and bounded persistent grid. Do **not** copy CuTeDSL/TMA code; first implementation uses CUDA C++ and guarded tails. TMA is conditional after alignment/CUTLASS checks. |
| `pr-flashinfer-1241`, `artifacts/prs/flashinfer/PR-1241/PROVENANCE.yaml` | flashinfer-ai/flashinfer | A preparatory device kernel derives `m = m_indptr[i+1]-m_indptr[i]`, group problem sizes, packed strides, and per-group A/B/D pointers; its grouped launcher allocates metadata in an aligned workspace. | Yes | Create explicit, lifetime-owned device metadata: counts/offsets, tile map, total-tile count, and pointer arithmetic validated before launch. Metadata-build timing will be reported separately. |
| `pr-sglang-9199`, `artifacts/prs/sglang/PR-9199/PROVENANCE.yaml` | sgl-project/sglang | The integration calls a `grouped_gemm_nt_masked` MoE path using expert token metadata; it makes a masked static-shape route available. | No for main path | This task targets packed ragged input. We use this as a counterexample: masked/padded layout is excluded from useful-FLOP accounting and may be evaluated only as a reference, since it wastes work for skewed M. |
| `pr-vllm-25990`, `artifacts/prs/vllm/PR-25990/PROVENANCE.yaml` | vllm-project/vllm | MoE integration passes `expert_num_tokens` as `masked_m` and separately manages reusable workspaces for independent GEMMs. | Partly | Preserve metadata/workspace ownership and stream propagation in the interface. We do not adopt the two-GEMM workspace layout because the target is one grouped GEMM. |
| `pr-deepgemm-304`, `artifacts/prs/deepgemm/PR-304/PROVENANCE.yaml` | deepseek-ai/DeepGEMM | SM90 BF16 code exposes contiguous and masked M-grouped modes; config selection controls tile layout, swizzles, TMA threads, math threads, and pipeline stages. | Partly | Add a shape-dispatch boundary and retain separate contiguous-ragged vs masked semantics in tests. We will not assert a DeepGEMM configuration is optimal without H200 results. |

## Fact vs inference vs new design

### Direct observations from the knowledge base

- The cited pages and their provenance identify actual upstream repositories,
  merged PRs, SHAs, paths, and asset modes.  All five PR bundles above were
  marked `verbatim`; DeepGEMM's diff additionally records `size_cap_truncated:
  true` while its key files are present.
- CUTLASS PR 3091's diff explicitly describes a `StaticPersistentGroupTileScheduler`,
  per-group TMA map updates, and WGMMA/TMA warp specialization.
- FlashInfer PR 1241's bundled header constructs per-group `ProblemShape(m,n,k)`,
  packed strides, and pointer arrays from `m_indptr`.
- DeepGEMM PR 304's bundled SM90 BF16 code has contiguous M-grouped and masked
  M-grouped entrypoints and passes `num_groups`/grouped layout to launch code.

### Engineering inferences for H200 BF16

- Flattened tiles should improve skewed-load utilization relative to a launch per
  expert, but the atomic scheduler can cost more than it saves for small uniform
  workloads.  This must be tested rather than assumed.
- Host-built tile maps avoid an in-kernel prefix-sum/binary-search cost; GPU-built
  metadata is a later alternative if host preparation becomes relevant.
- A 128x64 (or nearby) MMA tile is a starting point, not a cited optimum.  It
  must be benchmarked with tail masking and occupancy evidence.

### New task-specific design

- `candidate_01_kernelwiki` will use one CUDA launch, a device tile queue, and
  an atomic persistent dispatcher over `(expert, tile_m, tile_n)` work entries.
- The API accepts packed offsets directly, validates them in the test harness,
  and defines `M_e=0` as producing no work and no dereference.
- Results will report metadata preparation separately and useful rather than
  padded FLOPs.  No upstream kernel source will be copied into this project.

## Transfer limits

Several cited upstream implementations target SM100 or quantized FP8/FP4/NVFP4,
whereas this task is SM90 H200 BF16.  They substantiate data layout, grouped
metadata, scheduler, and test-design ideas; they do **not** validate direct
reuse of SM100 CLC or quantization-specific code.  TMA/WGMMA claims require
H200 build and measurement evidence before being treated as implemented.
