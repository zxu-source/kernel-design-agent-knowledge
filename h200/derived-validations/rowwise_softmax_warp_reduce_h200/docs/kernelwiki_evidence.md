# KernelWiki Evidence: Warp-Shuffle Reduction for Row-wise Softmax

**Query date**: 2026-07-17
**Skill**: KernelWiki (knowledge cutoff: 2026-04-27)
**Queries executed**:
1. `python3 scripts/query.py "warp shuffle reduction CUDA softmax row-wise FP16" --compact`
2. `python3 scripts/grep_wiki.py "warp.shuffle|shfl.down|__shfl_xor|warp.reduce" --any`
3. `python3 scripts/query.py "hierarchical reduction shared memory warp shuffle Hopper" --compact`

---

## Primary Evidence: Warp-Shuffle Reduction in Softmax

### Source: `wiki/techniques/software-exp.md` (id: `technique-software-exp`)

**Nature**: This is an **example inside a page** primarily about software-emulated exponential on Blackwell (SM100). The warp-shuffle reduction patterns appear as supporting code within the softmax implementation, not as the page's main topic.

**Confidence**: `source-reported` | **Reproducibility**: `snippet` | **Architectures**: `sm100`

**Relevant code pattern — Warp-level max reduction** (lines ~116-118):

```cuda
// Step 1: Find max across the row (warp reduction)
float local_max = -INFINITY;
for (int j = lane; j < N; j += 32) {
    local_max = fmaxf(local_max, scores[j]);
}
// Warp-level max reduction
for (int offset = 16; offset > 0; offset >>= 1) {
    local_max = fmaxf(local_max, __shfl_xor_sync(0xFFFFFFFF, local_max, offset));
}
float m_new = local_max;
```

**Relevant code pattern — Warp-level sum reduction** (lines ~133-135):

```cuda
// Warp-level sum reduction
for (int offset = 16; offset > 0; offset >>= 1) {
    local_sum += __shfl_xor_sync(0xFFFFFFFF, local_sum, offset);
}
```

**How this supports the task**: These patterns demonstrate the canonical warp-shuffle butterfly reduction for row-wise softmax: (1) per-thread partial max/sum accumulation across the row via strided access (lane, lane+32, lane+64, ...), (2) warp-level butterfly reduction using `__shfl_xor_sync` to produce a single lane-0 result, (3) broadcast of the reduced value to all lanes for the subsequent normalization step.

**Limitation for H200 (SM90)**: The page explicitly states software exponential is "Not recommended on Hopper" — the SFU-to-MMA ratio is balanced on SM90. Both our baseline and candidate will use hardware `__expf()` / `__hfma()` since H200 is SM90. The warp-shuffle reduction pattern itself is architecture-agnostic.

**Sources cited by this page**: `blog-flash-attention-4`, `doc-flash-attention-4`, `doc-ptx-isa-sm100`

---

## Secondary Evidence: Hierarchical (Warp + Shared Memory) Reduction

### Source: `wiki/techniques/vectorized-loads.md` (id: `technique-vectorized-loads`)

**Confidence**: `source-reported` | **Reproducibility**: `snippet` | **Architectures**: `sm100, sm90`

**Relevant code pattern — Two-phase reduction** (lines ~259-261):

```cuda
// Phase 1: warp-level reduction (32 threads → 1 partial sum per warp)
for (int offset = 16; offset > 0; offset >>= 1) {
    acc += __shfl_xor_sync(0xFFFFFFFF, acc, offset);
}

// Phase 2: shared memory reduction across the 2 warps assigned to this row
__shared__ float smem_reduce[BLOCK_M * 2];  // 2 warp partials per row
int warp_in_row = thread_in_row / 32;
int lane = thread_in_row % 32;
if (lane == 0) {
    smem_reduce[local_row * 2 + warp_in_row] = acc;
}
__syncthreads();

if (thread_in_row == 0) {
    float result = smem_reduce[local_row * 2] + smem_reduce[local_row * 2 + 1];
    C[row] = __float2half(result * global_scale_a * global_scale_b);
}
```

**How this supports the task**: This demonstrates the hierarchical approach: warp-shuffle reduction within each warp producing partial sums, followed by shared-memory reduction across warps. This is the exact pattern for our candidate kernel. The baseline will use pure shared-memory tree reduction; the candidate will replace the intra-warp shared-memory phase with `__shfl_xor_sync`.

---

## Tertiary Evidence: Warp Reduce in Production Kernels

### Source: `sources/prs/sglang/PR-8130.md` (id: `pr-sglang-8130`)

**Title**: "[sgl-kernel] Opt per_token_quant_fp8 with warp reduce"
**Author**: yuan-luo | **Repo**: sglang | **Status**: merged

**Summary**: Optimized per_token_quant_fp8 kernel with warp reduce, obtaining 5-7% speedup in large batch sizes.

**How this supports the task**: Real-world evidence that replacing shared-memory reduction with warp-shuffle reduction yields measurable performance improvements (5-7%) in production kernel code. The per-token quantization use case (row-wise reduction over a K dimension) is structurally similar to softmax's row-wise max/sum reduction.

---

### Source: `sources/blogs/amandeep-nvfp4-attempts.md` (id: `blog-amandeep-nvfp4`)

**Context**: GPU Mode NVFP4 Hackathon — 12 attempts at optimizing batched GEMV on B200.

**Relevant note** (line ~150): "Warp shuffle reduction: Expected to help for the K-dimension reduction, but the overhead of shuffle instructions exceeded the benefit over simple register accumulation at these K sizes."

**How this supports the task**: This provides a cautionary data point: warp-shuffle reduction may not always beat simpler approaches for very small K dimensions. This informs our benchmark shape selection — we should see whether the candidate outperforms the baseline consistently across all 8 shapes, or only at larger column counts.

---

## Summary of Evidence Quality

| Source | Type | Confidence | Relevance to Task |
|--------|------|-----------|-------------------|
| `technique-software-exp` | Wiki technique page (warp reduction as example within) | `source-reported` | **Primary** — exact `__shfl_xor_sync` max/sum patterns for softmax |
| `technique-vectorized-loads` | Wiki technique page | `source-reported` | **Secondary** — hierarchical warp+smem reduction architecture |
| `pr-sglang-8130` | Merged PR | `source-reported` | **Tertiary** — real-world 5-7% speedup from warp reduce |
| `blog-amandeep-nvfp4` | Blog | `source-reported` | **Tertiary** — caveat for small K dimensions |

**Verdict**: KernelWiki evidence supports warp-shuffle reduction as a valid, production-tested optimization pattern for row-wise reductions in CUDA kernels. The `__shfl_xor_sync` butterfly reduction is the canonical approach demonstrated in FlashAttention-4's softmax implementation and in SGLang's per-token quantization kernel. The evidence is from `source-reported` pages with `snippet` reproducibility.
