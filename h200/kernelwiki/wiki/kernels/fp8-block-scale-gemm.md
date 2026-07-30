---
id: kernel-fp8-block-scale-gemm
title: "FP8 Block-Scale GEMM"
type: kernel
architectures: [sm100, sm90]
tags: [gemm, fp8, block-scale, fine-grained-quantization, tcgen05, wgmma]
confidence: source-reported
reproducibility: snippet
kernel_types: [gemm]
languages: [cuda-cpp, cute-dsl]
related: [kernel-deepgemm, kernel-nvfp4-gemm, technique-fine-grained-quantization, hw-tcgen05-mma]
sources: [blog-deepgemm, doc-cutlass-blackwell, doc-cutlass-changelog-sm100]
performance_claims:
  - gpu: H800
    dtype: fp8
    shape: "M=4096, N=4096, K=4096"
    metric: TFLOPS
    value: 1550
    utilization: "~90% via CUDA core promotion"
    source_id: blog-deepgemm
blackwell_relevance: "SM100 tcgen05.mma has native UE8M0 block scaling; SM90 requires Nc=128 CUDA core promotion. Same kernel pattern works on both but different scale handling."
h200_validation:
  date: '2026-07-20 (overnight)'
  evidence_file: data/crawl-runs/h200/hopper-fp8-matmul-h200-results.md
  harness_dir: artifacts/kernels/hopper-fp8-matmul/variants
  gpu: H200
  arch: sm90
  toolchain: "Triton 3.6.0, PyTorch 2.11.0+cu130"
  correctness: "PASS — naive Triton fp8 e4m3 matmul matches dequantized fp32 ref within fp8 quantization floor (rel err 1-3%)"
  result: "kernel-quality limited — naive tl.dot fp8 attained only 156-187 TFLOPS (~2x SLOWER than bf16); a production fp8 kernel (DeepGEMM/CUTLASS) is required to realize fp8's ~2x advantage. Correctness validated; throughput NOT representative."
  scope: "Unscaled fp8 e4m3 via naive Triton tl.dot on H200; does not exercise block scales or the fast fp8 wgmma path."
---

# FP8 Block-Scale GEMM

## Overview

FP8 GEMM with fine-grained block scales (128x128 weights, 1x128 activations). Preserves more dynamic range than per-tensor FP8 scaling, critical for LLM inference and training where outliers dominate quantization error.

DeepGEMM is the reference implementation; CUTLASS provides SM100 schedules.

## H200 measured evidence (Triton, unscaled fp8 e4m3)

A naive Triton `tl.dot` fp8 e4m3 matmul (fp32 accumulation, no block scales) was
run on H200 to validate fp8 correctness and probe naive-kernel throughput:

- **Correctness PASS**: output matches a dequantized fp32 reference within the
  fp8 e4m3 quantization floor (relative error 1-3% across shapes — expected, as
  e4m3 has ~3 mantissa bits).
- **Throughput is kernel-quality limited, NOT representative of fp8 hardware**:
  the naive fp8 kernel attained only **156-187 TFLOPS** (~8-9% of the H200's
  ~1979 TFLOPS fp8 peak) and was **~2x SLOWER than the equally-naive bf16
  kernel** (330-501 TFLOPS) — the opposite of fp8's theoretical ~2x advantage.
  Enlarging the K-tile (BK=256) hit shared-memory out-of-resource on H200.

**Takeaway:** naive Triton fp8 `tl.dot` does not hit the fast fp8 wgmma path;
realizing fp8's throughput requires a production fp8 kernel (tuned K-tile /
operand layouts for the fp8 wgmma atom, plus per-block scale factors) such as
DeepGEMM (the reference impl) or CUTLASS fp8 GEMM. The H200 result validates fp8
*correctness* but documents that the naive path underutilizes the fp8 units.
Full numbers:
[`data/crawl-runs/h200/hopper-fp8-matmul-h200-results.md`](../../data/crawl-runs/h200/hopper-fp8-matmul-h200-results.md).

## Block Scaling Structure

```
Activations: tile-wise 1x128 scales
  [1x128 values] → 1 scale factor (FP32 or FP8 E4M3)

Weights: block-wise 128x128 scales
  [128x128 values] → 1 scale factor per block

Output accumulator: FP32
  Multiply A × B in FP8, accumulate in FP32, apply scales at MMA boundary
```

## SM90 Path (Hopper, WGMMA)

```cuda
// Hopper accumulator has ~22-bit precision (FP22)
// Every Nc=128 WGMMAs, promote partial sum to FP32 on CUDA cores
// This retains precision without adding MMA overhead

__device__ void sm90_fp8_gemm_with_promotion(...) {
    float acc_fp32 = 0.0f;

    for (int k = 0; k < K; k += 128) {
        float acc_fp22 = 0.0f;
        #pragma unroll 4
        for (int kk = 0; kk < 128; kk += 32) {
            wgmma_fp8_e4m3(acc_fp22, A_frag, B_frag);
        }
        // Promote to FP32 accumulator on CUDA cores
        acc_fp32 += acc_fp22 * scale_a * scale_b;
    }

    // Write acc_fp32 to output
}
```

## SM100 Path (Blackwell, tcgen05.mma)

```cuda
// tcgen05.mma has native UE8M0 block scale support in hardware
// No promotion needed - scales applied inside MMA

__global__ void sm100_fp8_gemm_block_scale(...) {
    uint32_t tmem = tmem_alloc(256);

    for (int k = 0; k < K; k += BLOCK_K) {
        mbarrier_wait(&tma_done);

        // tcgen05.mma.mxf8f6f4.block_scale variant
        // Reads A, B from SMEM; scales from scale SMEM; accumulates in TMEM
        asm volatile(
            "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X "
            "[%0], %1, %2, %3, %4, %5;"
            :: "r"(tmem), "l"(a_desc), "l"(b_desc),
               "r"(sf_a_desc), "r"(sf_b_desc), "n"(1)
        );
    }

    // Read from TMEM, apply global scale, store
    float result = tmem_load(tmem) * global_scale;
    output[row * N + col] = __float2half(result);
}
```

## Memory Layout

```
A (activations) [M, K] packed FP8 E4M3
sf_a            [M, K/128] FP32 or FP8 E4M3 scales  # 1 per 1x128 tile

B (weights)     [N, K] packed FP8 E4M3
sf_b            [N/128, K/128] scales               # 1 per 128x128 block
                # or packed UE8M0 format (Blackwell): 4 scales per int32
```

## Performance

- DeepGEMM on H800: up to 1550 TFLOPS FP8
- CUTLASS SM100 schedules: similar ratio vs peak

## When To Use

- LLM inference with FP8 quantized weights (DeepSeek V3, Qwen2-FP8, etc.)
- Training with FP8 activations (DeepSeek V3 training framework)
- Anywhere per-tensor FP8 accuracy is insufficient due to outliers
