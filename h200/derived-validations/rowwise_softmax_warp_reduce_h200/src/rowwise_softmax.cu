#include "rowwise_softmax.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

namespace rowwise_softmax {
namespace {

// ── Common parameters ─────────────────────────────────────────────────────
constexpr int BLOCK_SIZE    = 256;   // threads per block (8 warps × 32)
constexpr int ROWS_PER_BLOCK = 4;    // each block processes 4 rows
constexpr int THREADS_PER_ROW = BLOCK_SIZE / ROWS_PER_BLOCK;  // 64
constexpr int WARPS_PER_ROW = THREADS_PER_ROW / 32;           // 2

// ── Helper: FP16 → FP32 load, coalesced ───────────────────────────────────
__device__ __forceinline__ float load_fp32(const __half* input, int row,
                                           int col, int N, int stride) {
    int idx = row * N + col;
    return __half2float(input[idx]);
}

// ── Helper: FP32 → FP16 store, coalesced ──────────────────────────────────
__device__ __forceinline__ void store_fp16(__half* output, int row, int col,
                                           int N, float val, int stride) {
    int idx = row * N + col;
    output[idx] = __float2half(val);
}

// ═══════════════════════════════════════════════════════════════════════════
// Kernel 0: Baseline — shared-memory tree reduction
// ═══════════════════════════════════════════════════════════════════════════

__global__ void softmax_baseline_kernel(const __half* __restrict__ input,
                                        __half* __restrict__ output,
                                        int M, int N) {
    // Each block handles ROWS_PER_BLOCK rows starting at base_row
    int base_row = blockIdx.x * ROWS_PER_BLOCK;
    int tid = threadIdx.x;
    int row_in_block = tid / THREADS_PER_ROW;  // 0,1,2,3
    int col_in_row   = tid % THREADS_PER_ROW;  // 0..63

    int global_row = base_row + row_in_block;
    if (global_row >= M) return;

    // ── Shared memory ──
    // We use two phases of shared memory:
    // Phase 1: partial max/sum values (64 floats → reduced to 1 per row)
    // Phase 2: exp values for normalization
    __shared__ float smem_partials[ROWS_PER_BLOCK][THREADS_PER_ROW];
    __shared__ float smem_row_max[ROWS_PER_BLOCK];
    __shared__ float smem_row_sum[ROWS_PER_BLOCK];

    // ── Step 1: Per-row max reduction ──
    float local_max = -INFINITY;
    for (int j = col_in_row; j < N; j += THREADS_PER_ROW) {
        float v = load_fp32(input, global_row, j, N, 1);
        local_max = fmaxf(local_max, v);
    }
    smem_partials[row_in_block][col_in_row] = local_max;
    __syncthreads();

    // Tree reduction for max (stride halving)
    for (int stride = THREADS_PER_ROW / 2; stride > 0; stride >>= 1) {
        if (col_in_row < stride) {
            smem_partials[row_in_block][col_in_row] =
                fmaxf(smem_partials[row_in_block][col_in_row],
                      smem_partials[row_in_block][col_in_row + stride]);
        }
        __syncthreads();
    }

    float row_max = smem_partials[row_in_block][0];
    if (col_in_row == 0) smem_row_max[row_in_block] = row_max;
    __syncthreads();
    row_max = smem_row_max[row_in_block];

    // ── Step 2: Exp and sum reduction ──
    float local_sum = 0.0f;
    for (int j = col_in_row; j < N; j += THREADS_PER_ROW) {
        float v = load_fp32(input, global_row, j, N, 1);
        float exp_val = expf(v - row_max);
        smem_partials[row_in_block][col_in_row] = exp_val;  // reuse smem for exp
        // We'll need exp values again for normalization, so store in smem after reduction
        local_sum += exp_val;
    }
    // Write partial sums for reduction
    // First, store exp values for later use, then reduce sum
    smem_partials[row_in_block][col_in_row] = local_sum;  // temporarily store partial sum
    __syncthreads();

    // Tree reduction for sum
    for (int stride = THREADS_PER_ROW / 2; stride > 0; stride >>= 1) {
        if (col_in_row < stride) {
            smem_partials[row_in_block][col_in_row] +=
                smem_partials[row_in_block][col_in_row + stride];
        }
        __syncthreads();
    }

    float row_sum = smem_partials[row_in_block][0];
    if (col_in_row == 0) smem_row_sum[row_in_block] = row_sum;
    __syncthreads();
    row_sum = smem_row_sum[row_in_block];

    // ── Step 3: Normalize and write ──
    float inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;
    for (int j = col_in_row; j < N; j += THREADS_PER_ROW) {
        float v = load_fp32(input, global_row, j, N, 1);
        float exp_val = expf(v - row_max);
        store_fp16(output, global_row, j, N, exp_val * inv_sum, 1);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Kernel 1: Candidate — hierarchical warp-shuffle reduction
// ═══════════════════════════════════════════════════════════════════════════

__global__ void softmax_warp_shuffle_kernel(const __half* __restrict__ input,
                                            __half* __restrict__ output,
                                            int M, int N) {
    int base_row = blockIdx.x * ROWS_PER_BLOCK;
    int tid = threadIdx.x;
    int row_in_block = tid / THREADS_PER_ROW;  // 0,1,2,3
    int col_in_row   = tid % THREADS_PER_ROW;  // 0..63

    int global_row = base_row + row_in_block;
    if (global_row >= M) return;

    int warp_id_in_block = tid / 32;            // 0..7
    int lane = tid % 32;                         // 0..31
    int warp_in_row = col_in_row / 32;          // 0 or 1 (since 64 threads per row = 2 warps)

    // ── Shared memory for cross-warp reduction only ──
    __shared__ float smem_warp_max[ROWS_PER_BLOCK][WARPS_PER_ROW];
    __shared__ float smem_warp_sum[ROWS_PER_BLOCK][WARPS_PER_ROW];
    __shared__ float smem_row_max[ROWS_PER_BLOCK];
    __shared__ float smem_row_sum[ROWS_PER_BLOCK];

    // ── Step 1: Per-warp max reduction ──
    float warp_max = -INFINITY;
    for (int j = col_in_row; j < N; j += THREADS_PER_ROW) {
        float v = load_fp32(input, global_row, j, N, 1);
        warp_max = fmaxf(warp_max, v);
    }

    // Butterfly warp reduction for max
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        warp_max = fmaxf(warp_max, __shfl_xor_sync(0xFFFFFFFF, warp_max, offset));
    }

    // Write warp partial max to shared memory
    if (lane == 0) {
        smem_warp_max[row_in_block][warp_in_row] = warp_max;
    }
    __syncthreads();

    // Cross-warp: combine the 2 warp partial maxes
    float row_max = smem_warp_max[row_in_block][0];
    if (WARPS_PER_ROW > 1) {
        row_max = fmaxf(row_max, smem_warp_max[row_in_block][1]);
    }
    if (col_in_row == 0) smem_row_max[row_in_block] = row_max;
    __syncthreads();
    row_max = smem_row_max[row_in_block];

    // ── Step 2: Per-warp exp and sum ──
    float warp_sum = 0.0f;
    for (int j = col_in_row; j < N; j += THREADS_PER_ROW) {
        float v = load_fp32(input, global_row, j, N, 1);
        warp_sum += expf(v - row_max);
    }

    // Butterfly warp reduction for sum
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        warp_sum += __shfl_xor_sync(0xFFFFFFFF, warp_sum, offset);
    }

    // Write warp partial sum to shared memory
    if (lane == 0) {
        smem_warp_sum[row_in_block][warp_in_row] = warp_sum;
    }
    __syncthreads();

    // Cross-warp: combine the 2 warp partial sums
    float row_sum = smem_warp_sum[row_in_block][0];
    if (WARPS_PER_ROW > 1) {
        row_sum += smem_warp_sum[row_in_block][1];
    }
    if (col_in_row == 0) smem_row_sum[row_in_block] = row_sum;
    __syncthreads();
    row_sum = smem_row_sum[row_in_block];

    // ── Step 3: Normalize and write ──
    float inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;
    for (int j = col_in_row; j < N; j += THREADS_PER_ROW) {
        float v = load_fp32(input, global_row, j, N, 1);
        float exp_val = expf(v - row_max);
        store_fp16(output, global_row, j, N, exp_val * inv_sum, 1);
    }
}

}  // namespace

// ── Public API ─────────────────────────────────────────────────────────────

cudaError_t softmax_baseline(const __half* input, __half* output,
                             int M, int N, cudaStream_t stream) {
    if (!input || !output || M <= 0 || N <= 0)
        return cudaErrorInvalidValue;

    int num_blocks = (M + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    softmax_baseline_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        input, output, M, N);
    return cudaPeekAtLastError();
}

cudaError_t softmax_warp_shuffle(const __half* input, __half* output,
                                 int M, int N, cudaStream_t stream) {
    if (!input || !output || M <= 0 || N <= 0)
        return cudaErrorInvalidValue;

    int num_blocks = (M + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    softmax_warp_shuffle_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        input, output, M, N);
    return cudaPeekAtLastError();
}

}  // namespace rowwise_softmax
