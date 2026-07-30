/**
 * softmax.cu — H200 FP16 Row-wise Softmax
 *   Baseline:  Shared-memory tree reduction
 *   Candidate: Hierarchical warp-shuffle + cross-warp shared-memory reduction
 *
 * Compile SO:  nvcc -arch=sm_90a -O3 -std=c++17 -Xcompiler -fPIC -shared softmax.cu -o softmax.so
 * Compile EXE: nvcc -arch=sm_90a -O3 -std=c++17 -DSOFTMAX_BENCH softmax.cu -o softmax_bench
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <stdint.h>

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
#define THREADS_PER_BLOCK 256
#define ROWS_PER_BLOCK    4
#define THREADS_PER_ROW   64      // THREADS_PER_BLOCK / ROWS_PER_BLOCK
#define WARPS_PER_ROW     2       // THREADS_PER_ROW / 32
#define WARP_SIZE         32

// ---------------------------------------------------------------------------
// Error-checking macros
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call) do {                                          \
    cudaError_t _e = (call);                                           \
    if (_e != cudaSuccess) {                                           \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,  \
                cudaGetErrorString(_e));                               \
        exit(1);                                                       \
    }                                                                  \
} while (0)

// ========================================================================
// KERNEL 1: Baseline — Pure Shared-Memory Tree Reduction
// ========================================================================
//
// Each block handles ROWS_PER_BLOCK rows.
// Within a block, THREADS_PER_ROW threads collaborate on one row.
// Reduction uses shared-memory halving (log2(THREADS_PER_ROW) = 6 rounds
// per reduction, with __syncthreads() after each round).
//
__global__ void softmax_baseline_kernel(
    const __half* __restrict__ input,
    __half* __restrict__ output,
    int rows,
    int cols)
{
    // Shared memory: [ROWS_PER_BLOCK][THREADS_PER_ROW] floats
    // Reused for max then sum reduction.
    __shared__ float smem[ROWS_PER_BLOCK * THREADS_PER_ROW];

    int tid          = threadIdx.x;
    int row_in_block = tid / THREADS_PER_ROW;   // 0..3
    int tid_in_row   = tid % THREADS_PER_ROW;   // 0..63
    int row          = blockIdx.x * ROWS_PER_BLOCK + row_in_block;

    bool valid_row = row < rows;

    const __half* in_row  = input  + row * cols;
    __half*       out_row = output + row * cols;

    float* smem_row = smem + row_in_block * THREADS_PER_ROW;

    // --- Phase 1: Find row maximum via shared-memory tree reduction ---
    float local_max = -INFINITY;
    if (valid_row) {
        if (valid_row) for (int j = tid_in_row; j < cols; j += THREADS_PER_ROW) {
            float v = __half2float(in_row[j]);
            local_max = fmaxf(local_max, v);
        }
    }

    smem_row[tid_in_row] = local_max;
    __syncthreads();

    // Tree reduction: log2(64) = 6 rounds
    #pragma unroll
    for (int stride = THREADS_PER_ROW / 2; stride > 0; stride >>= 1) {
        if (tid_in_row < stride) {
            smem_row[tid_in_row] = fmaxf(smem_row[tid_in_row],
                                         smem_row[tid_in_row + stride]);
        }
        __syncthreads();
    }

    float row_max = smem_row[0];
    __syncthreads();  // ensure all threads read before overwrite

    // --- Phase 2: Compute exp sum via shared-memory tree reduction ---
    float local_sum = 0.0f;
    if (valid_row) {
        for (int j = tid_in_row; j < cols; j += THREADS_PER_ROW) {
            float v = __half2float(in_row[j]);
            local_sum += expf(v - row_max);
        }
    }

    smem_row[tid_in_row] = local_sum;
    __syncthreads();

    #pragma unroll
    for (int stride = THREADS_PER_ROW / 2; stride > 0; stride >>= 1) {
        if (tid_in_row < stride) {
            smem_row[tid_in_row] += smem_row[tid_in_row + stride];
        }
        __syncthreads();
    }

    float row_sum = smem_row[0];
    float inv_sum = 1.0f / (row_sum + 1e-12f);
    __syncthreads();

    // --- Phase 3: Compute normalized output ---
    if (valid_row) {
        for (int j = tid_in_row; j < cols; j += THREADS_PER_ROW) {
            float v = __half2float(in_row[j]);
            out_row[j] = __float2half(expf(v - row_max) * inv_sum);
        }
    }
}

// ========================================================================
// KERNEL 2: Candidate — Hierarchical Warp-Shuffle Reduction
// ========================================================================
//
// Differs from baseline ONLY in the reduction strategy:
//   - Intra-warp: __shfl_xor_sync butterfly reduction (5 steps, no sync)
//   - Cross-warp:  shared-memory reduction of WARPS_PER_ROW partials per row
//
// Same block size, rows-per-block, FP32 accumulation, expf(), and
// global-memory access pattern as the baseline.
//
__global__ void softmax_warp_shuffle_kernel(
    const __half* __restrict__ input,
    __half* __restrict__ output,
    int rows,
    int cols)
{
    // Shared memory: [ROWS_PER_BLOCK][WARPS_PER_ROW] floats
    // — only 2 floats per row instead of 64.
    __shared__ float smem[ROWS_PER_BLOCK * WARPS_PER_ROW];

    int tid          = threadIdx.x;
    int row_in_block = tid / THREADS_PER_ROW;   // 0..3
    int tid_in_row   = tid % THREADS_PER_ROW;   // 0..63
    int warp_in_row  = tid_in_row / WARP_SIZE;  // 0 or 1
    int lane_id      = tid_in_row % WARP_SIZE;  // 0..31
    int row          = blockIdx.x * ROWS_PER_BLOCK + row_in_block;

    bool valid_row = row < rows;

    const __half* in_row  = input  + row * cols;
    __half*       out_row = output + row * cols;

    float* smem_row = smem + row_in_block * WARPS_PER_ROW;

    unsigned mask = 0xFFFFFFFF;

    // --- Phase 1: Find row maximum via warp-shuffle reduction ---
    float local_max = -INFINITY;
    if (valid_row) {
        for (int j = tid_in_row; j < cols; j += THREADS_PER_ROW) {
            float v = __half2float(in_row[j]);
            local_max = fmaxf(local_max, v);
        }
    }

    // Butterfly shuffle reduction within the warp (log2(32) = 5 steps)
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        local_max = fmaxf(local_max, __shfl_xor_sync(mask, local_max, offset));
    }

    // Cross-warp reduction: each warp's lane 0 writes its partial max
    if (lane_id == 0) {
        smem_row[warp_in_row] = local_max;
    }
    __syncthreads();

    float row_max = fmaxf(smem_row[0], smem_row[1]);
    __syncthreads();  // ensure all threads read before overwrite

    // --- Phase 2: Compute exp sum via warp-shuffle reduction ---
    float local_sum = 0.0f;
    if (valid_row) {
        for (int j = tid_in_row; j < cols; j += THREADS_PER_ROW) {
            float v = __half2float(in_row[j]);
            local_sum += expf(v - row_max);
        }
    }

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        local_sum += __shfl_xor_sync(mask, local_sum, offset);
    }

    if (lane_id == 0) {
        smem_row[warp_in_row] = local_sum;
    }
    __syncthreads();

    float row_sum  = smem_row[0] + smem_row[1];
    float inv_sum  = 1.0f / (row_sum + 1e-12f);
    __syncthreads();

    // --- Phase 3: Compute normalized output ---
    if (valid_row) {
        for (int j = tid_in_row; j < cols; j += THREADS_PER_ROW) {
            float v = __half2float(in_row[j]);
            out_row[j] = __float2half(expf(v - row_max) * inv_sum);
        }
    }
}

// ========================================================================
// Host-side launch wrappers (C-linkage for Python ctypes)
// ========================================================================

extern "C" {

void launch_softmax_baseline(const __half* d_input, __half* d_output,
                              int rows, int cols, cudaStream_t stream)
{
    int blocks = (rows + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    softmax_baseline_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
        d_input, d_output, rows, cols);
}

void launch_softmax_warp_shuffle(const __half* d_input, __half* d_output,
                                  int rows, int cols, cudaStream_t stream)
{
    int blocks = (rows + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    softmax_warp_shuffle_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
        d_input, d_output, rows, cols);
}

cudaError_t copy_host_to_device(const __half* h_src, __half* d_dst, size_t count) {
    return cudaMemcpy(d_dst, h_src, count * sizeof(__half), cudaMemcpyHostToDevice);
}

cudaError_t copy_device_to_host(const __half* d_src, __half* h_dst, size_t count) {
    return cudaMemcpy(h_dst, d_src, count * sizeof(__half), cudaMemcpyDeviceToHost);
}

cudaError_t allocate_device(__half** d_ptr, size_t count) {
    return cudaMalloc(d_ptr, count * sizeof(__half));
}

cudaError_t free_device(__half* d_ptr) {
    return cudaFree(d_ptr);
}

} // extern "C"

// ========================================================================
// Benchmark harness (compiled as standalone executable with -DSOFTMAX_BENCH)
// ========================================================================
#ifdef SOFTMAX_BENCH

#include <vector>
#include <algorithm>
#include <string>

struct BenchResult {
    const char* kernel_name;
    int rows;
    int cols;
    double median_ms;
    double min_ms;
    double max_ms;
};

static std::vector<double> run_timing_group(
    void (*launch_fn)(const __half*, __half*, int, int, cudaStream_t),
    const __half* d_input, __half* d_output,
    int rows, int cols, int warmups, int iters)
{
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    for (int i = 0; i < warmups; i++) {
        launch_fn(d_input, d_output, rows, cols, stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<double> times;
    times.reserve(iters);
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < iters; i++) {
        CUDA_CHECK(cudaEventRecord(start, stream));
        launch_fn(d_input, d_output, rows, cols, stream);
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times.push_back((double)ms);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamDestroy(stream));
    return times;
}

static BenchResult benchmark_kernel(
    const char* name,
    void (*launch_fn)(const __half*, __half*, int, int, cudaStream_t),
    const __half* d_input, __half* d_output,
    int rows, int cols, int warmups, int iters, int groups)
{
    std::vector<double> all_times;
    for (int g = 0; g < groups; g++) {
        auto gt = run_timing_group(launch_fn, d_input, d_output,
                                   rows, cols, warmups, iters);
        all_times.insert(all_times.end(), gt.begin(), gt.end());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    std::sort(all_times.begin(), all_times.end());
    BenchResult r;
    r.kernel_name = name;
    r.rows = rows;
    r.cols = cols;
    r.median_ms = all_times[all_times.size() / 2];
    r.min_ms   = all_times.front();
    r.max_ms   = all_times.back();
    return r;
}

static double effective_bandwidth_gbs(int rows, int cols, double median_ms) {
    double bytes = 2.0 * (double)rows * (double)cols * sizeof(__half);
    return bytes / (median_ms * 1e6);
}

int main(int argc, char** argv) {
    int warmups = 25;
    int iters   = 120;
    int groups  = 3;

    typedef std::pair<int,int> Shape;
    std::vector<Shape> shapes;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--warmups") == 0 && i+1 < argc) {
            warmups = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--iters") == 0 && i+1 < argc) {
            iters = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--groups") == 0 && i+1 < argc) {
            groups = atoi(argv[++i]);
        } else if (i+1 < argc) {
            shapes.push_back(Shape(atoi(argv[i]), atoi(argv[i+1])));
            i++;
        }
    }

    if (shapes.empty()) {
        shapes.push_back(Shape(512, 512));
        shapes.push_back(Shape(1024, 1024));
        shapes.push_back(Shape(2048, 2048));
        shapes.push_back(Shape(4096, 4096));
        shapes.push_back(Shape(8192, 8192));
        shapes.push_back(Shape(512, 777));
        shapes.push_back(Shape(768, 3072));
        shapes.push_back(Shape(1000, 513));
    }

    int max_rows = 0, max_cols = 0;
    for (size_t i = 0; i < shapes.size(); i++) {
        if (shapes[i].first  > max_rows) max_rows = shapes[i].first;
        if (shapes[i].second > max_cols) max_cols = shapes[i].second;
    }

    size_t max_elems = (size_t)max_rows * (size_t)max_cols;
    __half *h_input = nullptr, *h_output = nullptr;
    __half *d_input = nullptr, *d_output = nullptr;

    CUDA_CHECK(cudaMallocHost(&h_input,  max_elems * sizeof(__half)));
    CUDA_CHECK(cudaMallocHost(&h_output, max_elems * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_input,  max_elems * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_output, max_elems * sizeof(__half)));

    srand(42);
    for (size_t i = 0; i < max_elems; i++) {
        float r = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
        h_input[i] = __float2half(r);
    }

    printf("[\n");
    bool first = true;

    for (size_t si = 0; si < shapes.size(); si++) {
        int rows = shapes[si].first, cols = shapes[si].second;
        CUDA_CHECK(cudaMemcpy(d_input, h_input,
                              (size_t)rows * cols * sizeof(__half),
                              cudaMemcpyHostToDevice));

        BenchResult br = benchmark_kernel(
            "baseline", launch_softmax_baseline,
            d_input, d_output, rows, cols, warmups, iters, groups);

        BenchResult wr = benchmark_kernel(
            "c001_warp_shuffle", launch_softmax_warp_shuffle,
            d_input, d_output, rows, cols, warmups, iters, groups);

        double bw_bl = effective_bandwidth_gbs(rows, cols, br.median_ms);
        double bw_ws = effective_bandwidth_gbs(rows, cols, wr.median_ms);
        double speedup = br.median_ms / wr.median_ms;

        if (!first) printf(",\n");
        first = false;

        printf("  {\n");
        printf("    \"shape\": [%d, %d],\n", rows, cols);
        printf("    \"baseline\": {\"median_ms\": %.6f, \"min_ms\": %.6f, \"max_ms\": %.6f, \"bandwidth_gbs\": %.3f},\n",
               br.median_ms, br.min_ms, br.max_ms, bw_bl);
        printf("    \"c001_warp_shuffle\": {\"median_ms\": %.6f, \"min_ms\": %.6f, \"max_ms\": %.6f, \"bandwidth_gbs\": %.3f},\n",
               wr.median_ms, wr.min_ms, wr.max_ms, bw_ws);
        printf("    \"candidate_to_baseline_speedup\": %.4f\n", speedup);
        printf("  }");
    }
    printf("\n]\n");

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFreeHost(h_input));
    CUDA_CHECK(cudaFreeHost(h_output));
    return 0;
}

#endif // SOFTMAX_BENCH
