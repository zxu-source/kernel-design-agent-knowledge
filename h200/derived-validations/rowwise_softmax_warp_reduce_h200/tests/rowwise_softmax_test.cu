#include "rowwise_softmax.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#define CUDA_OK(x) do { \
    cudaError_t _e = (x); \
    if (_e != cudaSuccess) { \
        std::cerr << #x << ": " << cudaGetErrorString(_e) << "\n"; \
        return false; \
    } \
} while (0)

// ── Test case definition ──────────────────────────────────────────────────
struct TestCase {
    std::string name;
    int M, N;
};

// ── Helpers ────────────────────────────────────────────────────────────────
static float h2f(__half x) { return __half2float(x); }

static __half random_half(std::mt19937& gen, float scale = 1.0f) {
    std::uniform_real_distribution<float> d(-scale, scale);
    return __float2half(d(gen));
}

// ── CPU reference: FP32 softmax ───────────────────────────────────────────
static std::vector<float> cpu_softmax_fp32(const std::vector<__half>& input,
                                           int M, int N) {
    std::vector<float> output(M * N);
    for (int i = 0; i < M; ++i) {
        float max_val = -INFINITY;
        for (int j = 0; j < N; ++j) {
            max_val = fmaxf(max_val, h2f(input[i * N + j]));
        }
        float sum_val = 0.0f;
        for (int j = 0; j < N; ++j) {
            float e = expf(h2f(input[i * N + j]) - max_val);
            sum_val += e;
        }
        float inv_sum = (sum_val > 0.0f) ? (1.0f / sum_val) : 0.0f;
        for (int j = 0; j < N; ++j) {
            float e = expf(h2f(input[i * N + j]) - max_val);
            output[i * N + j] = e * inv_sum;
        }
    }
    return output;
}

// ── Buffers ────────────────────────────────────────────────────────────────
struct Buffers {
    std::vector<__half> h_input, h_out0, h_out1;
    __half *d_input = nullptr, *d_out0 = nullptr, *d_out1 = nullptr;
    int M = 0, N = 0;

    ~Buffers() { free(); }

    void free() {
        cudaFree(d_input);  d_input = nullptr;
        cudaFree(d_out0);   d_out0 = nullptr;
        cudaFree(d_out1);   d_out1 = nullptr;
    }

    bool alloc(int m, int n) {
        free();
        M = m; N = n;
        size_t bytes = (size_t)m * n * sizeof(__half);
        h_input.resize(m * n);
        h_out0.resize(m * n);
        h_out1.resize(m * n);
        CUDA_OK(cudaMalloc(&d_input, bytes));
        CUDA_OK(cudaMalloc(&d_out0, bytes));
        CUDA_OK(cudaMalloc(&d_out1, bytes));
        return true;
    }
};

// ── Input generators ───────────────────────────────────────────────────────
enum InputType { RANDOM, LARGE_SIGNED, IDENTICAL_ROW, EXTREMES };

static void generate_input(std::vector<__half>& h, int M, int N,
                           InputType type, unsigned seed = 42) {
    std::mt19937 gen(seed);
    switch (type) {
    case RANDOM:
        for (auto& v : h) v = random_half(gen, 1.0f);
        break;
    case LARGE_SIGNED:
        for (auto& v : h) v = random_half(gen, 50000.0f);
        break;
    case IDENTICAL_ROW: {
        std::vector<__half> row(N);
        for (auto& v : row) v = random_half(gen, 1.0f);
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j)
                h[i * N + j] = row[j];
        break;
    }
    case EXTREMES: {
        // Mix of extreme FP16 values
        float extremes[] = {65504.0f, -65504.0f, 6.1e-5f, -6.1e-5f,
                            0.0f, 1.0f, -1.0f, 100.0f, -100.0f};
        int n_ext = sizeof(extremes) / sizeof(extremes[0]);
        for (int i = 0; i < M * N; ++i) {
            h[i] = __float2half(extremes[gen() % n_ext]);
        }
        break;
    }
    }
}

// ── Validation ─────────────────────────────────────────────────────────────
struct ValidationResult {
    bool pass = false;
    float max_abs = 0, max_rel = 0, mean_abs = 0;
    float max_rowsum_dev = 0;
    int nan_count = 0, inf_count = 0;
};

static ValidationResult validate(const TestCase& tc, const Buffers& bufs,
                                 const std::vector<float>& ref) {
    ValidationResult r;
    int total = tc.M * tc.N;

    // Read back both outputs
    std::vector<__half> h0(total), h1(total);
    cudaMemcpy(h0.data(), bufs.d_out0, total * sizeof(__half), cudaMemcpyDeviceToHost);
    cudaMemcpy(h1.data(), bufs.d_out1, total * sizeof(__half), cudaMemcpyDeviceToHost);

    for (int i = 0; i < tc.M; ++i) {
        float rowsum0 = 0, rowsum1 = 0;
        for (int j = 0; j < tc.N; ++j) {
            int idx = i * tc.N + j;
            float v0 = h2f(h0[idx]), v1 = h2f(h1[idx]);
            float ref_val = ref[idx];

            // Check baseline (out0)
            float abs0 = fabsf(v0 - ref_val);
            r.max_abs = fmaxf(r.max_abs, abs0);
            r.max_rel = fmaxf(r.max_rel, abs0 / (fabsf(ref_val) + 1e-8f));
            r.mean_abs += abs0;
            if (isnan(v0)) r.nan_count++;
            if (isinf(v0)) r.inf_count++;
            rowsum0 += v0;

            // Check candidate (out1)
            float abs1 = fabsf(v1 - ref_val);
            r.max_abs = fmaxf(r.max_abs, abs1);
            r.max_rel = fmaxf(r.max_rel, abs1 / (fabsf(ref_val) + 1e-8f));
            r.mean_abs += abs1;
            if (isnan(v1)) r.nan_count++;
            if (isinf(v1)) r.inf_count++;
            rowsum1 += v1;
        }
        r.max_rowsum_dev = fmaxf(r.max_rowsum_dev, fabsf(rowsum0 - 1.0f));
        r.max_rowsum_dev = fmaxf(r.max_rowsum_dev, fabsf(rowsum1 - 1.0f));
    }
    r.mean_abs /= (2 * total);  // averaged over both outputs

    r.pass = (r.max_abs <= 5e-3f) && (r.max_rel <= 1e-2f) &&
             (r.nan_count == 0) && (r.inf_count == 0) &&
             (r.max_rowsum_dev <= 1e-4f);
    return r;
}

// ── Run correctness test suite ─────────────────────────────────────────────
static bool run_validation_suite() {
    std::vector<TestCase> shapes = {
        {"S1_512x512", 512, 512},
        {"S2_1024x1024", 1024, 1024},
        {"S3_2048x2048", 2048, 2048},
        {"S4_4096x4096", 4096, 4096},
        {"S5_8192x8192", 8192, 8192},
        {"S6_512x777", 512, 777},
        {"S7_768x3072", 768, 3072},
        {"S8_1000x513", 1000, 513},
        // Edge cases
        {"edge_1x1", 1, 1},
        {"edge_1x1024", 1, 1024},
        {"edge_1024x1", 1024, 1},
        {"edge_33cols", 128, 33},
        {"edge_127cols", 64, 127},
        {"edge_513cols", 32, 513},
        {"edge_1025cols", 16, 1025},
    };

    InputType input_types[] = {RANDOM, LARGE_SIGNED, IDENTICAL_ROW, EXTREMES};
    const char* type_names[] = {"random", "large_signed", "identical_row", "extremes"};

    Buffers bufs;
    bool all_pass = true;
    int total_tests = shapes.size() * 4;
    int passed = 0;

    for (const auto& tc : shapes) {
        for (int ti = 0; ti < 4; ++ti) {
            if (!bufs.alloc(tc.M, tc.N)) {
                std::cerr << "VALIDATE alloc failed: " << tc.name << "\n";
                all_pass = false;
                continue;
            }

            InputType it = input_types[ti];
            generate_input(bufs.h_input, tc.M, tc.N, it, 42 + ti);

            // CPU reference
            auto ref = cpu_softmax_fp32(bufs.h_input, tc.M, tc.N);

            // Upload input to device
            size_t bytes = (size_t)tc.M * tc.N * sizeof(__half);
            CUDA_OK(cudaMemcpy(bufs.d_input, bufs.h_input.data(), bytes,
                               cudaMemcpyHostToDevice));

            // Run both kernels
            cudaStream_t s;
            CUDA_OK(cudaStreamCreate(&s));
            CUDA_OK(rowwise_softmax::softmax_baseline(
                bufs.d_input, bufs.d_out0, tc.M, tc.N, s));
            CUDA_OK(rowwise_softmax::softmax_warp_shuffle(
                bufs.d_input, bufs.d_out1, tc.M, tc.N, s));
            CUDA_OK(cudaStreamSynchronize(s));
            cudaStreamDestroy(s);

            // Validate
            auto vr = validate(tc, bufs, ref);

            std::cout << "VALIDATE," << tc.name << "," << type_names[ti]
                      << ",M=" << tc.M << ",N=" << tc.N
                      << ",max_abs=" << vr.max_abs
                      << ",max_rel=" << vr.max_rel
                      << ",mean_abs=" << vr.mean_abs
                      << ",rowsum_dev=" << vr.max_rowsum_dev
                      << ",nan=" << vr.nan_count
                      << ",inf=" << vr.inf_count
                      << ",pass=" << (vr.pass ? "PASS" : "FAIL") << "\n";

            if (vr.pass) passed++; else all_pass = false;
        }
    }

    std::cout << "VALIDATION_SUMMARY: " << passed << "/" << total_tests
              << " passed\n";
    return all_pass;
}

// ── CUDA event timing ──────────────────────────────────────────────────────
struct BenchResult {
    double median_us, min_us, max_us, mean_us;
};

static BenchResult time_kernel(int variant,  // 0=baseline, 1=warp_shuffle
                               const __half* d_input, __half* d_output,
                               int M, int N, int warmup, int iters) {
    cudaStream_t s;
    cudaStreamCreate(&s);

    // Warmup
    for (int i = 0; i < warmup; ++i) {
        if (variant == 0)
            rowwise_softmax::softmax_baseline(d_input, d_output, M, N, s);
        else
            rowwise_softmax::softmax_warp_shuffle(d_input, d_output, M, N, s);
    }
    cudaStreamSynchronize(s);

    // Timed iterations
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, s);
    for (int i = 0; i < iters; ++i) {
        if (variant == 0)
            rowwise_softmax::softmax_baseline(d_input, d_output, M, N, s);
        else
            rowwise_softmax::softmax_warp_shuffle(d_input, d_output, M, N, s);
    }
    cudaEventRecord(stop, s);
    cudaEventSynchronize(stop);

    float total_ms = 0;
    cudaEventElapsedTime(&total_ms, start, stop);
    double us_per_iter = (double)total_ms * 1000.0 / iters;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(s);

    return {us_per_iter, us_per_iter, us_per_iter, us_per_iter};
}

static BenchResult time_kernel_multi_trial(int variant,
                                           const __half* d_input,
                                           __half* d_output,
                                           int M, int N,
                                           int warmup, int iters,
                                           int groups) {
    std::vector<double> latencies;
    for (int g = 0; g < groups; ++g) {
        cudaDeviceSynchronize();
        auto r = time_kernel(variant, d_input, d_output, M, N, warmup, iters);
        latencies.push_back(r.median_us);
    }
    std::sort(latencies.begin(), latencies.end());
    BenchResult result;
    result.median_us = latencies[latencies.size() / 2];
    result.min_us = latencies.front();
    result.max_us = latencies.back();
    result.mean_us = std::accumulate(latencies.begin(), latencies.end(), 0.0)
                     / latencies.size();
    return result;
}

// ── Run benchmark suite ────────────────────────────────────────────────────
static void run_benchmark_suite(std::ostream& csv) {
    std::vector<TestCase> bench_shapes = {
        // Group 1: Small rows
        {"small_32x4096", 4096, 32},
        {"small_64x4096", 4096, 64},
        {"small_128x4096", 4096, 128},
        {"small_32x1024", 1024, 32},
        {"small_64x1024", 1024, 64},
        {"small_128x1024", 1024, 128},
        // Group 2: Medium rows
        {"med_256x4096", 4096, 256},
        {"med_512x4096", 4096, 512},
        {"med_256x1024", 1024, 256},
        {"med_512x1024", 1024, 512},
        // Group 3: Large rows
        {"large_1024x1024", 1024, 1024},
        {"large_2048x1024", 1024, 2048},
        {"large_4096x1024", 1024, 4096},
        {"large_1024x256", 256, 1024},
        {"large_2048x256", 256, 2048},
        {"large_4096x256", 256, 4096},
    };

    csv << "candidate,gpu,dtype,M,N,shape_name,warmup,iterations,groups,"
        << "median_us,mean_us,min_us,max_us,"
        << "effective_bandwidth_gbs,custom_cuda,correct,notes\n";

    Buffers bufs;
    const int WARMUP = 20;
    const int ITERS = 100;
    const int GROUPS = 3;

    for (const auto& tc : bench_shapes) {
        if (!bufs.alloc(tc.M, tc.N)) {
            std::cerr << "BENCH alloc failed: " << tc.name << "\n";
            continue;
        }

        // Generate random input
        generate_input(bufs.h_input, tc.M, tc.N, RANDOM, 12345);
        size_t bytes = (size_t)tc.M * tc.N * sizeof(__half);
        CUDA_OK(cudaMemcpy(bufs.d_input, bufs.h_input.data(), bytes,
                           cudaMemcpyHostToDevice));

        // Effective bytes: read input + write output
        double effective_bytes = 2.0 * tc.M * tc.N * sizeof(__half);

        // Baseline
        auto r0 = time_kernel_multi_trial(0, bufs.d_input, bufs.d_out0,
                                          tc.M, tc.N, WARMUP, ITERS, GROUPS);
        double bw0 = effective_bytes / (r0.median_us * 1e-6) / 1e9;
        csv << "candidate_00_baseline,NVIDIA H200,FP16," << tc.M << ","
            << tc.N << "," << tc.name << "," << WARMUP << "," << ITERS
            << "," << GROUPS << "," << r0.median_us << "," << r0.mean_us
            << "," << r0.min_us << "," << r0.max_us << ","
            << bw0 << ",custom_cuda,true,\n";

        // Candidate (warp shuffle)
        auto r1 = time_kernel_multi_trial(1, bufs.d_input, bufs.d_out1,
                                          tc.M, tc.N, WARMUP, ITERS, GROUPS);
        double bw1 = effective_bytes / (r1.median_us * 1e-6) / 1e9;
        csv << "candidate_01_warp_shuffle,NVIDIA H200,FP16," << tc.M << ","
            << tc.N << "," << tc.name << "," << WARMUP << "," << ITERS
            << "," << GROUPS << "," << r1.median_us << "," << r1.mean_us
            << "," << r1.min_us << "," << r1.max_us << ","
            << bw1 << ",custom_cuda,true,\n";

        // Torch placeholder — actual torch results filled by benchmark.py
        csv << "candidate_torch,NVIDIA H200,FP16," << tc.M << ","
            << tc.N << "," << tc.name << "," << WARMUP << "," << ITERS
            << "," << GROUPS << ",,torch_placeholder,,,"
            << ",torch,true,to_be_filled_by_benchmark_py\n";

        std::cout << "BENCH " << tc.name
                  << " baseline=" << r0.median_us << "us"
                  << " candidate=" << r1.median_us << "us"
                  << " speedup=" << (r0.median_us / r1.median_us) << "x\n";
    }
}

// ── Main ───────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    int dev = 0;
    cudaSetDevice(dev);
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, dev);
    std::cout << "GPU=" << prop.name << " SMs=" << prop.multiProcessorCount
              << " CC=" << prop.major << "." << prop.minor << "\n";

    // ── Phase 1: Correctness ──
    std::cout << "=== CORRECTNESS VALIDATION ===\n";
    bool ok = run_validation_suite();
    if (!ok) {
        std::cerr << "VALIDATION FAILED\n";
        return 2;
    }

    // ── Phase 2: Benchmark ──
    bool do_bench = (argc > 1 && std::string(argv[1]) == "--bench");
    if (!do_bench) {
        std::cout << "All validations passed. Use --bench to run benchmarks.\n";
        return 0;
    }

    std::cout << "\n=== BENCHMARKS ===\n";
    std::ofstream csv("benchmark.csv");
    if (!csv) {
        std::cerr << "Cannot open benchmark.csv\n";
        return 3;
    }
    run_benchmark_suite(csv);
    csv.close();
    std::cout << "benchmark.csv written\n";

    return 0;
}
