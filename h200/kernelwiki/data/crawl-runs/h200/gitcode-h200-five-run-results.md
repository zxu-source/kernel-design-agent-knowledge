# GitCode CUDA Samples — Five H200 Property Runs

Date: 2026-07-20

Sources were collected through the authenticated GitCode v5 content API from
`ZouJiu/cuda-samples` at `ef3068e998f1bf87e52cedec480c6c917ee92d82`.
Each result is a source-informed standalone CUDA microbenchmark compiled with
`nvcc -O3 -std=c++17 -arch=sm_90`; each latency is the mean of 50 launches
after 10 warmups except allocation timing, which is 100 host API pairs. Full
upstream samples and their published benchmark numbers were not claimed as
reproduced.

| Iteration | GitCode source | Property tested on H200 | Correctness | Baseline / optimized latency | Speedup | Conclusion |
|---|---|---|---|---|---:|---|
| 1 | `streamOrderedAllocation.cu` | `cudaMallocAsync/cudaFreeAsync` versus synchronous allocation | PASS | 0.1491 / 0.0559 ms | 2.667x | Stream-ordered allocation reduced allocation-pair overhead in this run. |
| 2 | `matrixMul.cu` | Shared-memory tiled GEMM versus naive global-memory GEMM | PASS, 0 mismatches | 0.0583 / 0.0386 ms | 1.510x | Shared-memory reuse property is supported. |
| 3 | `alignedTypes.cu` | Aligned `float4` copy versus scalar copy | PASS, 0 mismatches | 0.0538 / 0.0348 ms | 1.546x | Aligned vector access improved copy throughput. |
| 4 | `globalToShmemAsyncCopy.cu` | `cuda::memcpy_async` copy versus direct copy | PASS, 0 mismatches | 0.0537 / 0.0561 ms | 0.957x | Correct, but no gain without useful overlapped computation. |
| 5 | `LargeKernelParameter.cu` | 256-byte parameter by value versus device-pointer parameter | PASS, 0 mismatches | 0.0389 / 0.0118 ms | 3.283x | Pointer path was faster for this synthetic parameter-access pattern; not a universal parameter-passing rule. |
