#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace rowwise_softmax {

// Launch rowwise softmax using shared-memory tree reduction (baseline).
// input:  [M, N] FP16 row-major on device
// output: [M, N] FP16 row-major on device
// M: number of rows, N: number of columns
cudaError_t softmax_baseline(const __half* input, __half* output,
                             int M, int N, cudaStream_t stream = 0);

// Launch rowwise softmax using hierarchical warp-shuffle reduction (candidate).
// Same interface as baseline; differs only in reduction strategy.
cudaError_t softmax_warp_shuffle(const __half* input, __half* output,
                                 int M, int N, cudaStream_t stream = 0);

}  // namespace rowwise_softmax
