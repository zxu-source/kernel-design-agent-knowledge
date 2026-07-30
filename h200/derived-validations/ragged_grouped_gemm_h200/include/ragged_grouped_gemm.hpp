#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace ragged_gemm {

struct TileTask {
  int expert;
  int tile_m;
  int tile_n;
};

struct DeviceMetadata {
  TileTask* tasks = nullptr;
  int task_count = 0;
  int* ticket = nullptr;
};

// A/B/C use row-major BF16 elements. A and C are packed by their respective
// offsets; B contains E consecutive KxN matrices.  M_e==0 is legal.
cudaError_t build_metadata(DeviceMetadata* metadata, const int* host_counts,
                           int experts, int n, cudaStream_t stream);
void destroy_metadata(DeviceMetadata* metadata);

cudaError_t grouped_gemm_baseline(const __nv_bfloat16* a,
                                  const __nv_bfloat16* b,
                                  __nv_bfloat16* c,
                                  const int* host_counts,
                                  const int64_t* host_a_offsets,
                                  const int64_t* host_b_offsets,
                                  const int64_t* host_c_offsets,
                                  int experts, int k, int n,
                                  cudaStream_t stream);

cudaError_t grouped_gemm_persistent(const __nv_bfloat16* a,
                                    const __nv_bfloat16* b,
                                    __nv_bfloat16* c,
                                    const int* host_counts,
                                    const int64_t* host_a_offsets,
                                    const int64_t* host_b_offsets,
                                    const int64_t* host_c_offsets,
                                    int experts, int k, int n,
                                    DeviceMetadata* metadata,
                                    cudaStream_t stream);

cudaError_t grouped_gemm_static_queue(const __nv_bfloat16* a,
                                      const __nv_bfloat16* b,
                                      __nv_bfloat16* c,
                                      const int* host_counts,
                                      const int64_t* host_a_offsets,
                                      const int64_t* host_b_offsets,
                                      const int64_t* host_c_offsets,
                                      int experts, int k, int n,
                                      const DeviceMetadata* metadata,
                                      cudaStream_t stream);

}  // namespace ragged_gemm
