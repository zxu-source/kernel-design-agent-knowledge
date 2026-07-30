#include "ragged_grouped_gemm.hpp"

#include <cuda_bf16.h>
#include <algorithm>
#include <vector>

namespace ragged_gemm {
namespace {
constexpr int kBlockM = 32;
constexpr int kBlockN = 32;
constexpr int kThreads = 128;

struct DeviceArgs {
  const __nv_bfloat16* a;
  const __nv_bfloat16* b;
  __nv_bfloat16* c;
  const int* counts;
  const int64_t* a_offsets;
  const int64_t* b_offsets;
  const int64_t* c_offsets;
  int k;
  int n;
};

__device__ __forceinline__ void compute_tile(const DeviceArgs& x, int expert,
                                             int block_m, int block_n) {
  const int m = x.counts[expert];
  // Safe reference micro-kernel: all bounds are checked before each packed
  // access.  This is intentionally the shared math path for candidates 00/01;
  // the experiment isolates scheduling rather than conflating it with MMA
  // lowering.  A later Tensor-Core path must match this reference first.
  for (int linear = threadIdx.x; linear < kBlockM * kBlockN; linear += blockDim.x) {
    const int row = block_m + linear / kBlockN;
    const int col = block_n + linear % kBlockN;
    if (row >= m || col >= x.n) continue;
    const auto* ar = x.a + x.a_offsets[expert] + static_cast<int64_t>(row) * x.k;
    const auto* bc = x.b + x.b_offsets[expert] + col;
    float sum = 0.0f;
    for (int kk = 0; kk < x.k; ++kk)
      sum += __bfloat162float(ar[kk]) * __bfloat162float(bc[static_cast<int64_t>(kk) * x.n]);
    x.c[x.c_offsets[expert] + static_cast<int64_t>(row) * x.n + col] = __float2bfloat16(sum);
  }
}

__global__ void per_expert_kernel(DeviceArgs x, int expert) {
  compute_tile(x, expert, blockIdx.y * kBlockM, blockIdx.x * kBlockN);
}

__global__ void persistent_kernel(DeviceArgs x, const TileTask* tasks, int count,
                                  int* ticket) {
  __shared__ TileTask task;
  __shared__ int active;
  while (true) {
    if (threadIdx.x == 0) {
      const int id = atomicAdd(ticket, 1);
      active = id < count;
      if (active) task = tasks[id];
    }
    __syncthreads();
    if (!active) return;
    compute_tile(x, task.expert, task.tile_m * kBlockM, task.tile_n * kBlockN);
    __syncthreads();
  }
}

__global__ void static_queue_kernel(DeviceArgs x, const TileTask* tasks, int count) {
  const int id = blockIdx.x;
  if (id >= count) return;
  const TileTask t = tasks[id];
  compute_tile(x, t.expert, t.tile_m * kBlockM, t.tile_n * kBlockN);
}

cudaError_t upload_args(DeviceArgs* out, const __nv_bfloat16* a,
                        const __nv_bfloat16* b, __nv_bfloat16* c,
                        const int* counts, const int64_t* ao, const int64_t* bo,
                        const int64_t* co, int experts, int k, int n,
                        cudaStream_t stream, int** d_counts,
                        int64_t** d_ao, int64_t** d_bo, int64_t** d_co) {
  cudaError_t st;
  if ((st = cudaMallocAsync(reinterpret_cast<void**>(d_counts), experts * sizeof(int), stream)) != cudaSuccess ||
      (st = cudaMallocAsync(reinterpret_cast<void**>(d_ao), experts * sizeof(int64_t), stream)) != cudaSuccess ||
      (st = cudaMallocAsync(reinterpret_cast<void**>(d_bo), experts * sizeof(int64_t), stream)) != cudaSuccess ||
      (st = cudaMallocAsync(reinterpret_cast<void**>(d_co), experts * sizeof(int64_t), stream)) != cudaSuccess) return st;
  if ((st = cudaMemcpyAsync(*d_counts, counts, experts * sizeof(int), cudaMemcpyHostToDevice, stream)) != cudaSuccess ||
      (st = cudaMemcpyAsync(*d_ao, ao, experts * sizeof(int64_t), cudaMemcpyHostToDevice, stream)) != cudaSuccess ||
      (st = cudaMemcpyAsync(*d_bo, bo, experts * sizeof(int64_t), cudaMemcpyHostToDevice, stream)) != cudaSuccess ||
      (st = cudaMemcpyAsync(*d_co, co, experts * sizeof(int64_t), cudaMemcpyHostToDevice, stream)) != cudaSuccess) return st;
  *out = {a, b, c, *d_counts, *d_ao, *d_bo, *d_co, k, n};
  return cudaSuccess;
}

void free_args(int* counts, int64_t* ao, int64_t* bo, int64_t* co, cudaStream_t stream) {
  cudaFreeAsync(counts, stream); cudaFreeAsync(ao, stream); cudaFreeAsync(bo, stream); cudaFreeAsync(co, stream);
}
}  // namespace

cudaError_t build_metadata(DeviceMetadata* md, const int* counts, int experts, int n,
                           cudaStream_t stream) {
  if (!md || !counts || experts <= 0 || n <= 0) return cudaErrorInvalidValue;
  std::vector<TileTask> tasks;
  for (int e = 0; e < experts; ++e) {
    if (counts[e] < 0) return cudaErrorInvalidValue;
    for (int m = 0; m < counts[e]; m += kBlockM)
      for (int col = 0; col < n; col += kBlockN) tasks.push_back({e, m / kBlockM, col / kBlockN});
  }
  destroy_metadata(md);
  md->task_count = static_cast<int>(tasks.size());
  if (md->task_count == 0) return cudaSuccess;
  cudaError_t st = cudaMallocAsync(reinterpret_cast<void**>(&md->tasks), tasks.size() * sizeof(TileTask), stream);
  if (st != cudaSuccess) return st;
  if ((st = cudaMallocAsync(reinterpret_cast<void**>(&md->ticket), sizeof(int), stream)) != cudaSuccess) { destroy_metadata(md); return st; }
  return cudaMemcpyAsync(md->tasks, tasks.data(), tasks.size() * sizeof(TileTask), cudaMemcpyHostToDevice, stream);
}

void destroy_metadata(DeviceMetadata* md) {
  if (!md) return;
  if (md->tasks) cudaFree(md->tasks);
  if (md->ticket) cudaFree(md->ticket);
  *md = {};
}

cudaError_t grouped_gemm_baseline(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* c,
                                  const int* counts, const int64_t* ao, const int64_t* bo, const int64_t* co,
                                  int experts, int k, int n, cudaStream_t stream) {
  if (!a || !b || !c || !counts || !ao || !bo || !co || experts <= 0 || k <= 0 || n <= 0 || k % 16) return cudaErrorInvalidValue;
  for (int e = 0; e < experts; ++e) if (counts[e] < 0) return cudaErrorInvalidValue;
  DeviceArgs x; int *dc = nullptr; int64_t *da = nullptr, *db = nullptr, *dd = nullptr;
  cudaError_t st = upload_args(&x,a,b,c,counts,ao,bo,co,experts,k,n,stream,&dc,&da,&db,&dd); if (st != cudaSuccess) return st;
  for (int e=0;e<experts;++e) if (counts[e]) per_expert_kernel<<<dim3((n+31)/32,(counts[e]+31)/32),kThreads,0,stream>>>(x,e);
  st = cudaPeekAtLastError(); free_args(dc,da,db,dd,stream); return st;
}

cudaError_t grouped_gemm_persistent(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* c,
                                    const int* counts, const int64_t* ao, const int64_t* bo, const int64_t* co,
                                    int experts, int k, int n, DeviceMetadata* md, cudaStream_t stream) {
  if (!a || !b || !c || !counts || !ao || !bo || !co || !md || md->task_count < 0 || experts <= 0 || k <= 0 || n <= 0 || k % 16) return cudaErrorInvalidValue;
  for (int e = 0; e < experts; ++e) if (counts[e] < 0) return cudaErrorInvalidValue;
  if (md->task_count == 0) return cudaSuccess;
  DeviceArgs x; int *dc = nullptr; int64_t *da = nullptr, *db = nullptr, *dd = nullptr;
  cudaError_t st = upload_args(&x,a,b,c,counts,ao,bo,co,experts,k,n,stream,&dc,&da,&db,&dd); if (st != cudaSuccess) return st;
  cudaMemsetAsync(md->ticket,0,sizeof(int),stream);
  int dev=0,sms=1; cudaGetDevice(&dev); cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,dev);
  persistent_kernel<<<std::min(md->task_count, sms*2),kThreads,0,stream>>>(x,md->tasks,md->task_count,md->ticket);
  st=cudaPeekAtLastError(); free_args(dc,da,db,dd,stream); return st;
}

cudaError_t grouped_gemm_static_queue(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* c,
                                      const int* counts, const int64_t* ao, const int64_t* bo, const int64_t* co,
                                      int experts, int k, int n, const DeviceMetadata* md, cudaStream_t stream) {
  if (!a || !b || !c || !counts || !ao || !bo || !co || !md || md->task_count < 0 || experts <= 0 || k <= 0 || n <= 0 || k % 16) return cudaErrorInvalidValue;
  for (int e = 0; e < experts; ++e) if (counts[e] < 0) return cudaErrorInvalidValue;
  if (md->task_count == 0) return cudaSuccess;
  DeviceArgs x; int *dc = nullptr; int64_t *da = nullptr, *db = nullptr, *dd = nullptr;
  cudaError_t st = upload_args(&x,a,b,c,counts,ao,bo,co,experts,k,n,stream,&dc,&da,&db,&dd); if (st != cudaSuccess) return st;
  static_queue_kernel<<<md->task_count,kThreads,0,stream>>>(x,md->tasks,md->task_count);
  st=cudaPeekAtLastError(); free_args(dc,da,db,dd,stream); return st;
}
}  // namespace ragged_gemm
