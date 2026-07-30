// Minimal Hopper DSMEM probe: 2-block cluster. Block 0 writes 99 into block 1's
// shared via map_shared_rank; block 1 writes its shared value to global. Confirms
// whether cross-block shared WRITE via DSMEM is functional on this H200 toolchain.
#include <cstdio>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
__global__ void dsmem_probe(int* out) {
  __shared__ float x;
  cg::cluster_group cl = cg::this_cluster();
  int rank = cl.block_rank();
  if (threadIdx.x == 0) x = (rank == 0) ? 42.0f : -1.0f;
  __syncthreads();
  cl.sync();
  // block 0 pushes a value into block 1's shared via DSMEM
  if (rank == 0 && threadIdx.x == 0) {
    float* peer = cl.map_shared_rank(&x, 1);
    *peer = 99.0f;
  }
  cl.sync();
  // block 1 reads its (DSMEM-written) shared and emits to global
  if (rank == 1 && threadIdx.x == 0) *out = (int)x;
}
int main() {
  int* d; int h = -2;
  cudaMalloc(&d, sizeof(int));
  cudaLaunchConfig_t cfg = {}; cfg.gridDim = dim3(2,1,1); cfg.blockDim = dim3(32,1,1);
  cudaLaunchAttribute a; a.id = cudaLaunchAttributeClusterDimension;
  a.val.clusterDim = {2,1,1}; cfg.attrs=&a; cfg.numAttrs=1;
  cudaError_t le = cudaLaunchKernelEx(&cfg, dsmem_probe, d);
  cudaError_t se = cudaDeviceSynchronize();
  cudaMemcpy(&h, d, sizeof(int), cudaMemcpyDeviceToHost);
  printf("launch_err=%d sync_err=%d out=%d (99 == DSMEM write works)\n", (int)le,(int)se,h);
  cudaFree(d); return 0;
}
