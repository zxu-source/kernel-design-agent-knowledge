// Hopper TMA STORE (cp.async.bulk.tensor.2d shared->global) on H200 (SM90a).
// Counterpart to hopper_tma_copy.cu (TMA load). Each block fills a shared tile
// deterministically, then writes it to global either via TMA store
// (cp.async.bulk.tensor.2d.global.shared + commit_group/wait_group) or via a
// naive per-thread store. Both produce identical global output; we compare the
// pure store bandwidth (global write bytes / time). Compile: nvcc -O3 -std=c++17
// -arch=sm_90a -lcuda.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(x) do{auto e=(x); if(e!=cudaSuccess){fprintf(stderr,"CUDA %d %s\n",__LINE__,cudaGetErrorString(e));return 1;}}while(0)
#define CU_CHECK(x) do{auto r=(x); if(r!=CUDA_SUCCESS){fprintf(stderr,"CU %d rc=%d\n",__LINE__,(int)r);return 1;}}while(0)

#ifndef BOX_M
#define BOX_M 128
#endif
#ifndef BOX_N
#define BOX_N 128
#endif

__device__ inline void cp_async_bulk_tensor_2d_store(void* smem, const void* tmap, int x, int y){
  uint32_t s=(uint32_t)__cvta_generic_to_shared(smem);
  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group [%1, {%2, %3}], [%0];"
               ::"r"(s),"l"(tmap),"r"(x),"r"(y):"memory");
}
__device__ inline void cp_async_bulk_commit_group(){
  asm volatile("cp.async.bulk.commit_group;":::"memory");
}
__device__ inline void cp_async_bulk_wait_group_0(){
  asm volatile("cp.async.bulk.wait_group 0;":::"memory");
}

__global__ void tma_store(float* out, const CUtensorMap* tmap){
  __shared__ alignas(128) float smem[BOX_M*BOX_N];
  int tid=threadIdx.x;
  int x0=blockIdx.x*BOX_N, y0=blockIdx.y*BOX_M;
  // fill shared deterministically (same in both kernels)
  for(int i=tid;i<BOX_M*BOX_N;i+=blockDim.x) smem[i]=(float)((blockIdx.x*1000+blockIdx.y)*1.0f + i);
  __syncthreads();
  // TMA store shared -> global
  cp_async_bulk_tensor_2d_store(smem, tmap, x0, y0);
  cp_async_bulk_commit_group();
  cp_async_bulk_wait_group_0();
}

__global__ void naive_store(float* out){
  __shared__ float smem[BOX_M*BOX_N];
  int tid=threadIdx.x;
  int x0=blockIdx.x*BOX_N, y0=blockIdx.y*BOX_M;
  for(int i=tid;i<BOX_M*BOX_N;i+=blockDim.x) smem[i]=(float)((blockIdx.x*1000+blockIdx.y)*1.0f + i);
  __syncthreads();
  int GN=gridDim.x*BOX_N;
  for(int i=tid;i<BOX_M*BOX_N;i+=blockDim.x){ int r=i/BOX_N,c=i%BOX_N; out[(size_t)(y0+r)*GN + x0 + c]=smem[i]; }
}

int main(){
  const int GM=2048, GN=2048;
  const int nbx=GN/BOX_N, nby=GM/BOX_M;
  size_t N=(size_t)GM*GN;
  printf("H200 TMA STORE: box=%dx%d grid=%dx%d tensor=%dx%d (%.1fMB)\n",BOX_M,BOX_N,nbx,nby,GM,GN,(double)N*4/1e6);
  float *doutA,*doutB;
  CUDA_CHECK(cudaMalloc(&doutA,N*4)); CUDA_CHECK(cudaMalloc(&doutB,N*4));
  CUtensorMap tmap;
  cuuint64_t gd[2]={GN,GM}, gs[1]={(cuuint64_t)GN*4}; cuuint32_t bd[2]={BOX_N,BOX_M}, es[2]={1,1};
  CU_CHECK(cuTensorMapEncodeTiled(&tmap,CU_TENSOR_MAP_DATA_TYPE_FLOAT32,2,(void*)doutA,gd,gs,bd,es,
    CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_NONE,CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  CUtensorMap* dtmap; CUDA_CHECK(cudaMalloc(&dtmap,sizeof(CUtensorMap)));
  CUDA_CHECK(cudaMemcpy(dtmap,&tmap,sizeof(CUtensorMap),cudaMemcpyHostToDevice));
  cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
  dim3 grid(nbx,nby), blk(256);
  auto bench=[&](bool tma,double bytes)->double{
    double best=1e9,sum=0; const int T=20;
    for(int t=0;t<T;++t){ cudaEventRecord(s);
      if(tma) tma_store<<<grid,blk>>>(doutA,dtmap); else naive_store<<<grid,blk>>>(doutB);
      cudaEventRecord(e); cudaEventSynchronize(e);
      float ms=0; cudaEventElapsedTime(&ms,s,e); best=std::min(best,(double)ms); sum+=ms; }
    printf("%-5s min=%.4fms med=%.4fms bw=%.0fGB/s\n", tma?"tma":"naive", best,sum/T, bytes/(best*1e-3)/1e9);
    return best;
  };
  double bytes=(double)N*4;
  double bt=bench(true,bytes), bn=bench(false,bytes);
  std::vector<float> ha(N),hb(N);
  CUDA_CHECK(cudaMemcpy(ha.data(),doutA,N*4,cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(hb.data(),doutB,N*4,cudaMemcpyDeviceToHost));
  double maxdiff=0; for(size_t i=0;i<N;++i){double d=fabs((double)ha[i]-(double)hb[i]); if(d>maxdiff)maxdiff=d;}
  printf("correctness: |tma-naive|=%.3e\n", maxdiff);
  printf("RESULT tma_store=%.4fms naive_store=%.4fms naive/tma=%.3fx tma_bw=%.0fGB/s naive_bw=%.0fGB/s\n",
         bt,bn,bn/bt, bytes/(bt*1e-3)/1e9, bytes/(bn*1e-3)/1e9);
  cudaFree(doutA);cudaFree(doutB);cudaFree(dtmap); return 0;
}
