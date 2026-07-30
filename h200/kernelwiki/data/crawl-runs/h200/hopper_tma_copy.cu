// Hopper TMA (cp.async.bulk.tensor via CUtensorMap) 2D tile copy on H200 (SM90).
// Each block copies a BOX_M x BOX_N tile from global to shared via TMA
// (cp.async.bulk.tensor.2d + mbarrier), then copies shared -> global out.
// Compared against a naive per-thread global->shared->global copy of the same
// tile. Validated for correctness (out == src) and bandwidth (GB/s).
// Compile: nvcc -O3 -std=c++17 -arch=sm_90a ; TMA needs sm_90a.
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

__device__ inline void mbarrier_init(uint64_t* m, uint32_t count){
  uint32_t a=(uint32_t)__cvta_generic_to_shared(m);
  asm volatile("mbarrier.init.shared.b64 [%0], %1;"::"r"(a),"r"(count));
}
__device__ inline void mbarrier_arrive_expect_tx(uint64_t* m, uint32_t bytes){
  uint32_t a=(uint32_t)__cvta_generic_to_shared(m);
  asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;"::"r"(a),"r"(bytes));
}
__device__ inline void mbarrier_wait(uint64_t* m, int phase){
  uint32_t a=(uint32_t)__cvta_generic_to_shared(m);
  asm volatile(
    "{ .reg .pred p;\n"
    "LAB%=: mbarrier.try_wait.parity.shared.b64 p, [%0], %1;\n"
    "@!p bra LAB%=;\n"
    "}"
    :: "r"(a), "r"(phase));
}
// cp.async.bulk.tensor.2d: copy a 2D box from global (described by tmap) at coord
// {x,y} into shared [smem], tracked by mbarrier [mbar].
__device__ inline void cp_async_bulk_tensor_2d(void* smem, const void* tmap, int x, int y, uint64_t* mbar){
  uint32_t s=(uint32_t)__cvta_generic_to_shared(smem);
  uint32_t mb=(uint32_t)__cvta_generic_to_shared(mbar);
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
               ::"r"(s),"l"(tmap),"r"(x),"r"(y),"r"(mb):"memory");
}

__global__ void tma_copy(const float* src, float* out, const CUtensorMap* tmap){
  __shared__ alignas(128) float smem[BOX_M*BOX_N];
  __shared__ alignas(8) uint64_t mbar;
  int tx=threadIdx.x, ty=threadIdx.y;   // not used as 2D here
  int tid=threadIdx.x;
  // box origin for this block: block x -> column tile, block y -> row tile
  int x0 = blockIdx.x * BOX_N;   // fast dim
  int y0 = blockIdx.y * BOX_M;   // slow dim
  if(tid==0){ mbarrier_init(&mbar, 1); }
  __syncthreads();
  if(tid==0){
    cp_async_bulk_tensor_2d(smem, tmap, x0, y0, &mbar);
    mbarrier_arrive_expect_tx(&mbar, (uint32_t)(BOX_M*BOX_N*sizeof(float)));
  }
  mbarrier_wait(&mbar, 0);
  __syncthreads();
  // shared -> global out (correct row-major position; tile rows are GN apart)
  int GN = gridDim.x * BOX_N;
  for(int i=tid;i<BOX_M*BOX_N;i+=blockDim.x){
    int r=i/BOX_N, c=i%BOX_N;
    out[(size_t)(y0+r)*GN + x0 + c]=smem[i];
  }
}

__global__ void naive_copy(const float* src, float* out, int GM, int GN){
  __shared__ float smem[BOX_M*BOX_N];
  int x0=blockIdx.x*BOX_N, y0=blockIdx.y*BOX_M;
  int tid=threadIdx.x;
  for(int i=tid;i<BOX_M*BOX_N;i+=blockDim.x){
    int r=i/BOX_N, c=i%BOX_N;
    smem[i]=src[(size_t)(y0+r)*GN + (x0+c)];
  }
  __syncthreads();
  // shared -> global out (correct row-major position)
  int GNc = gridDim.x * BOX_N;
  for(int i=tid;i<BOX_M*BOX_N;i+=blockDim.x){
    int r=i/BOX_N, c=i%BOX_N;
    out[(size_t)(y0+r)*GNc + x0 + c]=smem[i];
  }
}

int main(){
  const int GM=2048, GN=2048;        // global tensor rows x cols (floats)
  const int nbx=GN/BOX_N, nby=GM/BOX_M;
  size_t N=(size_t)GM*GN;
  printf("H200 TMA 2D copy: box=%dx%d grid=%dx%d tensor=%dx%d floats (%.1fMB)\n",
         BOX_M,BOX_N,nbx,nby,GM,GN,(double)N*4/1e6);
  std::vector<float> hsrc(N);
  for(size_t i=0;i<N;++i) hsrc[i]=(float)((i*2654435761u)%1009)/97.0f;
  float *dsrc,*doutA,*doutB;
  CUDA_CHECK(cudaMalloc(&dsrc,N*4)); CUDA_CHECK(cudaMalloc(&doutA,N*4)); CUDA_CHECK(cudaMalloc(&doutB,N*4));
  CUDA_CHECK(cudaMemcpy(dsrc,hsrc.data(),N*4,cudaMemcpyHostToDevice));

  // build CUtensorMap (2D, row-major). globalDim is in elements; globalStride in bytes (rank-1 = 1 entry).
  CUtensorMap tmap;
  cuuint64_t globalDim[2]={GN, GM};           // {fast(dim0=cols), slow(dim1=rows)} in elements
  cuuint64_t globalStride[1]={(cuuint64_t)GN*sizeof(float)};  // bytes for dim1 stride
  cuuint32_t boxDim[2]={BOX_N, BOX_M};        // box size in elements per dim
  cuuint32_t elemStrides[2]={1,1};
  CU_CHECK(cuTensorMapEncodeTiled(&tmap,
    CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2, (void*)dsrc,
    globalDim, globalStride, boxDim, elemStrides,
    CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
    CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  // copy tmap to device mem (TMA reads the descriptor from generic memory)
  CUtensorMap* dtmap;
  CUDA_CHECK(cudaMalloc(&dtmap,sizeof(CUtensorMap)));
  CUDA_CHECK(cudaMemcpy(dtmap,&tmap,sizeof(CUtensorMap),cudaMemcpyHostToDevice));

  cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
  dim3 grid(nbx,nby), blk(256);
  auto bench=[&](bool tma,double bytes)->double{
    double best=1e9,sum=0; const int T=20;
    for(int t=0;t<T;++t){
      cudaEventRecord(s);
      if(tma) tma_copy<<<grid,blk>>>(dsrc,doutA,dtmap);
      else    naive_copy<<<grid,blk>>>(dsrc,doutB,GM,GN);
      cudaEventRecord(e); cudaEventSynchronize(e);
      float ms=0; cudaEventElapsedTime(&ms,s,e); best=std::min(best,(double)ms); sum+=ms;
    }
    printf("%-5s min=%.4fms med=%.4fms bw=%.0fGB/s\n", tma?"tma":"naive", best,sum/T, bytes/(best*1e-3)/1e9);
    return best;
  };
  double bytes=(double)N*4;     // global->shared read volume for bandwidth
  double bt=bench(true,bytes), bn=bench(false,bytes);
  // correctness
  std::vector<float> ha(N),hb(N);
  CUDA_CHECK(cudaMemcpy(ha.data(),doutA,N*4,cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(hb.data(),doutB,N*4,cudaMemcpyDeviceToHost));
  double maxerr=0; for(size_t i=0;i<N;++i){double d=fabs((double)ha[i]-(double)hsrc[i]); if(d>maxerr)maxerr=d;}
  double maxdiff=0; for(size_t i=0;i<N;++i){double d=fabs((double)ha[i]-(double)hb[i]); if(d>maxdiff)maxdiff=d;}
  printf("correctness: |tma-src|=%.3e |tma-naive|=%.3e\n", maxerr, maxdiff);
  printf("RESULT tma=%.4fms naive=%.4fms naive/tma=%.3fx tma_bw=%.0fGB/s naive_bw=%.0fGB/s\n",
         bt,bn,bn/bt, bytes/(bt*1e-3)/1e9, bytes/(bn*1e-3)/1e9);
  cudaFree(dsrc);cudaFree(doutA);cudaFree(doutB);cudaFree(dtmap);
  return 0;
}
