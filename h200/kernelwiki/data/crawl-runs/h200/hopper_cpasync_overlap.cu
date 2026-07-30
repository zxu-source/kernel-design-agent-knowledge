// Hopper cp.async multi-stage async-copy/compute overlap on H200 (SM90).
// Directly addresses Codex's negative control (cuda::memcpy_async copy-only
// was 0.957x with NO compute to overlap). Here a real per-tile compute is
// overlapped with the next tile's global->shared copy via __pipeline_* (cp.async
// + cp.async.commit_group/wait_group). Serial (copy then compute) vs pipelined.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_pipeline.h>

#define CUDA_CHECK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"CUDA %d %s\n",__LINE__,cudaGetErrorString(e));return 1;}}while(0)

#ifndef STAGES
#define STAGES 3
#endif

// Per-tile compute: deterministic, tunable flops. Accumulates in REGISTERS and
// stores to global ONCE (so more REPS adds pure compute, not bandwidth). This is
// what makes the kernel compute-bound enough for async-copy overlap to help.
template<int REPS>
__device__ inline void compute_tile(const float* in, float* out, int tile, int tid, int nthr) {
  for (int j = tid; j < tile; j += nthr) {
    float x = in[j];
    float y = x;
    #pragma unroll 1
    for (int r = 0; r < REPS; ++r) {
      y = y * y + x;   // 2 flops/rep, register-only (no extra memory traffic)
    }
    out[j] += y;       // single global store per element
  }
}

template<int REPS>
__global__ void serial_kernel(const float* in, float* out, int tiles_per_block, int tile) {
  extern __shared__ float s[];
  int tid = threadIdx.x, nthr = blockDim.x;
  size_t base = (size_t)blockIdx.x * tiles_per_block * tile;
  for (int i = 0; i < tiles_per_block; ++i) {
    for (int j = tid; j < tile; j += nthr) s[j] = in[base + (size_t)i*tile + j];
    __syncthreads();
    compute_tile<REPS>(s, out + base + (size_t)i*tile, tile, tid, nthr);
    __syncthreads();
  }
}

template<int REPS>
__global__ void pipelined_kernel(const float* in, float* out, int tiles_per_block, int tile) {
  extern __shared__ float s[];   // STAGES * tile floats
  int tid = threadIdx.x, nthr = blockDim.x;
  size_t base = (size_t)blockIdx.x * tiles_per_block * tile;
  int quads = tile / 4;
  for (int i = 0; i < tiles_per_block; ++i) {
    // issue async copy of tile i into stage i%STAGES (16-byte cp.async via float4)
    float4* sin4 = (float4*)&s[(i % STAGES) * tile];
    const float4* gin4 = (const float4*)&in[base + (size_t)i * tile];
    for (int q = tid; q < quads; q += nthr)
      __pipeline_memcpy_async(&sin4[q], &gin4[q], sizeof(float4));
    __pipeline_commit();
    if (i >= STAGES - 1) {
      __pipeline_wait_prior(STAGES - 1);   // oldest in-flight tile now done
      int c = i - (STAGES - 1);
      __syncthreads();                      // ensure copy landed in shared
      compute_tile<REPS>(&s[(c % STAGES) * tile], out + base + (size_t)c*tile, tile, tid, nthr);
      __syncthreads();                      // protect that stage before reuse
    }
  }
  // drain remaining STAGES-1 tiles (in issue order)
  for (int k = 1; k <= STAGES - 1; ++k) {
    __pipeline_wait_prior(STAGES - 1 - k);
    int c = tiles_per_block - (STAGES - 1) + (k - 1);
    __syncthreads();
    compute_tile<REPS>(&s[(c % STAGES) * tile], out + base + (size_t)c*tile, tile, tid, nthr);
    __syncthreads();
  }
}

template<int REPS>
static double run(bool piped, const float* in, float* out, int total_tiles, int tiles_per_block,
                  int tile, int threads, int blocks, cudaEvent_t s, cudaEvent_t e) {
  size_t smem = (piped ? STAGES : 1) * tile * sizeof(float);
  cudaEventRecord(s);
  if (piped) pipelined_kernel<REPS><<<blocks, threads, smem>>>(in, out, tiles_per_block, tile);
  else       serial_kernel<REPS><<<blocks, threads, smem>>>(in, out, tiles_per_block, tile);
  cudaEventRecord(e); cudaEventSynchronize(e);
  float ms = 0; cudaEventElapsedTime(&ms, s, e); return ms;
}

int main() {
  const int TILE = 4096, TPB = 16, THREADS = 256;
  const int BLOCKS = 132 * 8;                 // 8 blocks/SM on 132 SMs
  const int TOTAL_TILES = BLOCKS * TPB;       // 16896
  const size_t N = (size_t)TOTAL_TILES * TILE;
  printf("H200 cp.async overlap: STAGES=%d TILE=%d tiles/block=%d blocks=%d total_floats=%zu (~%.0fMB)\n",
         STAGES, TILE, TPB, BLOCKS, N, (double)N*4/1e6);
  std::vector<float> hin(N);
  for (size_t i = 0; i < N; ++i) hin[i] = (float)((i % 1000) - 500) / 500.0f;
  float *din, *doutA, *doutB;
  CUDA_CHECK(cudaMalloc(&din, N*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&doutA, N*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&doutB, N*sizeof(float)));
  CUDA_CHECK(cudaMemcpy(din, hin.data(), N*sizeof(float), cudaMemcpyHostToDevice));
  cudaEvent_t es, ee; cudaEventCreate(&es); cudaEventCreate(&ee);

  printf("%-6s %-8s %-10s %-10s %-9s %-10s\n","REPS","mode","min(ms)","med(ms)","speedup","agree");
  for (int REPS : {8, 32, 128, 512}) {
    // correctness: serial vs pipelined produce identical out
    cudaMemset(doutA, 0, N*sizeof(float)); cudaMemset(doutB, 0, N*sizeof(float));
    #define DISPATCH(R) case R: { run<R>(false,din,doutA,TOTAL_TILES,TPB,TILE,THREADS,BLOCKS,es,ee); \
                                   run<R>(true ,din,doutB,TOTAL_TILES,TPB,TILE,THREADS,BLOCKS,es,ee); break; }
    switch(REPS){ DISPATCH(8) DISPATCH(32) DISPATCH(128) DISPATCH(512) default: break; }
    #undef DISPATCH
    std::vector<float> ha(N), hb(N);
    CUDA_CHECK(cudaMemcpy(ha.data(), doutA, N*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hb.data(), doutB, N*sizeof(float), cudaMemcpyDeviceToHost));
    double maxd = 0; for (size_t i=0;i<N;++i){ double d=fabs((double)ha[i]-(double)hb[i]); if(d>maxd)maxd=d; }

    auto bench = [&](bool piped)->double{
      double best=1e9, sum=0; const int T=10;
      for(int t=0;t<T;++t){ double ms;
        switch(REPS){ case 8: ms=run<8>(piped,din,doutA,TOTAL_TILES,TPB,TILE,THREADS,BLOCKS,es,ee); break;
                       case 32: ms=run<32>(piped,din,doutA,TOTAL_TILES,TPB,TILE,THREADS,BLOCKS,es,ee); break;
                       case 128: ms=run<128>(piped,din,doutA,TOTAL_TILES,TPB,TILE,THREADS,BLOCKS,es,ee); break;
                       default: ms=run<512>(piped,din,doutA,TOTAL_TILES,TPB,TILE,THREADS,BLOCKS,es,ee); break; }
        best=std::min(best,ms); sum+=ms; }
      printf("%-6d %-8s %-10.4f %-10.4f\n", REPS, piped?"piped":"serial", best, sum/T);
      return best;
    };
    double bs = bench(false), bp = bench(true);
    printf("%-6d %-8s %-10.4f %-10.4f %-9.3f %-10.3e\n", REPS, "RESULT", bs, bp, bs/bp, maxd);
  }
  cudaFree(din); cudaFree(doutA); cudaFree(doutB);
  return 0;
}
