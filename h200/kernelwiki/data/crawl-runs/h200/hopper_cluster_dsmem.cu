// Hopper thread-block-cluster Distributed Shared Memory (DSMEM) broadcast on H200.
// Within a cluster of CLUSTER blocks, block 0 holds a source tile in its shared
// memory; the other blocks acquire it either (a) via DSMEM
// (cooperative_groups::cluster_group::map_shared_rank -> direct cross-SM shared
// read) or (b) via a global-memory bounce buffer (block 0 writes, cluster sync,
// peers read). Both produce identical output; we compare latency of the fan-out.
// Demonstrates the Hopper cluster / DSMEM feature correctness + the DSMEM-vs-HBM
// tradeoff for inter-block data exchange.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#define CUDA_CHECK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"CUDA %d %s\n",__LINE__,cudaGetErrorString(e));return 1;}}while(0)

#ifndef CLUSTER
#define CLUSTER 4
#endif
#ifndef TILE
#define TILE 2048          // floats per block shared (8 KB)
#endif
#ifndef THREADS
#define THREADS 256
#endif
#ifndef N_CLUSTERS
#define N_CLUSTERS (132 * 4)
#endif

__global__ void dsmem_broadcast(const float* src, float* out, float* gbuf, int use_dsmem, int* dbg_nb) {
  __shared__ float sbuf[TILE];
  cg::cluster_group cluster = cg::this_cluster();
  int crank = cluster.block_rank();          // 0..CLUSTER-1 within cluster
  int gblock = blockIdx.x;                   // global block id
  int cluster_id = gblock / CLUSTER;
  int tid = threadIdx.x;
  if (gblock == 0 && tid == 0) *dbg_nb = cluster.num_blocks();
  size_t src_off = (size_t)cluster_id * TILE;       // source tile for this cluster
  size_t out_off = (size_t)gblock * TILE;            // each block writes its own tile
  // block 0 of the cluster loads the source tile into its shared memory
  if (crank == 0) {
    for (int j = tid; j < TILE; j += THREADS) sbuf[j] = src[src_off + j];
  }
  __syncthreads();
  cluster.sync();    // block 0's shared tile is now visible cluster-wide

  if (use_dsmem) {
    // PUSH model: block 0 writes its tile into every peer's shared via DSMEM.
    if (crank == 0) {
      for (int r = 0; r < CLUSTER; ++r) {
        float* peer = cluster.map_shared_rank(sbuf, r);  // peer r's sbuf (r==0 -> own)
        for (int j = tid; j < TILE; j += THREADS) peer[j] = sbuf[j];
      }
    }
    cluster.sync();   // block 0 must finish DSMEM writes before any peer reads/exits
  } else {
    // global bounce: block 0 writes its tile to a per-cluster global buffer,
    // peers read it back
    if (crank == 0)
      for (int j = tid; j < TILE; j += THREADS) gbuf[src_off + j] = sbuf[j];
    cluster.sync();
    for (int j = tid; j < TILE; j += THREADS) sbuf[j] = gbuf[src_off + j];
  }
  __syncthreads();
  // every block writes its (now-filled) tile to out
  for (int j = tid; j < TILE; j += THREADS) out[out_off + j] = sbuf[j];
}

static double launch(const float* src, float* out, float* gbuf, int total_blocks, int use_dsmem, int* dbg_nb, const char* tag, cudaEvent_t s, cudaEvent_t e) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(total_blocks, 1, 1);
  cfg.blockDim = dim3(THREADS, 1, 1);
  cfg.stream = 0;
  cudaLaunchAttribute attr;
  attr.id = cudaLaunchAttributeClusterDimension;
  attr.val.clusterDim.x = CLUSTER; attr.val.clusterDim.y = 1; attr.val.clusterDim.z = 1;
  cfg.attrs = &attr; cfg.numAttrs = 1;
  cudaGetLastError();
  cudaEventRecord(s);
  cudaError_t le = cudaLaunchKernelEx(&cfg, dsmem_broadcast, src, out, gbuf, use_dsmem, dbg_nb);
  cudaEventRecord(e); cudaEventSynchronize(e);
  cudaError_t se = cudaDeviceSynchronize();
  if (le != cudaSuccess) printf("[%s] launch err=%d %s\n", tag, (int)le, cudaGetErrorString(le));
  if (se != cudaSuccess) printf("[%s] sync err=%d %s\n", tag, (int)se, cudaGetErrorString(se));
  float ms = 0; cudaEventElapsedTime(&ms, s, e); return ms;
}

int main() {
  const int TOTAL_BLOCKS = N_CLUSTERS * CLUSTER;
  const size_t SRC = (size_t)N_CLUSTERS * TILE;          // one tile per cluster
  const size_t OUT = (size_t)TOTAL_BLOCKS * TILE;        // one tile per block
  printf("H200 DSMEM broadcast: CLUSTER=%d TILE=%d THREADS=%d clusters=%d blocks=%d src_floats=%zu out_floats=%zu\n",
         CLUSTER, TILE, THREADS, N_CLUSTERS, TOTAL_BLOCKS, SRC, OUT);
  std::vector<float> hsrc(SRC);
  for (size_t i = 0; i < SRC; ++i) hsrc[i] = (float)((i * 31 + 7) % 1009) / 17.0f;
  float *dsrc, *dout, *dout2, *dgbuf;
  int *ddbg, hdbg = -1;
  CUDA_CHECK(cudaMalloc(&dsrc, SRC * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dout, OUT * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dout2, OUT * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dgbuf, SRC * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&ddbg, sizeof(int)));
  CUDA_CHECK(cudaMemcpy(dsrc, hsrc.data(), SRC * sizeof(float), cudaMemcpyHostToDevice));
  cudaEvent_t es, ee; cudaEventCreate(&es); cudaEventCreate(&ee);

  // correctness: run GLOBAL path FIRST (isolates cluster launch from DSMEM access).
  // If global faults too, cluster launch itself is the problem.
  launch(dsrc, dout2, dgbuf, TOTAL_BLOCKS, 0, ddbg, "global", es, ee);  // global (cluster launch only)
  CUDA_CHECK(cudaMemcpy(&hdbg, ddbg, sizeof(int), cudaMemcpyDeviceToHost));
  printf("cluster.num_blocks() reported by block0 = %d (CLUSTER=%d)\n", hdbg, CLUSTER);
  launch(dsrc, dout, dgbuf, TOTAL_BLOCKS, 1, ddbg, "dsmem", es, ee);   // dsmem
  std::vector<float> h1(OUT), h2(OUT);
  CUDA_CHECK(cudaMemcpy(h1.data(), dout, OUT * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h2.data(), dout2, OUT * sizeof(float), cudaMemcpyDeviceToHost));
  double maxd_dvsg = 0, maxerr = 0;
  for (int b = 0; b < TOTAL_BLOCKS; ++b) {
    int cluster_id = b / CLUSTER;
    for (int j = 0; j < TILE; ++j) {
      float exp = hsrc[cluster_id * TILE + j];
      maxd_dvsg = std::max(maxd_dvsg, std::fabs((double)h1[b*TILE+j] - (double)h2[b*TILE+j]));
      maxerr = std::max(maxerr, std::fabs((double)h1[b*TILE+j] - exp));
    }
  }
  printf("correctness: |dsmem-global| max=%.3e, |dsmem-src| max=%.3e\n", maxd_dvsg, maxerr);

  auto bench = [&](int u, const char* tag)->double{
    double best = 1e9, sum = 0; const int T = 20;
    for (int t = 0; t < T; ++t) { double ms = launch(dsrc, dout, dgbuf, TOTAL_BLOCKS, u, ddbg, tag, es, ee);
      best = std::min(best, ms); sum += ms; }
    printf("%-7s min=%.4fms med=%.4fms\n", tag, best, sum / T); return best;
  };
  double bg = bench(0, "global"), bd = bench(1, "dsmem");
  printf("RESULT global=%.4fms dsmem=%.4fms global/dsmem=%.3fx (dsmem fan-out vs global bounce)\n", bg, bd, bg / bd);

  cudaFree(dsrc); cudaFree(dout); cudaFree(dout2); cudaFree(dgbuf); cudaFree(ddbg);
  return 0;
}
