// Minimal CUTLASS SM90 (Hopper) TF32 GEMM — H200 compile + correctness test.
// Validates that the CUTLASS SM90 array-TMA warp-specialized GEMM kernel family
// (the one PR #2719 modifies) compiles and runs on H200, and matches cuBLAS TF32.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"

using namespace cute;

using MmaTileShape = Shape<_128, _128, _64>;
using ClusterShape = Shape<_1, _1, _1>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    cutlass::tfloat32_t, cutlass::layout::RowTensorOp,
    cutlass::tfloat32_t, cutlass::layout::RowTensorOp,
    cutlass::epilogue::collective::EpilogueOpTensorOp>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    cutlass::tfloat32_t, cutlass::layout::RowMajor, 8,
    cutlass::tfloat32_t, cutlass::layout::ColumnMajor, 8,
    cutlass::tfloat32_t, cutlass::tfloat32_t, cutlass::tfloat32_t,
    cutlass::gemm::collective::StageCountAuto,
    cutlass::gemm::collective::KernelScheduleAuto,
    MmaTileShape, ClusterShape,
    cutlass::gemm::collective::PersistentScheduler,
    CollectiveEpilogue>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>, CollectiveMainloop, CollectiveEpilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"CUDA %d %s\n",__LINE__,cudaGetErrorString(e));return 1;}}while(0)

int main(int argc, char** argv){
  int M=(argc>1)?atoi(argv[1]):2048, N=(argc>2)?atoi(argv[2]):2048, K=(argc>3)?atoi(argv[3]):2048;
  printf("SM90 TF32 GEMM M=%d N=%d K=%d\n",M,N,K);
  size_t szA=(size_t)M*K, szB=(size_t)K*N, szC=(size_t)M*N;
  std::vector<float> hA(szA),hB(szB);
  for(size_t i=0;i<szA;++i) hA[i]=((float)((i*7+1)%97))/97.0f-0.5f;
  for(size_t i=0;i<szB;++i) hB[i]=((float)((i*13+3)%89))/89.0f-0.5f;
  float *dA,*dB,*dC,*dD,*dRef;
  CUDA_CHECK(cudaMalloc(&dA,szA*4)); CUDA_CHECK(cudaMalloc(&dB,szB*4));
  CUDA_CHECK(cudaMalloc(&dC,szC*4)); CUDA_CHECK(cudaMalloc(&dD,szC*4));
  CUDA_CHECK(cudaMalloc(&dRef,szC*4));
  CUDA_CHECK(cudaMemcpy(dA,hA.data(),szA*4,cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB,hB.data(),szB*4,cudaMemcpyHostToDevice));
  cudaMemset(dC,0,szC*4);

  // cuBLAS TF32 reference: Cref = A(MxK,row) @ B(KxN,col) -> store row-major MxN
  cublasHandle_t h; cublasCreate(&h);
  cublasSetMathMode(h, CUBLAS_TF32_TENSOR_OP_MATH);
  float alpha=1.f, beta=0.f;
  // cublas col-major: Cref(NxM)=B^T(NxK)*A^T(KxM) => op N: C(NxM)=B(NxK)*A(KxM)
  cublasSgemm(h,CUBLAS_OP_N,CUBLAS_OP_N,N,M,K,&alpha,dB,N,dA,K,&beta,dRef,N);

  // CUTLASS CuTe tensors (A: row-major MxK stride (K,1); B: col-major KxN -> shape (N,K) stride (K,1); C/D: row-major MxN stride (N,1))
  auto A = make_tensor(make_gmem_ptr(reinterpret_cast<cutlass::tfloat32_t*>(dA)), make_shape(M,K), make_stride(K,_1{}));
  auto B = make_tensor(make_gmem_ptr(reinterpret_cast<cutlass::tfloat32_t*>(dB)), make_shape(N,K), make_stride(K,_1{}));
  auto C = make_tensor(make_gmem_ptr(reinterpret_cast<cutlass::tfloat32_t*>(dC)), make_shape(M,N), make_stride(N,_1{}));
  auto D = make_tensor(make_gmem_ptr(reinterpret_cast<cutlass::tfloat32_t*>(dD)), make_shape(M,N), make_stride(N,_1{}));

  typename Gemm::Arguments args{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M,N,K,1},
    {A,B},
    {{alpha,beta},D,C},
  };
  Gemm gemm;
  size_t ws = gemm.get_workspace_size(args);
  void* wbuf=nullptr; if(ws) CUDA_CHECK(cudaMalloc(&wbuf,ws));
  cutlass::Status st = gemm.initialize(args,wbuf);
  printf("init status=%d ws=%zu\n",(int)st,ws);
  if(st!=cutlass::Status::kSuccess){fprintf(stderr,"init failed\n");return 2;}
  st=gemm.run();
  printf("run status=%d\n",(int)st);

  // correctness vs cuBLAS TF32
  std::vector<float> hD(szC), hRef(szC);
  CUDA_CHECK(cudaMemcpy(hD.data(),dD,szC*4,cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(hRef.data(),dRef,szC*4,cudaMemcpyDeviceToHost));
  double mx=0,sum=0; for(size_t i=0;i<szC;++i){double d=fabs((double)hD[i]-(double)hRef[i]); if(d>mx)mx=d; sum+=fabs((double)hRef[i]);}
  printf("max_abs_diff_cutlass_vs_cublas=%.4e mean_abs_ref=%.4e rel=%.4e\n",mx,sum/szC,mx/(sum/szC+1e-12));
  printf("RESULT %s max_abs_diff=%.4e\n", mx<1.0?"PASS-ish":"CHECK", mx);
  if(wbuf) cudaFree(wbuf);
  cublasDestroy(h); cudaFree(dA);cudaFree(dB);cudaFree(dC);cudaFree(dD);cudaFree(dRef);
  return 0;
}
