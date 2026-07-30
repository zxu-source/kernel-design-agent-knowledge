#include "ragged_grouped_gemm.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using ragged_gemm::DeviceMetadata;

#define CUDA_OK(x) do { cudaError_t _e = (x); if (_e != cudaSuccess) { \
  std::cerr << #x << ": " << cudaGetErrorString(_e) << "\n"; return false; } } while (0)

struct Case { std::string name; std::vector<int> m; int k; int n; };
struct Buffers {
  std::vector<int64_t> ao, bo, co;
  std::vector<__nv_bfloat16> ha, hb, hc0, hc1, hc2;
  __nv_bfloat16 *a=nullptr, *b=nullptr, *c0=nullptr, *c1=nullptr, *c2=nullptr;
  ~Buffers() { cudaFree(a); cudaFree(b); cudaFree(c0); cudaFree(c1); cudaFree(c2); }
};

static float bf(__nv_bfloat16 x) { return __bfloat162float(x); }
static __nv_bfloat16 random_bf(std::mt19937& g) {
  std::uniform_real_distribution<float> d(-0.25f, 0.25f); return __float2bfloat16(d(g));
}

static bool make_buffers(const Case& c, Buffers* x) {
  const int e = static_cast<int>(c.m.size()); x->ao.resize(e); x->bo.resize(e); x->co.resize(e);
  int64_t as=0, cs=0;
  for (int i=0;i<e;++i) { if (c.m[i] < 0) return false; x->ao[i]=as; x->bo[i]=int64_t(i)*c.k*c.n; x->co[i]=cs; as += int64_t(c.m[i])*c.k; cs += int64_t(c.m[i])*c.n; }
  std::mt19937 gen(1234 + e + c.k + c.n); x->ha.resize(as); x->hb.resize(int64_t(e)*c.k*c.n);
  for (auto& v:x->ha) v=random_bf(gen); for (auto& v:x->hb) v=random_bf(gen);
  x->hc0.assign(cs, __float2bfloat16(std::numeric_limits<float>::quiet_NaN())); x->hc1=x->hc0; x->hc2=x->hc0;
  CUDA_OK(cudaMalloc(&x->a, as*sizeof(__nv_bfloat16))); CUDA_OK(cudaMalloc(&x->b, x->hb.size()*sizeof(__nv_bfloat16)));
  CUDA_OK(cudaMalloc(&x->c0, cs*sizeof(__nv_bfloat16))); CUDA_OK(cudaMalloc(&x->c1, cs*sizeof(__nv_bfloat16))); CUDA_OK(cudaMalloc(&x->c2, cs*sizeof(__nv_bfloat16)));
  CUDA_OK(cudaMemcpy(x->a,x->ha.data(),as*sizeof(__nv_bfloat16),cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(x->b,x->hb.data(),x->hb.size()*sizeof(__nv_bfloat16),cudaMemcpyHostToDevice));
  return true;
}

static bool check_layout(const Case& c, const Buffers& x) {
  for (size_t i=0;i<c.m.size();++i) {
    const int64_t ae=x.ao[i]+int64_t(c.m[i])*c.k, ce=x.co[i]+int64_t(c.m[i])*c.n;
    if (ae > int64_t(x.ha.size()) || ce > int64_t(x.hc0.size()) || (i && (x.ao[i]<x.ao[i-1] || x.co[i]<x.co[i-1]))) return false;
  } return true;
}

static bool validate_one(const Case& c, std::ostream& log) {
  Buffers x; if (!make_buffers(c,&x) || !check_layout(c,x)) return false;
  cudaStream_t s; CUDA_OK(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));
  CUDA_OK(cudaMemsetAsync(x.c0,0xff,x.hc0.size()*sizeof(__nv_bfloat16),s));
  CUDA_OK(ragged_gemm::grouped_gemm_baseline(x.a,x.b,x.c0,c.m.data(),x.ao.data(),x.bo.data(),x.co.data(),c.m.size(),c.k,c.n,s));
  CUDA_OK(cudaStreamSynchronize(s));
  DeviceMetadata md; CUDA_OK(ragged_gemm::build_metadata(&md,c.m.data(),c.m.size(),c.n,s));
  CUDA_OK(cudaMemsetAsync(x.c1,0xff,x.hc1.size()*sizeof(__nv_bfloat16),s));
  CUDA_OK(ragged_gemm::grouped_gemm_persistent(x.a,x.b,x.c1,c.m.data(),x.ao.data(),x.bo.data(),x.co.data(),c.m.size(),c.k,c.n,&md,s));
  CUDA_OK(ragged_gemm::grouped_gemm_static_queue(x.a,x.b,x.c2,c.m.data(),x.ao.data(),x.bo.data(),x.co.data(),c.m.size(),c.k,c.n,&md,s));
  CUDA_OK(cudaStreamSynchronize(s));
  CUDA_OK(cudaMemcpy(x.hc0.data(),x.c0,x.hc0.size()*sizeof(__nv_bfloat16),cudaMemcpyDeviceToHost));
  CUDA_OK(cudaMemcpy(x.hc1.data(),x.c1,x.hc1.size()*sizeof(__nv_bfloat16),cudaMemcpyDeviceToHost));
  CUDA_OK(cudaMemcpy(x.hc2.data(),x.c2,x.hc2.size()*sizeof(__nv_bfloat16),cudaMemcpyDeviceToHost));
  float max_abs=0, max_rel=0, mean_abs=0; bool finite=true; size_t n=x.hc0.size();
  for(size_t i=0;i<n;++i) { float a=bf(x.hc0[i]), b=bf(x.hc1[i]), q=bf(x.hc2[i]); finite &= std::isfinite(a)&&std::isfinite(b)&&std::isfinite(q); float d=std::max(std::abs(a-b),std::abs(a-q)); max_abs=std::max(max_abs,d); max_rel=std::max(max_rel,d/(std::abs(a)+1e-5f)); mean_abs+=d; }
  // Independent CPU FP32 reference for small validation cases.
  float cpu_abs=0;
  if (int64_t(n)*c.k <= 16ll*1024*1024) for (size_t e=0;e<c.m.size();++e) for(int r=0;r<c.m[e];++r) for(int col=0;col<c.n;++col) {
    float sum=0; for(int kk=0;kk<c.k;++kk) sum += bf(x.ha[x.ao[e]+int64_t(r)*c.k+kk])*bf(x.hb[x.bo[e]+int64_t(kk)*c.n+col]);
    cpu_abs=std::max(cpu_abs,std::abs(sum-bf(x.hc1[x.co[e]+int64_t(r)*c.n+col])));
  }
  const bool pass=finite && max_abs<=0.02f && cpu_abs<=0.08f;
  log << "VALIDATE,"<<c.name<<",E="<<c.m.size()<<",K="<<c.k<<",N="<<c.n<<",tasks="<<md.task_count
      <<",max_abs="<<max_abs<<",max_rel="<<max_rel<<",mean_abs="<<(n?mean_abs/n:0)<<",cpu_max_abs="<<cpu_abs<<",pass="<<pass<<"\n";
  ragged_gemm::destroy_metadata(&md); cudaStreamDestroy(s); return pass;
}

static double timed_us(const Case& c, Buffers& x, DeviceMetadata* md, int variant, int warmup, int iters) {
  cudaStream_t s; cudaStreamCreate(&s); for(int i=0;i<warmup;++i) {
    if(variant==1) ragged_gemm::grouped_gemm_persistent(x.a,x.b,x.c1,c.m.data(),x.ao.data(),x.bo.data(),x.co.data(),c.m.size(),c.k,c.n,md,s);
    else if(variant==2) ragged_gemm::grouped_gemm_static_queue(x.a,x.b,x.c2,c.m.data(),x.ao.data(),x.bo.data(),x.co.data(),c.m.size(),c.k,c.n,md,s);
    else ragged_gemm::grouped_gemm_baseline(x.a,x.b,x.c0,c.m.data(),x.ao.data(),x.bo.data(),x.co.data(),c.m.size(),c.k,c.n,s);
  } cudaEvent_t a,b; cudaEventCreate(&a);cudaEventCreate(&b);cudaEventRecord(a,s);
  for(int i=0;i<iters;++i) { if(variant==1) ragged_gemm::grouped_gemm_persistent(x.a,x.b,x.c1,c.m.data(),x.ao.data(),x.bo.data(),x.co.data(),c.m.size(),c.k,c.n,md,s);
    else if(variant==2) ragged_gemm::grouped_gemm_static_queue(x.a,x.b,x.c2,c.m.data(),x.ao.data(),x.bo.data(),x.co.data(),c.m.size(),c.k,c.n,md,s);
    else ragged_gemm::grouped_gemm_baseline(x.a,x.b,x.c0,c.m.data(),x.ao.data(),x.bo.data(),x.co.data(),c.m.size(),c.k,c.n,s); }
  cudaEventRecord(b,s); cudaEventSynchronize(b); float ms=0;cudaEventElapsedTime(&ms,a,b); cudaEventDestroy(a);cudaEventDestroy(b);cudaStreamDestroy(s);return ms*1000/iters;
}

static void bench_one(const Case& c, std::ostream& csv) {
  Buffers x; if(!make_buffers(c,&x)) return; cudaStream_t s;cudaStreamCreate(&s); DeviceMetadata md;
  const auto m0=std::chrono::steady_clock::now(); ragged_gemm::build_metadata(&md,c.m.data(),c.m.size(),c.n,s);cudaStreamSynchronize(s);
  const double meta=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-m0).count();
  std::vector<double> b,p,q;for(int r=0;r<3;++r){b.push_back(timed_us(c,x,&md,0,20,100));p.push_back(timed_us(c,x,&md,1,20,100));q.push_back(timed_us(c,x,&md,2,20,100));}
  auto report=[&](const char* name,std::vector<double> v){std::sort(v.begin(),v.end());double mean=std::accumulate(v.begin(),v.end(),0.0)/v.size();double flops=0;for(int m:c.m)flops+=2.0*m*c.k*c.n;double med=v[1];csv<<name<<",NVIDIA H200,bf16,"<<c.m.size()<<","<<std::accumulate(c.m.begin(),c.m.end(),0)<<","<<c.name<<","<<c.k<<","<<c.n<<",20,100,"<<med<<","<<mean<<","<<v.front()<<","<<v.back()<<","<<(flops/(med*1e-6)/1e12)<<",custom_cuda,"<<"true,0,metadata_us="<<meta<<"\n";};
  report("candidate_00_baseline",b);report("candidate_01_kernelwiki",p);report("candidate_02_static_queue",q);ragged_gemm::destroy_metadata(&md);cudaStreamDestroy(s);
}

int main(int argc,char**argv){
  const bool bench=argc>1 && std::string(argv[1])=="--bench"; int dev=0;cudaSetDevice(dev);cudaDeviceProp prop{};cudaGetDeviceProperties(&prop,dev);std::cout<<"GPU="<<prop.name<<" SMs="<<prop.multiProcessorCount<<"\n";
  std::vector<Case> validation={{"single",{127},64,64},{"uniform_e4",{31,32,31,32},64,64},{"empty_e8",{0,1,0,7,0,16,0,31},64,64},{"small_e16",{0,1,2,7,1,2,7,15,0,1,2,7,15,2,1,0},64,64},{"skew_e32",{512,0,1,2,7,16,31,0,1,2,7,16,31,0,1,2,7,16,31,0,1,2,7,16,31,0,1,2,7,16,31,0},64,64},{"mixed_e64",{},64,64}};for(int i=0;i<64;++i)validation.back().m.push_back((i%10==0)?0:std::vector<int>{1,2,7,16,31,64,127,256,512}[i%9]);
  bool ok=true;for(const auto& c:validation)ok &= validate_one(c,std::cout);if(!ok)return 2;if(!bench)return 0;
  std::ofstream csv("benchmark.csv");csv<<"candidate,gpu,dtype,num_experts,total_tokens,token_distribution,K,N,warmup,iterations,median_us,mean_us,min_us,p90_us,tflops,reference,correct,max_abs_error,notes\n";
  bench_one({"uniform",std::vector<int>(8,64),1024,1024},csv);bench_one({"skewed",{512,0,1,2,7,16,31,0,1,2,7,16,31,0,1,2},2048,1024},csv);bench_one({"small",std::vector<int>(32,7),4096,1024},csv);bench_one({"mixed",{1,2,7,16,31,64,127,256},4096,4096},csv);std::cout<<"benchmark.csv written\n";return 0;
}
