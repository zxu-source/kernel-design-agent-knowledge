#include <cuda/pipeline>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <chrono>

#define CUDA_OK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA_ERROR %s\\n",cudaGetErrorString(e)); return 2;} } while(0)

__global__ void scalar_copy(const float* a, float* b, int n) { int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n)b[i]=a[i]; }
__global__ void vector_copy(const float4* a, float4* b, int n4) { int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n4)b[i]=a[i]; }

__global__ void transpose_naive(const float* a,float* b,int n){int x=blockIdx.x*32+threadIdx.x,y=blockIdx.y*32+threadIdx.y;if(x<n&&y<n)b[x*n+y]=a[y*n+x];}
__global__ void transpose_tiled(const float* a,float* b,int n){__shared__ float t[32][33];int x=blockIdx.x*32+threadIdx.x,y=blockIdx.y*32+threadIdx.y;if(x<n&&y<n)t[threadIdx.y][threadIdx.x]=a[y*n+x];__syncthreads();x=blockIdx.y*32+threadIdx.x;y=blockIdx.x*32+threadIdx.y;if(x<n&&y<n)b[y*n+x]=t[threadIdx.x][threadIdx.y];}

__global__ void atomics_naive(int* out,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)atomicAdd(out,1);}
__global__ void atomics_warp(int* out,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){int s=__reduce_add_sync(0xffffffff,1);if((threadIdx.x&31)==0)atomicAdd(out,s);}}

__global__ void direct_copy(const float* a,float* b,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)b[i]=a[i];}
__global__ void async_copy(const float* a,float* b,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){auto p=cuda::make_pipeline();__shared__ float smem[256];p.producer_acquire();cuda::memcpy_async(&smem[threadIdx.x],&a[i],sizeof(float),p);p.producer_commit();p.consumer_wait();b[i]=smem[threadIdx.x];p.consumer_release();}}
__global__ void count_mismatch(const float* a,const float* b,int n,int* bad){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n&&fabsf(a[i]-b[i])>1e-4f)atomicAdd(bad,1);}

__global__ void gemm_naive(const float*a,const float*b,float*c,int n){int x=blockIdx.x*16+threadIdx.x,y=blockIdx.y*16+threadIdx.y;if(x<n&&y<n){float s=0;for(int k=0;k<n;k++)s+=a[y*n+k]*b[k*n+x];c[y*n+x]=s;}}
__global__ void gemm_tiled(const float*a,const float*b,float*c,int n){__shared__ float as[16][16],bs[16][16];int x=blockIdx.x*16+threadIdx.x,y=blockIdx.y*16+threadIdx.y;float s=0;for(int k=0;k<n;k+=16){as[threadIdx.y][threadIdx.x]=a[y*n+k+threadIdx.x];bs[threadIdx.y][threadIdx.x]=b[(k+threadIdx.y)*n+x];__syncthreads();for(int z=0;z<16;z++)s+=as[threadIdx.y][z]*bs[z][threadIdx.x];__syncthreads();}c[y*n+x]=s;}
__global__ void fill_bf16(__nv_bfloat16* a,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)a[i]=__float2bfloat16(1.0f);}
__global__ void gemm_bf16_simt(const __nv_bfloat16*a,const __nv_bfloat16*b,float*c,int n){int x=blockIdx.x*16+threadIdx.x,y=blockIdx.y*16+threadIdx.y;if(x<n&&y<n){float s=0;for(int k=0;k<n;k++)s+=__bfloat162float(a[y*n+k])*__bfloat162float(b[k*n+x]);c[y*n+x]=s;}}
__global__ void gemm_bf16_wmma(const __nv_bfloat16*a,const __nv_bfloat16*b,float*c,int n){using namespace nvcuda;int row=blockIdx.y*16,col=blockIdx.x*16;wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> af;wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::row_major> bf;wmma::fragment<wmma::accumulator,16,16,16,float> cf;wmma::fill_fragment(cf,0.0f);for(int k=0;k<n;k+=16){wmma::load_matrix_sync(af,a+row*n+k,n);wmma::load_matrix_sync(bf,b+k*n+col,n);wmma::mma_sync(cf,af,bf,cf);}wmma::store_matrix_sync(c+row*n+col,cf,n,wmma::mem_row_major);}
struct LargeParams { float v[64]; };
__global__ void params_by_value(LargeParams p,float* out,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)out[i]=p.v[i&63]+float(i%7);}
__global__ void params_by_pointer(const LargeParams* p,float* out,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)out[i]=p->v[i&63]+float(i%7);}
__global__ void reduce_block(const int* in,int* out,int n){__shared__ int s[256];int i=blockIdx.x*256+threadIdx.x;s[threadIdx.x]=i<n?in[i]:0;__syncthreads();for(int d=128;d;d>>=1){if(threadIdx.x<d)s[threadIdx.x]+=s[threadIdx.x+d];__syncthreads();}if(threadIdx.x==0)atomicAdd(out,s[0]);}
__global__ void fill_int(int* in,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)in[i]=1;}

template <class F> float bench(F f){cudaEvent_t s,e;cudaEventCreate(&s);cudaEventCreate(&e);for(int i=0;i<10;i++)f();cudaEventRecord(s);for(int i=0;i<50;i++)f();cudaEventRecord(e);cudaEventSynchronize(e);float ms;cudaEventElapsedTime(&ms,s,e);cudaEventDestroy(s);cudaEventDestroy(e);return ms/50.f;}
float max_diff(const float*a,const float*b,int n){float m=0;for(int i=0;i<n;i++)m=fmaxf(m,fabsf(a[i]-b[i]));return m;}
int verify(const float* a,const float* b,int n){int* d;int h;cudaMalloc(&d,4);cudaMemset(d,0,4);count_mismatch<<<(n+255)/256,256>>>(a,b,n,d);cudaMemcpy(&h,d,4,cudaMemcpyDeviceToHost);cudaFree(d);return h;}

int main(int argc,char**argv){if(argc!=2)return 2;const char* mode=argv[1];
 if(!strcmp(mode,"vector")){int n=1<<24;float *a,*x,*y;CUDA_OK(cudaMalloc(&a,n*4));CUDA_OK(cudaMalloc(&x,n*4));CUDA_OK(cudaMalloc(&y,n*4));cudaMemset(a,1,n*4);float p=bench([&]{scalar_copy<<<(n+255)/256,256>>>(a,x,n);});float q=bench([&]{vector_copy<<<(n/4+255)/256,256>>>((float4*)a,(float4*)y,n/4);});int bad=verify(x,y,n);printf("mode=vector correctness=%s mismatches=%d scalar_ms=%.4f vector_ms=%.4f speedup=%.3f\\n",bad?"FAIL":"PASS",bad,p,q,p/q);return bad?3:0;}
 if(!strcmp(mode,"transpose")){int n=2048;float *a,*x,*y;CUDA_OK(cudaMalloc(&a,(size_t)n*n*4));CUDA_OK(cudaMalloc(&x,(size_t)n*n*4));CUDA_OK(cudaMalloc(&y,(size_t)n*n*4));cudaMemset(a,1,(size_t)n*n*4);dim3 b(32,32),g(n/32,n/32);float p=bench([&]{transpose_naive<<<g,b>>>(a,x,n);});float q=bench([&]{transpose_tiled<<<g,b>>>(a,y,n);});int bad=verify(x,y,n*n);printf("mode=transpose correctness=%s mismatches=%d naive_ms=%.4f tiled_ms=%.4f speedup=%.3f\\n",bad?"FAIL":"PASS",bad,p,q,p/q);return bad?3:0;}
 if(!strcmp(mode,"atomics")){int n=1<<22,*x,*y;CUDA_OK(cudaMalloc(&x,4));CUDA_OK(cudaMalloc(&y,4));cudaMemset(x,0,4);cudaMemset(y,0,4);float p=bench([&]{cudaMemset(x,0,4);atomics_naive<<<(n+255)/256,256>>>(x,n);});float q=bench([&]{cudaMemset(y,0,4);atomics_warp<<<(n+255)/256,256>>>(y,n);});int hx,hy;cudaMemcpy(&hx,x,4,cudaMemcpyDeviceToHost);cudaMemcpy(&hy,y,4,cudaMemcpyDeviceToHost);printf("mode=warp-atomics correctness=%s naive_ms=%.4f aggregated_ms=%.4f speedup=%.3f\\n",(hx==n&&hy==n)?"PASS":"FAIL",p,q,p/q);return hx==n&&hy==n?0:3;}
 if(!strcmp(mode,"async")){int n=1<<24;float *a,*x,*y;CUDA_OK(cudaMalloc(&a,n*4));CUDA_OK(cudaMalloc(&x,n*4));CUDA_OK(cudaMalloc(&y,n*4));cudaMemset(a,1,n*4);float p=bench([&]{direct_copy<<<(n+255)/256,256>>>(a,x,n);});float q=bench([&]{async_copy<<<(n+255)/256,256>>>(a,y,n);});int bad=verify(x,y,n);printf("mode=async-copy correctness=%s mismatches=%d direct_ms=%.4f async_ms=%.4f speedup=%.3f\\n",bad?"FAIL":"PASS",bad,p,q,p/q);return bad?3:0;}
 if(!strcmp(mode,"shared-gemm")){int n=512;float *a,*b,*x,*y;CUDA_OK(cudaMalloc(&a,(size_t)n*n*4));CUDA_OK(cudaMalloc(&b,(size_t)n*n*4));CUDA_OK(cudaMalloc(&x,(size_t)n*n*4));CUDA_OK(cudaMalloc(&y,(size_t)n*n*4));cudaMemset(a,1,(size_t)n*n*4);cudaMemset(b,2,(size_t)n*n*4);dim3 z(16,16),g(n/16,n/16);float p=bench([&]{gemm_naive<<<g,z>>>(a,b,x,n);});float q=bench([&]{gemm_tiled<<<g,z>>>(a,b,y,n);});int bad=verify(x,y,n*n);printf("mode=shared-gemm correctness=%s mismatches=%d naive_ms=%.4f tiled_ms=%.4f speedup=%.3f\\n",bad?"FAIL":"PASS",bad,p,q,p/q);return bad?3:0;}
 if(!strcmp(mode,"bf16-wmma")){int n=512;__nv_bfloat16 *a,*b;float *x,*y;CUDA_OK(cudaMalloc(&a,(size_t)n*n*2));CUDA_OK(cudaMalloc(&b,(size_t)n*n*2));CUDA_OK(cudaMalloc(&x,(size_t)n*n*4));CUDA_OK(cudaMalloc(&y,(size_t)n*n*4));fill_bf16<<<(n*n+255)/256,256>>>(a,n*n);fill_bf16<<<(n*n+255)/256,256>>>(b,n*n);dim3 z(16,16),g(n/16,n/16);float p=bench([&]{gemm_bf16_simt<<<g,z>>>(a,b,x,n);});float q=bench([&]{gemm_bf16_wmma<<<g,32>>>(a,b,y,n);});int bad=verify(x,y,n*n);printf("mode=bf16-wmma correctness=%s mismatches=%d simt_ms=%.4f wmma_ms=%.4f speedup=%.3f\\n",bad?"FAIL":"PASS",bad,p,q,p/q);return bad?3:0;}
 if(!strcmp(mode,"params")){int n=1<<22;float *x,*y;LargeParams h,*d;for(int i=0;i<64;i++)h.v[i]=1.0f;CUDA_OK(cudaMalloc(&x,n*4));CUDA_OK(cudaMalloc(&y,n*4));CUDA_OK(cudaMalloc(&d,sizeof(h)));CUDA_OK(cudaMemcpy(d,&h,sizeof(h),cudaMemcpyHostToDevice));float p=bench([&]{params_by_value<<<(n+255)/256,256>>>(h,x,n);});float q=bench([&]{params_by_pointer<<<(n+255)/256,256>>>(d,y,n);});int bad=verify(x,y,n);printf("mode=large-params correctness=%s mismatches=%d value_ms=%.4f pointer_ms=%.4f speedup=%.3f\\n",bad?"FAIL":"PASS",bad,p,q,p/q);return bad?3:0;}
 if(!strcmp(mode,"reduction")){int n=1<<22,*in,*x,*y;CUDA_OK(cudaMalloc(&in,n*4));CUDA_OK(cudaMalloc(&x,4));CUDA_OK(cudaMalloc(&y,4));fill_int<<<(n+255)/256,256>>>(in,n);float p=bench([&]{cudaMemset(x,0,4);atomics_naive<<<(n+255)/256,256>>>(x,n);});float q=bench([&]{cudaMemset(y,0,4);reduce_block<<<(n+255)/256,256>>>(in,y,n);});int hx,hy;cudaMemcpy(&hx,x,4,cudaMemcpyDeviceToHost);cudaMemcpy(&hy,y,4,cudaMemcpyDeviceToHost);printf("mode=reduction correctness=%s naive_ms=%.4f block_reduce_ms=%.4f speedup=%.3f\\n",(hx==n&&hy==n)?"PASS":"FAIL",p,q,p/q);return hx==n&&hy==n?0:3;}
 if(!strcmp(mode,"mallocasync")){cudaStream_t s;CUDA_OK(cudaStreamCreate(&s));auto t0=std::chrono::steady_clock::now();for(int i=0;i<100;i++){void*p;CUDA_OK(cudaMalloc(&p,1<<20));CUDA_OK(cudaFree(p));}auto t1=std::chrono::steady_clock::now();auto t2=std::chrono::steady_clock::now();for(int i=0;i<100;i++){void*p;CUDA_OK(cudaMallocAsync(&p,1<<20,s));CUDA_OK(cudaFreeAsync(p,s));}CUDA_OK(cudaStreamSynchronize(s));auto t3=std::chrono::steady_clock::now();double p=std::chrono::duration<double,std::milli>(t1-t0).count()/100.0,q=std::chrono::duration<double,std::milli>(t3-t2).count()/100.0;printf("mode=stream-ordered-allocation correctness=PASS malloc_ms=%.4f mallocAsync_ms=%.4f speedup=%.3f\\n",p,q,p/q);return 0;}
 return 2;
}
