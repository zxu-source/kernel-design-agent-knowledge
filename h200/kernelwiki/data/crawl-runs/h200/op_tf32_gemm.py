#!/usr/bin/env python3
"""H200 TF32 GEMM (Triton) vs cuBLAS characterization.
TF32 (fp32 inputs, TF32 tensor cores). Triton tl.dot with fp32 inputs (default
TF32 rounding on Hopper) vs torch.matmul with allow_tf32=True (cuBLAS TF32).
PURPOSE = speedup baseline. H200, Triton 3.6. TF32 peak ~989 TF (same MMA rate
as bf16 on Hopper).
"""
import json, statistics
import torch, triton, triton.language as tl
def cdiv(a,b): return (a+b-1)//b
@triton.jit
def matmul_tf32(a_ptr,b_ptr,c_ptr,M,N,K,sam,sak,sbk,sbn,scm,scn,
                BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    pm=tl.program_id(0); pn=tl.program_id(1)
    om=pm*BM+tl.arange(0,BM); on=pn*BN+tl.arange(0,BN)
    acc=tl.zeros((BM,BN),dtype=tl.float32)
    for k0 in range(0,K,BK):
        ok=k0+tl.arange(0,BK)
        a=tl.load(a_ptr+om[:,None]*sam+ok[None,:]*sak,mask=(om[:,None]<M)&(ok[None,:]<K),other=0.0)
        b=tl.load(b_ptr+ok[:,None]*sbk+on[None,:]*sbn,mask=(ok[:,None]<K)&(on[None,:]<N),other=0.0)
        acc+=tl.dot(a,b,input_precision='tf32')   # fp32 inputs -> TF32 tensor cores
    tl.store(c_ptr+om[:,None]*scm+on[None,:]*scn,acc,mask=(om[:,None]<M)&(on[None,:]<N))
def tri_mm(a,b,BM=128,BN=128,BK=32,nw=4,ns=3):
    M,K=a.shape; _,N=b.shape; c=torch.empty((M,N),device=a.device,dtype=torch.float32)
    matmul_tf32[(cdiv(M,BM),cdiv(N,BN))](a,b,c,M,N,K,a.stride(0),a.stride(1),b.stride(0),b.stride(1),c.stride(0),c.stride(1),BM=BM,BN=BN,BK=BK,num_warps=nw,num_stages=ns); return c
def time_fn(fn,trials=20):
    for _ in range(5): fn(); torch.cuda.synchronize()
    ts=[]
    for _ in range(trials):
        s,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    return min(ts), statistics.median(ts)
def main():
    torch.backends.cuda.matmul.allow_tf32=True
    print(f"triton={triton.__version__} dev={torch.cuda.get_device_name(0)} SMs={torch.cuda.get_device_properties(0).multi_processor_count} tf32_peak~989TF")
    rows=[]
    for M,N,K in [(1024,1024,1024),(2048,2048,2048),(4096,4096,4096),(8192,8192,4096),(8192,8192,8192)]:
        a=torch.randn(M,K,device="cuda",dtype=torch.float32)*0.5; b=torch.randn(K,N,device="cuda",dtype=torch.float32)*0.5
        ref=a@b   # TF32 (allow_tf32=True)
        out=tri_mm(a,b)
        err=(out-ref).abs().max().item()
        tri=lambda: tri_mm(a,b); cub=lambda: a@b
        tm,_=time_fn(tri); cm,_=time_fn(cub)
        fl=2*M*N*K; tt=fl/(tm*1e-3)/1e12; ct=fl/(cm*1e-3)/1e12
        rows.append(dict(M=M,N=N,K=K,err=round(err,3),triton_ms=round(tm,4),cublas_ms=round(cm,4),triton_tf=round(tt,0),cublas_tf=round(ct,0),cub_over_tri=round(cm/tm,3),tri_util=round(tt/989*100,1)))
        print(f"M={M} N={N} K={K} err={err:.3f} triton={tm:.4f}ms({tt:.0f}TF,{tt/989*100:.0f}%) cublas={cm:.4f}ms({ct:.0f}TF,{ct/989*100:.0f}%) cublas/triton={cm/tm:.2f}x")
    print("RESULT_JSON", json.dumps(rows))
if __name__=="__main__": main()
