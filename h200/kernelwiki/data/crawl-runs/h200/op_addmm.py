#!/usr/bin/env python3
"""H200 Fused AddMM (alpha*A@B + beta*C) (Triton) vs torch.addmm.
GEMM accumulate, epilogue acc = alpha*acc + beta*C (residual add fused into the
GEMM epilogue, vs a separate add). PURPOSE = speedup (epilogue fusion).
vs torch.addmm (cuBLAS, already fused) on H200, Triton 3.6.
"""
import json, statistics
import torch, triton, triton.language as tl
def cdiv(a,b): return (a+b-1)//b
@triton.jit
def addmm(a_ptr,b_ptr,c_ptr,o_ptr,alpha,beta,M,N,K,sam,sak,sbk,sbn,scm,scn,
          BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    pm=tl.program_id(0); pn=tl.program_id(1)
    om=pm*BM+tl.arange(0,BM); on=pn*BN+tl.arange(0,BN); mask=(om[:,None]<M)&(on[None,:]<N)
    acc=tl.zeros((BM,BN),dtype=tl.float32)
    for k0 in range(0,K,BK):
        ok=k0+tl.arange(0,BK)
        a=tl.load(a_ptr+om[:,None]*sam+ok[None,:]*sak,mask=(om[:,None]<M)&(ok[None,:]<K),other=0.0)
        b=tl.load(b_ptr+ok[:,None]*sbk+on[None,:]*sbn,mask=(ok[:,None]<K)&(on[None,:]<N),other=0.0)
        acc+=tl.dot(a,b)
    c=tl.load(c_ptr+om[:,None]*scm+on[None,:]*scn, mask=mask, other=0.0).to(tl.float32)
    out=alpha*acc + beta*c
    tl.store(o_ptr+om[:,None]*scm+on[None,:]*scn, out.to(o_ptr.dtype.element_ty), mask=mask)
def tri_addmm(A,B,C,alpha=1.0,beta=1.0,BM=128,BN=256,BK=64,nw=8,ns=3):
    M,K=A.shape; _,N=B.shape; o=torch.empty((M,N),device=A.device,dtype=A.dtype)
    addmm[(cdiv(M,BM),cdiv(N,BN))](A,B,C,o,alpha,beta,M,N,K,A.stride(0),1,B.stride(0),B.stride(1),C.stride(0),C.stride(1),BM=BM,BN=BN,BK=BK,num_warps=nw,num_stages=ns); return o
def time_fn(fn,trials=20):
    for _ in range(5): fn(); torch.cuda.synchronize()
    ts=[]
    for _ in range(trials):
        s,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    return min(ts), statistics.median(ts)
def main():
    torch.backends.cuda.matmul.allow_tf32=False
    print(f"triton={triton.__version__} dev={torch.cuda.get_device_name(0)} SMs={torch.cuda.get_device_properties(0).multi_processor_count}")
    rows=[]
    for M,N,K in [(2048,2048,2048),(4096,4096,4096),(8192,8192,4096),(8192,8192,8192),(8192,11008,4096)]:
        A=torch.randn(M,K,device="cuda",dtype=torch.bfloat16)*0.3; B=torch.randn(K,N,device="cuda",dtype=torch.bfloat16)*0.3; C=torch.randn(M,N,device="cuda",dtype=torch.bfloat16)*0.1
        ref=torch.addmm(C, A, B, alpha=1.0, beta=1.0)
        out=tri_addmm(A,B,C)
        err=(out.float()-ref.float()).abs().max().item()
        tri=lambda: tri_addmm(A,B,C); tor=lambda: torch.addmm(C,A,B,alpha=1.0,beta=1.0)
        tm,_=time_fn(tri); tm_t,_=time_fn(tor)
        fl=2*M*N*K; tt=fl/(tm*1e-3)/1e12; ct=fl/(tm_t*1e-3)/1e12
        rows.append(dict(M=M,N=N,K=K,err=round(err,3),triton_ms=round(tm,4),torch_ms=round(tm_t,4),triton_tf=round(tt,0),torch_tf=round(ct,0),torch_over_triton=round(tm_t/tm,3)))
        print(f"M={M} N={N} K={K} err={err:.3f} triton={tm:.4f}ms({tt:.0f}TF) torch={tm_t:.4f}ms({ct:.0f}TF) torch/triton={tm_t/tm:.2f}x")
    print("RESULT_JSON", json.dumps(rows))
if __name__=="__main__": main()
