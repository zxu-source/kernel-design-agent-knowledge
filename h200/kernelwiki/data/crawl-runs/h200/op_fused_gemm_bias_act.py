#!/usr/bin/env python3
"""H200 Fused GEMM+Bias+SiLU (Triton) validation.
Epilogue fusion: bf16 GEMM accumulates to fp32, then add per-row bias and apply
SiLU in the same kernel (one read of A/B, one write of out) vs torch sequential
(a@b -> +bias -> silu = 3 launches + 2 intermediates). PURPOSE = speedup.
vs torch on H200, Triton 3.6.
"""
import json, statistics
import torch, triton, triton.language as tl
def cdiv(a,b): return (a+b-1)//b
@triton.jit
def gemm_bias_silu(a_ptr,b_ptr,bias_ptr,o_ptr,M,N,K,sam,sak,sbk,sbn,scm,scn,
                   BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    pm=tl.program_id(0); pn=tl.program_id(1)
    om=pm*BM+tl.arange(0,BM); on=pn*BN+tl.arange(0,BN)
    acc=tl.zeros((BM,BN),dtype=tl.float32)
    for k0 in range(0,K,BK):
        ok=k0+tl.arange(0,BK)
        a=tl.load(a_ptr+om[:,None]*sam+ok[None,:]*sak,mask=(om[:,None]<M)&(ok[None,:]<K),other=0.0)
        b=tl.load(b_ptr+ok[:,None]*sbk+on[None,:]*sbn,mask=(ok[:,None]<K)&(on[None,:]<N),other=0.0)
        acc+=tl.dot(a,b)
    bias=tl.load(bias_ptr+on, mask=on<N, other=0.0).to(tl.float32)
    acc=acc+bias[None,:]
    silu=acc/(1.0+tl.exp(-acc))    # fused activation
    tl.store(o_ptr+om[:,None]*scm+on[None,:]*scn, silu.to(o_ptr.dtype.element_ty), mask=(om[:,None]<M)&(on[None,:]<N))
def tri_fused(a,b,bias,BM=128,BN=256,BK=64,nw=8,ns=3):
    M,K=a.shape; _,N=b.shape; o=torch.empty((M,N),device=a.device,dtype=torch.bfloat16)
    gemm_bias_silu[(cdiv(M,BM),cdiv(N,BN))](a,b,bias,o,M,N,K,a.stride(0),a.stride(1),b.stride(0),b.stride(1),o.stride(0),o.stride(1),BM=BM,BN=BN,BK=BK,num_warps=nw,num_stages=ns); return o
def torch_seq(a,b,bias):
    g=(a@b); return torch.nn.functional.silu(g+bias[None,:])
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
        a=torch.randn(M,K,device="cuda",dtype=torch.bfloat16)*0.3; b=torch.randn(K,N,device="cuda",dtype=torch.bfloat16)*0.3
        bias=torch.randn(N,device="cuda",dtype=torch.bfloat16)*0.1
        ref=torch_seq(a,b,bias); out=tri_fused(a,b,bias)
        err=(out.float()-ref.float()).abs().max().item()
        tri=lambda: tri_fused(a,b,bias); tor=lambda: torch_seq(a,b,bias)
        tm,_=time_fn(tri); tm_t,_=time_fn(tor)
        rows.append(dict(M=M,N=N,K=K,err=round(err,3),triton_ms=round(tm,4),torch_ms=round(tm_t,4),torch_over_triton=round(tm_t/tm,3)))
        print(f"M={M} N={N} K={K} err={err:.3f} fused={tm:.4f}ms seq={tm_t:.4f}ms seq/fused={tm_t/tm:.2f}x")
    print("RESULT_JSON", json.dumps(rows))
if __name__=="__main__": main()
