#!/usr/bin/env python3
"""H200 Fused Gated MLP (gate-up projection) (Triton) vs torch sequential.
LLM MLP: concat gate_W|up_W into Wu [K,2N]; ONE GEMM x@Wu -> [M,2N]; split ->
silu(gate)*up. vs torch sequential: silu(x@gate_W)*(x@up_W) (2 GEMMs, x read
twice). PURPOSE = speedup (1 GEMM + x read once). H200, Triton 3.6.
"""
import json, statistics
import torch, triton, triton.language as tl
def cdiv(a,b): return (a+b-1)//b
@triton.jit
def gemm_2n(a_ptr, wu_ptr, o_ptr, M, K, N2, sam,
            BM:tl.constexpr, BN:tl.constexpr, BK:tl.constexpr):
    pm=tl.program_id(0); pn=tl.program_id(1)
    om=pm*BM+tl.arange(0,BM); on=pn*BN+tl.arange(0,BN)
    acc=tl.zeros((BM,BN),dtype=tl.float32)
    for k0 in range(0,K,BK):
        ok=k0+tl.arange(0,BK)
        a=tl.load(a_ptr+om[:,None]*sam+ok[None,:],mask=(om[:,None]<M)&(ok[None,:]<K),other=0.0)
        w=tl.load(wu_ptr+ok[:,None]*N2+on[None,:],mask=(ok[:,None]<K)&(on[None,:]<N2),other=0.0)
        acc+=tl.dot(a,w)
    tl.store(o_ptr+om[:,None]*N2+on[None,:], acc.to(o_ptr.dtype.element_ty), mask=(om[:,None]<M)&(on[None,:]<N2))
@triton.jit
def silu_and_mul(g_ptr,u_ptr,o_ptr,total,BLOCK:tl.constexpr):
    pid=tl.program_id(0); offs=pid*BLOCK+tl.arange(0,BLOCK); mask=offs<total
    g=tl.load(g_ptr+offs,mask=mask,other=0.0).to(tl.float32)
    u=tl.load(u_ptr+offs,mask=mask,other=0.0).to(tl.float32)
    tl.store(o_ptr+offs, (g/(1.0+tl.exp(-g))*u).to(o_ptr.dtype.element_ty), mask=mask)
def tri_fused(x, Wu, BM=128, BN=256, BK=64, nw=8, ns=3):
    M,K=x.shape; N2=Wu.shape[1]; N=N2//2
    o2=torch.empty((M,N2),device=x.device,dtype=x.dtype)
    gemm_2n[(cdiv(M,BM),cdiv(N2,BN))](x,Wu,o2,M,K,N2,x.stride(0),BM=BM,BN=BN,BK=BK,num_warps=nw,num_stages=ns)
    out=torch.empty((M,N),device=x.device,dtype=x.dtype)
    nb=cdiv(M*N,4096)
    silu_and_mul[(nb,)](o2[:, :N].contiguous(), o2[:, N:].contiguous(), out, M*N, BLOCK=4096)
    return out
def torch_seq(x, gateW, upW):
    return torch.nn.functional.silu(x@gateW) * (x@upW)
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
    for M,K,N in [(2048,4096,4096),(4096,4096,4096),(8192,4096,11008),(8192,8192,14336),(8192,4096,14336)]:
        dt=torch.bfloat16
        x=torch.randn(M,K,device="cuda",dtype=dt)*0.3
        gateW=torch.randn(K,N,device="cuda",dtype=dt)*0.3; upW=torch.randn(K,N,device="cuda",dtype=dt)*0.3
        Wu=torch.cat([gateW,upW],dim=1).contiguous()  # [K,2N]
        ref=torch_seq(x,gateW,upW); out=tri_fused(x,Wu)
        err=(out.float()-ref.float()).abs().max().item()
        tri=lambda: tri_fused(x,Wu); tor=lambda: torch_seq(x,gateW,upW)
        tm,_=time_fn(tri); tm_t,_=time_fn(tor)
        rows.append(dict(M=M,K=K,N=N,err=round(err,3),triton_ms=round(tm,4),torch_ms=round(tm_t,4),torch_over_triton=round(tm_t/tm,3)))
        print(f"M={M} K={K} N={N} err={err:.3f} fused={tm:.4f}ms seq={tm_t:.4f}ms seq/fused={tm_t/tm:.2f}x")
    print("RESULT_JSON", json.dumps(rows))
if __name__=="__main__": main()
