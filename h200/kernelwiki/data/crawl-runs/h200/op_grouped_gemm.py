#!/usr/bin/env python3
"""H200 Grouped GEMM (Triton) vs torch loop.
GROUP expert GEMMs A_g[Mg,K] x B_g[K,N] (uniform Mg, same K,N). 2D grid over
(M-tiles, N-tiles); each tile looks up its group from pid_m and loads the right
B_g. PURPOSE = speedup: one launch + better SM occupancy vs a torch loop.
H200, Triton 3.6.
"""
import json, statistics
import torch, triton, triton.language as tl
def cdiv(a,b): return (a+b-1)//b
@triton.jit
def grouped_gemm(a_ptr,b_ptr,o_ptr,GROUP,Mg,N,K,sam,sak,sbgrp,srow,scom,scn,
                 BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,TILES_PER_GROUP:tl.constexpr):
    pid_m=tl.program_id(0); pid_n=tl.program_id(1)
    g = pid_m // TILES_PER_GROUP
    local = pid_m % TILES_PER_GROUP
    om = g*Mg + local*BM + tl.arange(0,BM)
    on = pid_n*BN + tl.arange(0,BN)
    acc=tl.zeros((BM,BN),dtype=tl.float32)
    b_base = b_ptr + g*sbgrp                       # group g's B[K,N]
    for k0 in range(0,K,BK):
        ok=k0+tl.arange(0,BK)
        a=tl.load(a_ptr+om[:,None]*sam+ok[None,:]*sak,mask=(om[:,None]<GROUP*Mg)&(ok[None,:]<K),other=0.0)
        b=tl.load(b_base+ok[:,None]*srow+on[None,:]*scn,mask=(ok[:,None]<K)&(on[None,:]<N),other=0.0)
        acc+=tl.dot(a,b)
    tl.store(o_ptr+om[:,None]*scom+on[None,:]*scn, acc, mask=(om[:,None]<GROUP*Mg)&(on[None,:]<N))
def tri_grouped(A,B,BM=128,BN=128,BK=64,nw=8,ns=3):
    GROUP,Mg,K=A.shape; N=B.shape[2]
    A2=A.reshape(GROUP*Mg,K).contiguous(); B2=B.reshape(GROUP*K,N).contiguous()
    O=torch.empty((GROUP*Mg,N),device=A.device,dtype=torch.bfloat16)
    tpg=cdiv(Mg,BM)
    grouped_gemm[(GROUP*tpg, cdiv(N,BN))](A2,B2,O,GROUP,Mg,N,K,
        A2.stride(0),1, K*N, N, O.stride(0),1,
        BM=BM,BN=BN,BK=BK,TILES_PER_GROUP=tpg,num_warps=nw,num_stages=ns)
    return O.reshape(GROUP,Mg,N)
def torch_loop(A,B):
    return torch.stack([A[g]@B[g] for g in range(A.shape[0])])
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
    for GROUP,Mg,N,K in [(8,512,4096,4096),(8,1024,4096,4096),(16,256,4096,4096),(32,128,4096,4096),(8,2048,8192,4096)]:
        A=torch.randn(GROUP,Mg,K,device="cuda",dtype=torch.bfloat16)*0.3
        B=torch.randn(GROUP,K,N,device="cuda",dtype=torch.bfloat16)*0.3
        ref=torch_loop(A,B); out=tri_grouped(A,B)
        err=(out.float()-ref.float()).abs().max().item()
        rel = err/(ref.float().abs().mean().item()+1e-9)
        tri=lambda: tri_grouped(A,B); tor=lambda: torch_loop(A,B)
        tm,_=time_fn(tri); tm_t,_=time_fn(tor)
        rows.append(dict(GROUP=GROUP,Mg=Mg,N=N,K=K,err=round(err,3),rel=round(rel,4),triton_ms=round(tm,4),torch_ms=round(tm_t,4),torch_over_triton=round(tm_t/tm,3)))
        print(f"G={GROUP} Mg={Mg} N={N} K={K} err={err:.3f} rel={rel:.2e} grouped={tm:.4f}ms loop={tm_t:.4f}ms loop/grouped={tm_t/tm:.2f}x")
    print("RESULT_JSON", json.dumps(rows))
if __name__=="__main__": main()
