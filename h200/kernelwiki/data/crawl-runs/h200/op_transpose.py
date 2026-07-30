#!/usr/bin/env python3
"""H200 Tiled Transpose (Triton) vs torch — square + non-square.
Coalesced tiled transpose: load [BM,BN] tile from input, tl.trans, store
coalesced to output (the transposed position). Avoids the uncoalesced
strided writes of a naive transpose. PURPOSE = speedup (coalescing).
vs torch A.t().contiguous() on H200, Triton 3.6.
"""
import json, statistics
import torch, triton, triton.language as tl
def cdiv(a,b): return (a+b-1)//b
@triton.jit
def tiled_transpose(a_ptr, o_ptr, M, N, sam, son, BM: tl.constexpr, BN: tl.constexpr):
    pm=tl.program_id(0); pn=tl.program_id(1)
    rm=pm*BM+tl.arange(0,BM); cn=pn*BN+tl.arange(0,BN)
    # load tile [BM,BN] from A (row-major): A[rm, cn]
    a=tl.load(a_ptr + rm[:,None]*sam + cn[None,:], mask=(rm[:,None]<M)&(cn[None,:]<N), other=0.0)
    at=tl.trans(a)   # [BN,BM]
    rn=pn*BN+tl.arange(0,BN); cm=pm*BM+tl.arange(0,BM)
    tl.store(o_ptr + rn[:,None]*son + cm[None,:], at, mask=(rn[:,None]<N)&(cm[None,:]<M))
def tri_t(a, BM=64, BN=64):
    M,N=a.shape
    o=torch.empty((N,M),device=a.device,dtype=a.dtype)
    tiled_transpose[(cdiv(M,BM),cdiv(N,BN))](a,o,M,N,a.stride(0),o.stride(0),BM=BM,BN=BN)
    return o
def time_fn(fn,trials=50):
    for _ in range(10): fn(); torch.cuda.synchronize()
    ts=[]
    for _ in range(trials):
        s,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    return min(ts), statistics.median(ts)
def main():
    print(f"triton={triton.__version__} dev={torch.cuda.get_device_name(0)} SMs={torch.cuda.get_device_properties(0).multi_processor_count}")
    rows=[]
    for M,N in [(4096,4096),(8192,8192),(8192,4096),(16384,16384),(4096,14336)]:
        for dt in (torch.float32, torch.float16):
            a=torch.randn(M,N,device="cuda",dtype=dt)
            ref=a.t().contiguous(); out=tri_t(a)
            err=(out-ref).abs().max().item()
            tri=lambda: tri_t(a); tor=lambda: a.t().contiguous()
            tm,_=time_fn(tri); tm_t,_=time_fn(tor)
            rows.append(dict(M=M,N=N,dtype=str(dt).split(".")[-1],err=round(err,5),triton_ms=round(tm,4),torch_ms=round(tm_t,4),torch_over_triton=round(tm_t/tm,3)))
            print(f"M={M} N={N} {str(dt).split('.')[-1]:8s} err={err:.2e} triton={tm:.4f}ms torch={tm_t:.4f}ms torch/triton={tm_t/tm:.2f}x")
    print("RESULT_JSON", json.dumps(rows))
if __name__=="__main__": main()
