#!/usr/bin/env python3
"""H200 Groupwise (group=128) Symmetric INT8 Quantization (Triton) vs torch.
AWQ/GPTQ granularity: each group of G=128 consecutive elements gets its own
scale (finer than per-channel for large N -> better accuracy). One kernel per
(row, group): amax over G, scale, round/clamp/cast to int8. PURPOSE = both
(speedup + accuracy). vs torch on H200, Triton 3.6.
"""
import json, statistics
import torch, triton, triton.language as tl
G_ = 128
@triton.jit
def groupwise_quant(w_ptr, o_ptr, scale_ptr, M, Ngroups, G: tl.constexpr, N: tl.constexpr):
    pid=tl.program_id(0)
    row=pid // Ngroups
    g=pid % Ngroups
    offs=tl.arange(0, G); base=row*N + g*G; mask=offs<G
    w=tl.load(w_ptr+base+offs, mask=mask, other=0.0).to(tl.float32)
    amax=tl.maximum(tl.max(tl.abs(w),axis=0), 1e-12)
    inv=127.0/amax
    q=tl.extra.libdevice.llrint(w*inv); q=tl.maximum(tl.minimum(q,127.0),-128.0)
    tl.store(o_ptr+base+offs, q.to(tl.int8), mask=mask)
    tl.store(scale_ptr + row*Ngroups + g, amax/127.0)
def tri_gq(W, G=128):
    M,N=W.shape; Ng=N//G
    o=torch.empty((M,N),device=W.device,dtype=torch.int8); s=torch.empty((M,Ng),device=W.device,dtype=torch.float32)
    groupwise_quant[(M*Ng,)](W, o, s, M, Ng, G=G, N=N); return o,s
def torch_gq(W, G=128):
    M,N=W.shape; Ng=N//G
    Wr=W.float().reshape(M,Ng,G); amax=Wr.abs().amax(dim=-1).clamp(min=1e-12); scale=amax/127.0
    o=(Wr/scale[...,None]).round().clamp(-128,127).to(torch.int8).reshape(M,N)
    return o, scale
def time_fn(fn,trials=50):
    for _ in range(10): fn(); torch.cuda.synchronize()
    ts=[]
    for _ in range(trials):
        s,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    return min(ts), statistics.median(ts)
def main():
    print(f"triton={triton.__version__} dev={torch.cuda.get_device_name(0)} SMs={torch.cuda.get_device_properties(0).multi_processor_count} group=128")
    rows=[]
    for M,N in [(4096,4096),(8192,8192),(8192,11008),(8192,14336),(16384,14336)]:
        for dt in (torch.float32, torch.bfloat16):
            torch.manual_seed(0); W=torch.randn(M,N,device="cuda",dtype=dt)*0.3
            o_t,s_t=tri_gq(W); o_ref,s_ref=torch_gq(W)
            match=(o_t==o_ref).float().mean().item(); diff=(o_t.to(torch.int32)-o_ref.to(torch.int32)).abs().max().item()
            serr=(s_t-s_ref).abs().max().item()
            tri=lambda: tri_gq(W); tor=lambda: torch_gq(W)
            tm,_=time_fn(tri); tm_t,_=time_fn(tor)
            rows.append(dict(M=M,N=N,dtype=str(dt).split(".")[-1],match=round(match,4),maxdiff=int(diff),serr=round(serr,5),triton_ms=round(tm,4),torch_ms=round(tm_t,4),torch_over_triton=round(tm_t/tm,3)))
            print(f"M={M} N={N} {str(dt).split('.')[-1]:8s} match={match:.3f} maxdiff={diff} serr={serr:.2e} triton={tm:.4f}ms torch={tm_t:.4f}ms torch/triton={tm_t/tm:.2f}x")
    print("RESULT_JSON", json.dumps(rows))
if __name__=="__main__": main()
