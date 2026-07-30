#!/usr/bin/env python3
"""H200 Cosine Similarity (fused dot + norms) (Triton) vs torch.
y[i] = (a[i].b[i]) / (||a[i]|| ||b[i]|| + eps). One fused kernel per row
(dot + sum_sq a + sum_sq b + divide). PURPOSE = speedup. H200, Triton 3.6.
"""
import json, statistics
import torch, triton, triton.language as tl
@triton.jit
def cos_sim(a_ptr, b_ptr, o_ptr, eps, N, BLOCK_N: tl.constexpr):
    row=tl.program_id(0)
    offs=tl.arange(0, BLOCK_N); mask=offs<N
    a=tl.load(a_ptr+row*N+offs, mask=mask, other=0.0).to(tl.float32)
    b=tl.load(b_ptr+row*N+offs, mask=mask, other=0.0).to(tl.float32)
    dot=tl.sum(a*b, axis=0)
    na=tl.sqrt(tl.sum(a*a, axis=0))
    nb=tl.sqrt(tl.sum(b*b, axis=0))
    y=dot/(na*nb+eps)
    tl.store(o_ptr+row, y)
def tri_cos(a,b,eps=1e-8):
    M,N=a.shape; o=torch.empty((M,),device=a.device,dtype=torch.float32)
    cos_sim[(M,)](a,b,o,eps,N,BLOCK_N=triton.next_power_of_2(N)); return o
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
    for M,N in [(4096,4096),(8192,8192),(8192,11008),(8192,14336),(16384,14336)]:
        torch.manual_seed(0); a=torch.randn(M,N,device="cuda",dtype=torch.float32)*0.3
        b=torch.randn(M,N,device="cuda",dtype=torch.float32)*0.3
        ref=torch.nn.functional.cosine_similarity(a,b,dim=-1)
        out=tri_cos(a,b)
        err=(out-ref).abs().max().item()
        tri=lambda: tri_cos(a,b); tor=lambda: torch.nn.functional.cosine_similarity(a,b,dim=-1)
        tm,_=time_fn(tri); tm_t,_=time_fn(tor)
        rows.append(dict(M=M,N=N,err=round(err,6),triton_ms=round(tm,4),torch_ms=round(tm_t,4),torch_over_triton=round(tm_t/tm,3)))
        print(f"M={M} N={N} err={err:.3e} triton={tm:.4f}ms torch={tm_t:.4f}ms torch/triton={tm_t/tm:.2f}x")
    print("RESULT_JSON", json.dumps(rows))
if __name__=="__main__": main()
