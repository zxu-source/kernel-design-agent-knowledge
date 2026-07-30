#!/usr/bin/env python3
"""H200 Block+Warp Sum Reduction (Triton) vs torch.sum.
Reduce a large 1D array: each block sums its chunk (tl.sum tree reduce ->
warp-shuffle) and atomicAdds to a global fp32 accumulator. PURPOSE =
characterization (reduction building block). vs torch.sum on H200, Triton 3.6.
"""
import json, statistics
import torch, triton, triton.language as tl
def cdiv(a,b): return (a+b-1)//b
@triton.jit
def reduce_sum(x_ptr, partials_ptr, N, BLOCK: tl.constexpr):
    pid=tl.program_id(0)
    offs=pid*BLOCK+tl.arange(0,BLOCK); mask=offs<N
    x=tl.load(x_ptr+offs, mask=mask, other=0.0).to(tl.float32)
    s=tl.sum(x, axis=0)                       # tree reduce (warp shuffle) -> scalar
    tl.store(partials_ptr+pid, s)
def tri_reduce(x, BLOCK=4096):
    nb=cdiv(x.numel(),BLOCK)
    partials=torch.empty((nb,),device=x.device,dtype=torch.float32)
    reduce_sum[(nb,)](x, partials, x.numel(), BLOCK=BLOCK)
    return partials.sum()                     # tiny 2nd-stage reduce
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
    for N in [1<<20, 1<<22, 1<<24, 1<<26]:
        for dt in (torch.float32, torch.bfloat16):
            x=torch.randn(N,device="cuda",dtype=dt)*0.5
            ref=x.float().sum()
            out=tri_reduce(x)
            err=abs(out.item()-ref.item())
            rel=err/(abs(ref.item())+1e-9)
            tri=lambda: tri_reduce(x); tor=lambda: x.sum()
            tm,_=time_fn(tri); tm_t,_=time_fn(tor)
            rows.append(dict(N=N,dtype=str(dt).split(".")[-1],err=round(err,3),rel=round(rel,5),
                             triton_ms=round(tm,4),torch_ms=round(tm_t,4),torch_over_triton=round(tm_t/tm,3)))
            print(f"N={N:8d} {str(dt).split('.')[-1]:8s} err={err:.3e} rel={rel:.2e} triton={tm:.4f}ms torch={tm_t:.4f}ms torch/triton={tm_t/tm:.2f}x")
    print("RESULT_JSON", json.dumps(rows))
if __name__=="__main__": main()
