#!/usr/bin/env python3
"""H200 Per-Tensor Symmetric INT8 Quantization (Triton) vs torch.
amax over the whole tensor -> scale = amax/127 -> quantize x*inv_scale to int8
(round + clamp). Two kernels: (1) amax via atomic_max, (2) elementwise cast.
PURPOSE = speedup (produces int8 that enables faster INT8 GEMM). vs torch on H200.
"""
import json, statistics
import torch, triton, triton.language as tl
def cdiv(a,b): return (a+b-1)//b
@triton.jit
def amax_kernel(x_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid=tl.program_id(0)
    offs=pid*BLOCK+tl.arange(0,BLOCK); mask=offs<N
    x=tl.load(x_ptr+offs, mask=mask, other=0.0)
    m=tl.max(tl.abs(x), axis=0).to(tl.float32)
    tl.atomic_max(out_ptr, m)
@triton.jit
def quant_kernel(x_ptr, o_ptr, inv_scale, N, BLOCK: tl.constexpr):
    pid=tl.program_id(0)
    offs=pid*BLOCK+tl.arange(0,BLOCK); mask=offs<N
    x=tl.load(x_ptr+offs, mask=mask, other=0.0).to(tl.float32)
    q=(x * inv_scale)
    q=tl.extra.libdevice.llrint(q)            # round to nearest int
    q=tl.maximum(tl.minimum(q, 127.0), -128.0)
    tl.store(o_ptr+offs, q.to(tl.int8), mask=mask)
def tri_quant(x, BLOCK=4096):
    N=x.numel()
    amax=torch.zeros((1,),device=x.device,dtype=torch.float32)+1e-12
    amax_kernel[(cdiv(N,BLOCK),)](x, amax, N, BLOCK=BLOCK)
    scale=amax/127.0
    o=torch.empty((N,),device=x.device,dtype=torch.int8)
    quant_kernel[(cdiv(N,BLOCK),)](x, o, (127.0/amax).item(), N, BLOCK=BLOCK)
    return o, scale
def torch_quant(x):
    amax=x.abs().max(); scale=amax/127.0
    o=(x/scale).round().clamp(-128,127).to(torch.int8); return o, scale
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
    for N in [1<<20, 1<<22, 1<<24, 1<<25]:
        for dt in (torch.float32, torch.bfloat16):
            x=torch.randn(N,device="cuda",dtype=dt)*0.5
            o_t,s_t=tri_quant(x); o_ref,s_ref=torch_quant(x)
            # int8 match fraction (rounding may differ by 1 ulp)
            match=(o_t==o_ref).float().mean().item()
            diff=(o_t.to(torch.int32)-o_ref.to(torch.int32)).abs().max().item()
            tri=lambda: tri_quant(x); tor=lambda: torch_quant(x)
            tm,_=time_fn(tri); tm_t,_=time_fn(tor)
            rows.append(dict(N=N,dtype=str(dt).split(".")[-1],match=round(match,5),maxdiff=int(diff),
                             triton_ms=round(tm,4),torch_ms=round(tm_t,4),torch_over_triton=round(tm_t/tm,3)))
            print(f"N={N:8d} {str(dt).split('.')[-1]:8s} match={match:.4f} maxdiff={diff} triton={tm:.4f}ms torch={tm_t:.4f}ms torch/triton={tm_t/tm:.2f}x")
    print("RESULT_JSON", json.dumps(rows))
if __name__=="__main__": main()
