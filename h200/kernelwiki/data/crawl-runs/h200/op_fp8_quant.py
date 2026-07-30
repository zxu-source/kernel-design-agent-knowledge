#!/usr/bin/env python3
"""H200 Per-Tensor FP8 e4m3 Quantization (Triton) vs torch.
amax over tensor -> scale = amax/448 (fp8e4m3 max) -> cast x*scale to fp8_e4m3.
PURPOSE = speedup (produces fp8 enabling FP8 GEMM). vs torch on H200, Triton 3.6.
"""
import json, statistics
import torch, triton, triton.language as tl
def cdiv(a,b): return (a+b-1)//b
FP8_E4M3_MAX=448.0
@triton.jit
def amax_k(x_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid=tl.program_id(0); offs=pid*BLOCK+tl.arange(0,BLOCK); mask=offs<N
    x=tl.load(x_ptr+offs, mask=mask, other=0.0)
    m=tl.max(tl.abs(x), axis=0).to(tl.float32)
    tl.atomic_max(out_ptr, m)
@triton.jit
def fp8_quant_k(x_ptr, o_ptr, inv_scale, N, BLOCK: tl.constexpr):
    pid=tl.program_id(0); offs=pid*BLOCK+tl.arange(0,BLOCK); mask=offs<N
    x=tl.load(x_ptr+offs, mask=mask, other=0.0).to(tl.float32)
    q=x*inv_scale
    q=tl.minimum(tl.maximum(q, -448.0), 448.0)   # clamp to fp8e4m3 range
    tl.store(o_ptr+offs, q.to(tl.float8e4nv), mask=mask)
def tri_fp8q(x, BLOCK=4096):
    N=x.numel()
    amax=torch.zeros((1,),device=x.device,dtype=torch.float32)+1e-12
    amax_k[(cdiv(N,BLOCK),)](x, amax, N, BLOCK=BLOCK)
    scale=amax/FP8_E4M3_MAX
    o=torch.empty((N,),device=x.device,dtype=torch.float8_e4m3fn)
    fp8_quant_k[(cdiv(N,BLOCK),)](x, o, (FP8_E4M3_MAX/amax).item(), N, BLOCK=BLOCK)
    return o, scale
def torch_fp8q(x):
    amax=x.abs().max().clamp(min=1e-12).float(); scale=amax/FP8_E4M3_MAX
    o=(x/scale).clamp(-FP8_E4M3_MAX,FP8_E4M3_MAX).to(torch.float8_e4m3fn); return o,scale
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
            o_t,s_t=tri_fp8q(x); o_ref,s_ref=torch_fp8q(x)
            # dequantize and compare relative error
            err=((o_t.float()-o_ref.float()).abs().max().item())
            match=(o_t==o_ref).float().mean().item()
            tri=lambda: tri_fp8q(x); tor=lambda: torch_fp8q(x)
            tm,_=time_fn(tri); tm_t,_=time_fn(tor)
            rows.append(dict(N=N,dtype=str(dt).split(".")[-1],match=round(match,4),maxdiff=round(err,3),triton_ms=round(tm,4),torch_ms=round(tm_t,4),torch_over_triton=round(tm_t/tm,3)))
            print(f"N={N:8d} {str(dt).split('.')[-1]:8s} match={match:.4f} maxdiff={err:.3f} triton={tm:.4f}ms torch={tm_t:.4f}ms torch/triton={tm_t/tm:.2f}x")
    print("RESULT_JSON", json.dumps(rows))
if __name__=="__main__": main()
