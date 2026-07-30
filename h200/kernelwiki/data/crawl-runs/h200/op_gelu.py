#!/usr/bin/env python3
"""H200 GELU (tanh approx) (Triton) validation.
GELU tanh: 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3))). Elementwise, one kernel
vs torch.nn.functional.gelu(approximate='tanh'). PURPOSE = speedup/characterize.
"""
import json, statistics
import torch, triton, triton.language as tl
import math

@triton.jit
def gelu_tanh(x_ptr, o_ptr, total, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid*BLOCK + tl.arange(0, BLOCK); mask = offs < total
    x = tl.load(x_ptr+offs, mask=mask, other=0.0).to(tl.float32)
    c = tl.full((), 0.7978845608028654, tl.float32)   # sqrt(2/pi)
    inner = c * (x + 0.044715 * x*x*x)
    e2 = tl.exp(2.0 * inner)
    tanh = (e2 - 1.0) / (e2 + 1.0)
    y = 0.5 * x * (1.0 + tanh)
    tl.store(o_ptr+offs, y.to(o_ptr.dtype.element_ty), mask=mask)

def triton_gelu(x):
    o=torch.empty_like(x); total=x.numel(); BLOCK=4096
    gelu_tanh[(triton.cdiv(total,BLOCK),)](x, o, total, BLOCK=BLOCK); return o

def time_fn(fn, trials=50):
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
        for dt in (torch.bfloat16, torch.float16):
            torch.manual_seed(0)
            x=torch.randn(M,N,device="cuda",dtype=dt)*0.7
            ref=torch.nn.functional.gelu(x, approximate='tanh')
            out=triton_gelu(x)
            err=(out.float()-ref.float()).abs().max().item()
            tri=lambda: triton_gelu(x); tor=lambda: torch.nn.functional.gelu(x, approximate='tanh')
            tm,_=time_fn(tri); tm_t,_=time_fn(tor)
            rows.append(dict(M=M,N=N,dtype=str(dt).split(".")[-1],err=round(err,5),
                             triton_ms=round(tm,4),torch_ms=round(tm_t,4),torch_over_triton=round(tm_t/tm,3)))
            print(f"M={M:5d} N={N:5d} {str(dt).split('.')[-1]:8s} err={err:.3e} triton={tm:.4f}ms torch={tm_t:.4f}ms torch/triton={tm_t/tm:.2f}x")
    print("RESULT_JSON", json.dumps(rows)); print("max_err=", max(r["err"] for r in rows))

if __name__=="__main__": main()
