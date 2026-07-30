#!/usr/bin/env python3
"""H200 Online Softmax (Triton) validation.
Online softmax: max-subtract (numerically stable), exp, sum, divide in one fused
pass per row. PURPOSE = both: (robustness) max-subtraction prevents exp overflow;
(speedup) fused vs torch's multi-op softmax. vs torch.softmax on H200, Triton 3.6.
"""
import json, statistics
import torch, triton, triton.language as tl

@triton.jit
def softmax_fwd(x_ptr, o_ptr, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N); mask = offs < N
    x = tl.load(x_ptr + row*N + offs, mask=mask, other=-float('inf')).to(tl.float32)
    m = tl.max(x, axis=0)                  # online max (stability)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    tl.store(o_ptr + row*N + offs, (e / s).to(o_ptr.dtype.element_ty), mask=mask)

def triton_softmax(x):
    M,N = x.shape; o=torch.empty_like(x)
    softmax_fwd[(M,)](x, o, N, BLOCK_N=triton.next_power_of_2(N)); return o

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
    # (rows, N): N = vocab / hidden
    shapes=[(4096,4096),(8192,8192),(8192,11008),(4096,32000),(8192,32000),(4096,128256)]
    for M,N in shapes:
        for dt in (torch.bfloat16, torch.float16):
            torch.manual_seed(0)
            x=torch.randn(M,N,device="cuda",dtype=dt)
            ref=torch.softmax(x.float(),dim=-1).to(dt)
            out=triton_softmax(x)
            err=(out.float()-ref.float()).abs().max().item()
            tri=lambda: triton_softmax(x); tor=lambda: torch.softmax(x,dim=-1)
            tm,_=time_fn(tri); tm_t,_=time_fn(tor)
            rows.append(dict(M=M,N=N,dtype=str(dt).split(".")[-1],err=round(err,5),
                             triton_ms=round(tm,4),torch_ms=round(tm_t,4),torch_over_triton=round(tm_t/tm,3)))
            print(f"M={M:5d} N={N:6d} {str(dt).split('.')[-1]:8s} err={err:.3e} triton={tm:.4f}ms torch={tm_t:.4f}ms torch/triton={tm_t/tm:.2f}x")
    print("RESULT_JSON", json.dumps(rows)); print("max_err=", max(r["err"] for r in rows))

if __name__=="__main__": main()
