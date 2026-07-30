#!/usr/bin/env python3
"""H200 RMSNorm forward (Triton) validation.

RMSNorm = y = x / sqrt(mean(x^2)+eps) * w. Canonical LLM final-norm / block-norm.
PURPOSE = both: (speedup) one fused kernel replaces reduce + rsqrt + normalize +
mul (fewer launches, one read/one write); (robustness) fp32 reduction + eps
avoids fp16 overflow/underflow in the variance. Compared to torch
rms_norm (CUDA) for correctness and latency on H200 (SM90), Triton 3.6.
"""
import json, statistics
import torch, triton, triton.language as tl

@triton.jit
def rmsnorm_fwd(x_ptr, w_ptr, o_ptr, eps, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    xrow = tl.load(x_ptr + row*N + offs, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(xrow*xrow, axis=0) / N          # fp32 reduction
    rrms = tl.rsqrt(var + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = xrow * rrms * w
    tl.store(o_ptr + row*N + offs, y.to(o_ptr.dtype.element_ty), mask=mask)

def triton_rmsnorm(x, w, eps):
    M, N = x.shape
    o = torch.empty_like(x)
    BLOCK_N = triton.next_power_of_2(N)
    rmsnorm_fwd[(M,)](x, w, o, eps, N, BLOCK_N=BLOCK_N)
    return o

def time_fn(fn, trials=30):
    for _ in range(8): fn(); torch.cuda.synchronize()
    ts=[]
    for _ in range(trials):
        s,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    return min(ts), statistics.median(ts)

def main():
    print(f"triton={triton.__version__} dev={torch.cuda.get_device_name(0)} SMs={torch.cuda.get_device_properties(0).multi_processor_count}")
    eps=1e-6
    # (rows, hidden): typical LLM shapes
    shapes=[(4096,4096),(8192,4096),(4096,8192),(8192,8192),(1,11008),(4096,11008),(8192,14336)]
    rows=[]
    for M,N in shapes:
        for dt in (torch.bfloat16, torch.float16):
            torch.manual_seed(0)
            x=torch.randn(M,N,device="cuda",dtype=dt)*0.3
            w=torch.randn(N,device="cuda",dtype=dt)*0.1+1.0
            ref=torch.nn.functional.rms_norm(x, normalized_shape=(N,), weight=w, eps=eps)
            out=triton_rmsnorm(x,w,eps)
            err=(out.float()-ref.float()).abs().max().item()
            tri=lambda: triton_rmsnorm(x,w,eps)
            tor=lambda: torch.nn.functional.rms_norm(x,(N,),weight=w,eps=eps)
            tm,_=time_fn(tri); tm_t,_=time_fn(tor)
            rows.append(dict(M=M,N=N,dtype=str(dt).split(".")[-1], err=round(err,5),
                             triton_ms=round(tm,4), torch_ms=round(tm_t,4),
                             torch_over_triton=round(tm_t/tm,3)))
            print(f"M={M:5d} N={N:5d} {str(dt).split('.')[-1]:8s} err={err:.3e} triton={tm:.4f}ms torch={tm_t:.4f}ms torch/triton={tm_t/tm:.2f}x")
    print("RESULT_JSON", json.dumps(rows)); print("max_err=", max(r["err"] for r in rows))

if __name__=="__main__": main()
