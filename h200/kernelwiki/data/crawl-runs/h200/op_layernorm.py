#!/usr/bin/env python3
"""H200 LayerNorm forward (Triton) validation.
LayerNorm: y = (x-mean)*rsqrt(var+eps)*w + b, var=mean((x-mean)^2). One fused
kernel (load once, compute sum_x & sum_x2 -> mean/var, normalize).
PURPOSE = both: (speedup) fused single-pass vs reduce+center+normalize+scale+bias;
(robustness) fp32 reductions + eps. vs torch.nn.functional.layer_norm on H200.
"""
import json, statistics
import torch, triton, triton.language as tl

@triton.jit
def layernorm_fwd(x_ptr, w_ptr, b_ptr, o_ptr, eps, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N); mask = offs < N
    x = tl.load(x_ptr + row*N + offs, mask=mask, other=0.0).to(tl.float32)
    sum_x = tl.sum(x, axis=0)
    sum_x2 = tl.sum(x*x, axis=0)
    mean = sum_x / N
    var = sum_x2 / N - mean*mean
    rrms = tl.rsqrt(var + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = (x - mean) * rrms * w + b
    tl.store(o_ptr + row*N + offs, y.to(o_ptr.dtype.element_ty), mask=mask)

def triton_ln(x, w, b, eps):
    M,N = x.shape; o=torch.empty_like(x)
    layernorm_fwd[(M,)](x, w, b, o, eps, N, BLOCK_N=triton.next_power_of_2(N))
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
    eps=1e-5
    shapes=[(4096,4096),(8192,4096),(4096,8192),(8192,8192),(4096,11008),(8192,14336)]
    rows=[]
    for M,N in shapes:
        for dt in (torch.bfloat16, torch.float16):
            torch.manual_seed(0)
            x=torch.randn(M,N,device="cuda",dtype=dt)*0.3
            w=torch.randn(N,device="cuda",dtype=dt)*0.1+1.0
            bb=torch.zeros(N,device="cuda",dtype=dt)
            ref=torch.nn.functional.layer_norm(x,(N,),w,bb,eps)
            out=triton_ln(x,w,bb,eps)
            err=(out.float()-ref.float()).abs().max().item()
            tri=lambda: triton_ln(x,w,bb,eps)
            tor=lambda: torch.nn.functional.layer_norm(x,(N,),w,bb,eps)
            tm,_=time_fn(tri); tm_t,_=time_fn(tor)
            rows.append(dict(M=M,N=N,dtype=str(dt).split(".")[-1], err=round(err,5),
                             triton_ms=round(tm,4), torch_ms=round(tm_t,4),
                             torch_over_triton=round(tm_t/tm,3)))
            print(f"M={M:5d} N={N:5d} {str(dt).split('.')[-1]:8s} err={err:.3e} triton={tm:.4f}ms torch={tm_t:.4f}ms torch/triton={tm_t/tm:.2f}x")
    print("RESULT_JSON", json.dumps(rows)); print("max_err=", max(r["err"] for r in rows))

if __name__=="__main__": main()
