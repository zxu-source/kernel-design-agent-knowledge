#!/usr/bin/env python3
"""H200 L2-Normalize (unit norm per row) (Triton) vs torch.nn.functional.normalize.
y = x / ||x||_2 per row. One fused kernel (sum x^2, rsqrt, multiply).
PURPOSE = speedup. vs torch on H200, Triton 3.6.
"""
import json, statistics
import torch, triton, triton.language as tl
@triton.jit
def l2norm_fwd(x_ptr, o_ptr, eps, N, BLOCK_N: tl.constexpr):
    row=tl.program_id(0)
    offs=tl.arange(0, BLOCK_N); mask=offs<N
    x=tl.load(x_ptr+row*N+offs, mask=mask, other=0.0).to(tl.float32)
    s=tl.sum(x*x, axis=0)
    inv=tl.rsqrt(s+eps)
    tl.store(o_ptr+row*N+offs, (x*inv).to(o_ptr.dtype.element_ty), mask=mask)
def tri_l2(x, eps=1e-8):
    M,N=x.shape; o=torch.empty_like(x)
    l2norm_fwd[(M,)](x, o, eps, N, BLOCK_N=triton.next_power_of_2(N)); return o
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
        for dt in (torch.float32, torch.bfloat16):
            torch.manual_seed(0); x=torch.randn(M,N,device="cuda",dtype=dt)*0.3
            ref=torch.nn.functional.normalize(x, dim=-1)
            out=tri_l2(x)
            err=(out.float()-ref.float()).abs().max().item()
            tri=lambda: tri_l2(x); tor=lambda: torch.nn.functional.normalize(x, dim=-1)
            tm,_=time_fn(tri); tm_t,_=time_fn(tor)
            rows.append(dict(M=M,N=N,dtype=str(dt).split(".")[-1],err=round(err,5),triton_ms=round(tm,4),torch_ms=round(tm_t,4),torch_over_triton=round(tm_t/tm,3)))
            print(f"M={M} N={N} {str(dt).split('.')[-1]:8s} err={err:.3e} triton={tm:.4f}ms torch={tm_t:.4f}ms torch/triton={tm_t/tm:.2f}x")
    print("RESULT_JSON", json.dumps(rows))
if __name__=="__main__": main()
