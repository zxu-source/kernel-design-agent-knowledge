#!/usr/bin/env python3
"""H200 Fused Pre-Norm Residual + Fused Add+SiLU (Triton) vs torch.
Pre-norm residual: out = x + LayerNorm(x, w, b). Common in GPT-style transformers.
Add+SiLU: out = silu(a + b). Two-input elementwise fusion.
PURPOSE = speedup (fusion). H200, Triton 3.6.
"""
import json, statistics
import torch, triton, triton.language as tl

@triton.jit
def fused_prenorm(x_ptr, w_ptr, b_ptr, o_ptr, eps, N, BLOCK_N: tl.constexpr):
    row=tl.program_id(0); offs=tl.arange(0, BLOCK_N); mask=offs<N
    x=tl.load(x_ptr+row*N+offs, mask=mask, other=0.0).to(tl.float32)
    mean=tl.sum(x, axis=0)/N; var=tl.sum((x-mean)*(x-mean), axis=0)/N
    rrms=tl.rsqrt(var+eps)
    w=tl.load(w_ptr+offs, mask=mask, other=0.0).to(tl.float32)
    b=tl.load(b_ptr+offs, mask=mask, other=0.0).to(tl.float32)
    y = x + (x-mean)*rrms*w + b     # pre-norm residual: x + LN(x)
    tl.store(o_ptr+row*N+offs, y.to(o_ptr.dtype.element_ty), mask=mask)

@triton.jit
def fused_add_silu(a_ptr, b_ptr, o_ptr, total, BLOCK: tl.constexpr):
    pid=tl.program_id(0); offs=pid*BLOCK+tl.arange(0,BLOCK); mask=offs<total
    a=tl.load(a_ptr+offs, mask=mask, other=0.0).to(tl.float32)
    b=tl.load(b_ptr+offs, mask=mask, other=0.0).to(tl.float32)
    s=a+b; sig=1.0/(1.0+tl.exp(-s))
    tl.store(o_ptr+offs, (s*sig).to(o_ptr.dtype.element_ty), mask=mask)

def time_fn(fn, trials=50):
    for _ in range(10): fn(); torch.cuda.synchronize()
    ts=[]
    for _ in range(trials):
        s,e=torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    return min(ts), statistics.median(ts)

def main():
    print(f"triton={triton.__version__} dev={torch.cuda.get_device_name(0)} SMs={torch.cuda.get_device_properties(0).multi_processor_count}")
    eps=1e-5
    print("--- Fused Pre-Norm Residual (x+LN(x)) ---")
    for M,N in [(4096,4096),(8192,8192),(8192,14336)]:
        dt=torch.float32; torch.manual_seed(0)
        x=torch.randn(M,N,device="cuda",dtype=dt)*0.3
        w=torch.randn(N,device="cuda",dtype=dt)*0.1+1.0; b=torch.zeros(N,device="cuda",dtype=dt)
        ref=x+torch.nn.functional.layer_norm(x,(N,),w,b,eps)
        o=torch.empty_like(x); fused_prenorm[(M,)](x,w,b,o,eps,N,BLOCK_N=triton.next_power_of_2(N))
        err=(o-ref).abs().max().item()
        tri=lambda: fused_prenorm[(M,)](x,w,b,torch.empty_like(x),eps,N,BLOCK_N=triton.next_power_of_2(N))
        tor=lambda: x+torch.nn.functional.layer_norm(x,(N,),w,b,eps)
        tm,_=time_fn(tri); tm_t,_=time_fn(tor)
        print(f"  M={M} N={N} err={err:.3e} triton={tm:.4f}ms torch={tm_t:.4f}ms torch/triton={tm_t/tm:.2f}x")
    print("--- Fused Add+SiLU (silu(a+b)) ---")
    for M,N in [(4096,4096),(8192,8192),(8192,14336)]:
        for dt in (torch.float32, torch.bfloat16):
            torch.manual_seed(0)
            a=torch.randn(M,N,device="cuda",dtype=dt)*0.3; b2=torch.randn(M,N,device="cuda",dtype=dt)*0.3
            ref=torch.nn.functional.silu(a+b2)
            o=torch.empty_like(a); fused_add_silu[(triton.cdiv(M*N,4096),)](a,b2,o,M*N,BLOCK=4096)
            err=(o.float()-ref.float()).abs().max().item()
            tri=lambda: fused_add_silu[(triton.cdiv(M*N,4096),)](a,b2,torch.empty_like(a),M*N,BLOCK=4096)
            tor=lambda: torch.nn.functional.silu(a+b2)
            tm,_=time_fn(tri); tm_t,_=time_fn(tor)
            print(f"  M={M} N={N} {str(dt).split('.')[-1]:8s} err={err:.3e} triton={tm:.4f}ms torch={tm_t:.4f}ms torch/triton={tm_t/tm:.2f}x")
if __name__=="__main__": main()
