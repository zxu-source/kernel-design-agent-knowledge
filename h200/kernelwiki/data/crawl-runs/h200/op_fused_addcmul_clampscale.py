#!/usr/bin/env python3
"""H200 Fused Addcmul + Fused Clamp+Scale (Triton) vs torch.
Addcmul: out = c + a*b (FMA pattern; optimizer helper, interpolation).
Clamp+Scale: out = clamp(x, lo, hi) * scale (mixed-precision safety; clamp before quant).
PURPOSE = speedup/both. H200, Triton 3.6.
"""
import json, statistics
import torch, triton, triton.language as tl

@triton.jit
def fused_addcmul(c_ptr, a_ptr, b_ptr, o_ptr, total, BLOCK: tl.constexpr):
    pid=tl.program_id(0); offs=pid*BLOCK+tl.arange(0,BLOCK); mask=offs<total
    c=tl.load(c_ptr+offs, mask=mask, other=0.0).to(tl.float32)
    a=tl.load(a_ptr+offs, mask=mask, other=0.0).to(tl.float32)
    b=tl.load(b_ptr+offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(o_ptr+offs, (c+a*b).to(o_ptr.dtype.element_ty), mask=mask)

@triton.jit
def fused_clamp_scale(x_ptr, o_ptr, lo, hi, scale, total, BLOCK: tl.constexpr):
    pid=tl.program_id(0); offs=pid*BLOCK+tl.arange(0,BLOCK); mask=offs<total
    x=tl.load(x_ptr+offs, mask=mask, other=0.0).to(tl.float32)
    clamped=tl.minimum(tl.maximum(x, lo), hi)
    tl.store(o_ptr+offs, (clamped*scale).to(o_ptr.dtype.element_ty), mask=mask)

def time_fn(fn, trials=50):
    for _ in range(10): fn(); torch.cuda.synchronize()
    ts=[]
    for _ in range(trials):
        s,e=torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    return min(ts), statistics.median(ts)

def main():
    print(f"triton={triton.__version__} dev={torch.cuda.get_device_name(0)} SMs={torch.cuda.get_device_properties(0).multi_processor_count}")
    print("--- Fused Addcmul (c+a*b) ---")
    for M,N in [(4096,4096),(8192,8192),(8192,14336)]:
        for dt in (torch.float32, torch.bfloat16):
            torch.manual_seed(0)
            a=torch.randn(M,N,device="cuda",dtype=dt)*0.3; b=torch.randn(M,N,device="cuda",dtype=dt)*0.3; c=torch.randn(M,N,device="cuda",dtype=dt)*0.1
            ref=torch.addcmul(c, a, b)
            o=torch.empty_like(c); fused_addcmul[(triton.cdiv(M*N,4096),)](c,a,b,o,M*N,BLOCK=4096)
            err=(o.float()-ref.float()).abs().max().item()
            tri=lambda: fused_addcmul[(triton.cdiv(M*N,4096),)](c,a,b,torch.empty_like(c),M*N,BLOCK=4096)
            tor=lambda: torch.addcmul(c,a,b)
            tm,_=time_fn(tri); tm_t,_=time_fn(tor)
            print(f"  M={M} N={N} {str(dt).split('.')[-1]:8s} err={err:.3e} triton={tm:.4f}ms torch={tm_t:.4f}ms torch/triton={tm_t/tm:.2f}x")
    print("--- Fused Clamp+Scale ---")
    for M,N in [(4096,4096),(8192,8192),(8192,14336)]:
        dt=torch.float32; torch.manual_seed(0)
        x=torch.randn(M,N,device="cuda",dtype=dt)*10.0
        lo,hi,scale=-5.0,5.0,0.1
        ref=torch.clamp(x,lo,hi)*scale
        o=torch.empty_like(x); fused_clamp_scale[(triton.cdiv(M*N,4096),)](x,o,lo,hi,scale,M*N,BLOCK=4096)
        err=(o-ref).abs().max().item()
        tri=lambda: fused_clamp_scale[(triton.cdiv(M*N,4096),)](x,torch.empty_like(x),lo,hi,scale,M*N,BLOCK=4096)
        tor=lambda: torch.clamp(x,lo,hi)*scale
        tm,_=time_fn(tri); tm_t,_=time_fn(tor)
        print(f"  M={M} N={N} err={err:.3e} triton={tm:.4f}ms torch={tm_t:.4f}ms torch/triton={tm_t/tm:.2f}x")
if __name__=="__main__": main()
