#!/usr/bin/env python3
"""H200 Fused LayerNorm+GELU + Fused RMSNorm+RoPE (Triton) vs torch 2-op.
Fused-LN+GELU: layer_norm then GELU in one kernel (norm reduction + activation).
Fused-RMSNorm+RoPE: rms_norm then RoPE in one kernel (norm + positional rotate).
PURPOSE = speedup (fusion of two common transformer-block ops). H200, Triton 3.6.
"""
import json, statistics
import torch, triton, triton.language as tl

# ---- Fused LayerNorm + GELU (tanh) ----
@triton.jit
def fused_ln_gelu(x_ptr, w_ptr, b_ptr, o_ptr, eps, N, BLOCK_N: tl.constexpr):
    row=tl.program_id(0)
    offs=tl.arange(0, BLOCK_N); mask=offs<N
    x=tl.load(x_ptr+row*N+offs, mask=mask, other=0.0).to(tl.float32)
    mean=tl.sum(x, axis=0)/N
    var=tl.sum((x-mean)*(x-mean), axis=0)/N
    rrms=tl.rsqrt(var+eps)
    w=tl.load(w_ptr+offs, mask=mask, other=0.0).to(tl.float32)
    b=tl.load(b_ptr+offs, mask=mask, other=0.0).to(tl.float32)
    xn=(x-mean)*rrms*w+b
    # GELU tanh: 0.5*xn*(1+tanh(c*(xn+0.044715*xn^3)))
    c=0.7978845608028654
    inner=c*(xn+0.044715*xn*xn*xn)
    e2=tl.exp(2.0*inner); tanh=(e2-1.0)/(e2+1.0)
    y=0.5*xn*(1.0+tanh)
    tl.store(o_ptr+row*N+offs, y.to(o_ptr.dtype.element_ty), mask=mask)

# ---- Fused RMSNorm + RoPE (rotate-half) ----
@triton.jit
def fused_rmsnorm_rope(x_ptr, w_ptr, cos_ptr, sin_ptr, o_ptr, eps, NH, S, D,
                       stride_bh, stride_row, BLOCK_D: tl.constexpr):
    row=tl.program_id(0)
    d=tl.arange(0, BLOCK_D); mask=d<D; half=D//2
    x=tl.load(x_ptr+row*D+d, mask=mask, other=0.0).to(tl.float32)
    var=tl.sum(x*x, axis=0)/D
    rrms=tl.rsqrt(var+eps)
    w=tl.load(w_ptr+d, mask=mask, other=0.0).to(tl.float32)
    xn=x*rrms*w
    s=row%S
    cos=tl.load(cos_ptr+s*half+(d%half), mask=mask, other=0.0).to(tl.float32)
    sin=tl.load(sin_ptr+s*half+(d%half), mask=mask, other=0.0).to(tl.float32)
    lo=d<half; partner=tl.where(lo, d+half, d-half)
    xp=tl.load(x_ptr+row*D+partner, mask=mask, other=0.0).to(tl.float32)*rrms*tl.load(w_ptr+partner, mask=mask, other=0.0).to(tl.float32)
    y=tl.where(lo, xn*cos-xp*sin, xn*cos+xp*sin)
    tl.store(o_ptr+row*D+d, y.to(o_ptr.dtype.element_ty), mask=mask)

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
    # Fused LN+GELU
    print("--- Fused LayerNorm+GELU ---")
    for M,N in [(4096,4096),(8192,8192),(8192,14336)]:
        dt=torch.float32; torch.manual_seed(0)
        x=torch.randn(M,N,device="cuda",dtype=dt)*0.3
        w=torch.randn(N,device="cuda",dtype=dt)*0.1+1.0; b=torch.zeros(N,device="cuda",dtype=dt)
        ref=torch.nn.functional.gelu(torch.nn.functional.layer_norm(x,(N,),w,b,eps),approximate='tanh')
        o=torch.empty_like(x)
        fused_ln_gelu[(M,)](x,w,b,o,eps,N,BLOCK_N=triton.next_power_of_2(N))
        err=(o-ref).abs().max().item()
        tri=lambda: fused_ln_gelu[(M,)](x,w,b,torch.empty_like(x),eps,N,BLOCK_N=triton.next_power_of_2(N))
        tor=lambda: torch.nn.functional.gelu(torch.nn.functional.layer_norm(x,(N,),w,b,eps),approximate='tanh')
        tm,_=time_fn(tri); tm_t,_=time_fn(tor)
        print(f"  M={M} N={N} err={err:.3e} triton={tm:.4f}ms torch={tm_t:.4f}ms torch/triton={tm_t/tm:.2f}x")
    # Fused RMSNorm+RoPE
    print("--- Fused RMSNorm+RoPE ---")
    for B,H,S,D in [(1,32,4096,128),(2,32,2048,128),(1,8,4096,64)]:
        dt=torch.float32; torch.manual_seed(0)
        x=torch.randn(B,H,S,D,device="cuda",dtype=dt)*0.3
        w=torch.randn(D,device="cuda",dtype=dt)*0.1+1.0
        half=D//2
        inv_freq=1.0/(10000.0**(torch.arange(0,half,device="cuda",dtype=torch.float32)/half))
        pos=torch.arange(S,device="cuda",dtype=torch.float32)
        cos=(pos[:,None]*inv_freq[None,:]).cos(); sin=(pos[:,None]*inv_freq[None,:]).sin()
        # torch ref: rms_norm then rope
        xn=torch.nn.functional.rms_norm(x,(D,),w,eps)
        x1=xn[...,:half]; x2=xn[...,half:]; c=cos[None,None,:,:].expand_as(x1); s=sin[None,None,:,:].expand_as(x1)
        ref=torch.cat([x1*c-x2*s, x2*c+x1*s],dim=-1)
        # triton
        x2_flat=x.reshape(B*H*S,D).contiguous(); o2=torch.empty_like(x2_flat)
        fused_rmsnorm_rope[(B*H*S,)](x2_flat,w,cos,sin,o2,eps,H,S,D,x2_flat.stride(0),x2_flat.stride(1),BLOCK_D=triton.next_power_of_2(D))
        out=o2.reshape(B,H,S,D)
        err=(out-ref).abs().max().item()
        x2d=x2_flat
        tri=lambda: fused_rmsnorm_rope[(B*H*S,)](x2d,w,cos,sin,torch.empty_like(x2d),eps,H,S,D,x2d.stride(0),x2d.stride(1),BLOCK_D=triton.next_power_of_2(D))
        def tor():
            xn=torch.nn.functional.rms_norm(x,(D,),w,eps)
            x1=xn[...,:half]; x2=xn[...,half:]
            return torch.cat([x1*cos[None,None].expand_as(x1)-x2*sin[None,None].expand_as(x1), x2*cos[None,None].expand_as(x1)+x1*sin[None,None].expand_as(x1)],dim=-1)
        tm,_=time_fn(tri); tm_t,_=time_fn(tor)
        print(f"  B={B} H={H} S={S} D={D} err={err:.3e} triton={tm:.4f}ms torch={tm_t:.4f}ms torch/triton={tm_t/tm:.2f}x")
if __name__=="__main__": main()
