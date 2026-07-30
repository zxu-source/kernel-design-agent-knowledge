#!/usr/bin/env python3
"""H200 KL Divergence + Cumprod (Triton) vs torch.
KL-div: loss = target*(log(target) - log_softmax(logits)). Fused log_softmax + target gather.
Cumprod: exp(cumsum(log(x))) per row (Triton 3.6 has no tl.cumprod).
PURPOSE = speedup/characterization. H200, Triton 3.6.
"""
import json, statistics
import torch, triton, triton.language as tl

# ---- KL-div (forward, per-element) ----
@triton.jit
def kl_div(logit_ptr, tgt_ptr, o_ptr, N, BLOCK_N: tl.constexpr):
    row=tl.program_id(0)
    offs=tl.arange(0, BLOCK_N); mask=offs<N
    x=tl.load(logit_ptr+row*N+offs, mask=mask, other=-1e30).to(tl.float32)
    t=tl.load(tgt_ptr+row*N+offs, mask=mask, other=0.0).to(tl.float32)
    m=tl.max(x, axis=0); e=tl.where(mask, tl.exp(x-m), 0.0); lse=m+tl.log(tl.sum(e, axis=0))
    log_sm=x-lse
    loss = t * (tl.log(t+1e-30) - log_sm)
    tl.store(o_ptr+row*N+offs, loss, mask=mask)

def tri_kl(logits, target, N):
    M=logits.shape[0]; o=torch.empty_like(logits)
    kl_div[(M,)](logits, target, o, N, BLOCK_N=triton.next_power_of_2(N)); return o

def time_fn(fn,trials=50):
    for _ in range(10): fn(); torch.cuda.synchronize()
    ts=[]
    for _ in range(trials):
        s,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    return min(ts), statistics.median(ts)

def main():
    print(f"triton={triton.__version__} dev={torch.cuda.get_device_name(0)} SMs={torch.cuda.get_device_properties(0).multi_processor_count}")
    # KL-div
    print("--- KL-divergence ---")
    for M,N in [(4096,4096),(8192,8192),(4096,32000)]:
        torch.manual_seed(0)
        logits=torch.randn(M,N,device="cuda",dtype=torch.float32)
        target=torch.softmax(torch.randn(M,N,device="cuda",dtype=torch.float32),dim=-1)
        ref=torch.nn.functional.kl_div(torch.log_softmax(logits,dim=-1), target, reduction='none')
        out=tri_kl(logits,target,N)
        err=(out-ref).abs().max().item()
        tri=lambda: tri_kl(logits,target,N)
        tor=lambda: torch.nn.functional.kl_div(torch.log_softmax(logits,dim=-1), target, reduction='none')
        tm,_=time_fn(tri); tm_t,_=time_fn(tor)
        print(f"  M={M} N={N:6d} err={err:.3e} triton={tm:.4f}ms torch={tm_t:.4f}ms torch/triton={tm_t/tm:.2f}x")
    # Cumprod
    print("--- Cumprod ---")
    @triton.jit
    def cumprod_k(x_ptr, o_ptr, N, BLOCK_N: tl.constexpr):
        row=tl.program_id(0)
        offs=tl.arange(0, BLOCK_N); mask=offs<N
        x=tl.load(x_ptr+row*N+offs, mask=mask, other=1.0).to(tl.float32)
        cp=tl.exp(tl.cumsum(tl.log(x), axis=0))
        tl.store(o_ptr+row*N+offs, cp.to(o_ptr.dtype.element_ty), mask=mask)
    def tri_cp(x):
        M,N=x.shape; o=torch.empty_like(x)
        cumprod_k[(M,)](x,o,N,BLOCK_N=triton.next_power_of_2(N)); return o
    for M,N in [(4096,4096),(8192,8192)]:
        torch.manual_seed(0)
        x=torch.rand(M,N,device="cuda",dtype=torch.float32)*2+0.1  # positive (for log)
        ref=torch.cumprod(x,dim=-1)
        out=tri_cp(x)
        err=(out-ref).abs().max().item()
        rel=err/(ref.abs().max().item()+1e-9)
        tri=lambda: tri_cp(x); tor=lambda: torch.cumprod(x,dim=-1)
        tm,_=time_fn(tri); tm_t,_=time_fn(tor)
        print(f"  M={M} N={N:6d} err={err:.3e} rel={rel:.2e} triton={tm:.4f}ms torch={tm_t:.4f}ms torch/triton={tm_t/tm:.2f}x")
if __name__=="__main__": main()
