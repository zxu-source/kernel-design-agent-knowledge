#!/usr/bin/env python3
"""H200 BCE-with-Logits Loss (Triton) vs torch.
loss = max(x,0) - x*target + log(1+exp(-|x|)) (numerically stable BCE-with-logits).
Fused elementwise. PURPOSE = speedup. vs torch.nn.functional.binary_cross_entropy_with_logits.
H200, Triton 3.6.
"""
import json, statistics
import torch, triton, triton.language as tl
@triton.jit
def bce_logits(x_ptr, t_ptr, o_ptr, total, BLOCK: tl.constexpr):
    pid=tl.program_id(0); offs=pid*BLOCK+tl.arange(0,BLOCK); mask=offs<total
    x=tl.load(x_ptr+offs, mask=mask, other=0.0).to(tl.float32)
    t=tl.load(t_ptr+offs, mask=mask, other=0.0).to(tl.float32)
    loss=tl.maximum(x, 0.0) - x*t + tl.log(1.0+tl.exp(-tl.abs(x)))
    tl.store(o_ptr+offs, loss, mask=mask)
def tri_bce(logits, target, BLOCK=4096):
    o=torch.empty_like(logits); bce_logits[(triton.cdiv(logits.numel(),BLOCK),)](logits,target,o,logits.numel(),BLOCK=BLOCK); return o
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
    for M,N in [(4096,4096),(8192,8192),(8192,14336),(16384,14336)]:
        torch.manual_seed(0)
        logits=torch.randn(M,N,device="cuda",dtype=torch.float32)
        target=torch.randint(0,2,(M,N),device="cuda",dtype=torch.float32)
        ref=torch.nn.functional.binary_cross_entropy_with_logits(logits,target,reduction='none')
        out=tri_bce(logits,target)
        err=(out-ref).abs().max().item()
        tri=lambda: tri_bce(logits,target)
        tor=lambda: torch.nn.functional.binary_cross_entropy_with_logits(logits,target,reduction='none')
        tm,_=time_fn(tri); tm_t,_=time_fn(tor)
        rows.append(dict(M=M,N=N,err=round(err,6),triton_ms=round(tm,4),torch_ms=round(tm_t,4),torch_over_triton=round(tm_t/tm,3)))
        print(f"M={M} N={N} err={err:.3e} triton={tm:.4f}ms torch={tm_t:.4f}ms torch/triton={tm_t/tm:.2f}x")
    print("RESULT_JSON", json.dumps(rows))
if __name__=="__main__": main()
