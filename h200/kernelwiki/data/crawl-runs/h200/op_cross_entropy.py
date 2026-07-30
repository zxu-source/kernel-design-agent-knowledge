#!/usr/bin/env python3
"""H200 Cross-Entropy Loss forward (Triton) vs torch.
loss[i] = logsumexp(logits[i]) - logits[i, target[i]]. Fused: one pass per row
(max-subtract, exp, sum, log -> lse; gather target logit). PURPOSE = speedup
(fused logsumexp + gather vs torch's log_softmax + gather + neg). H200, Triton 3.6.
"""
import json, statistics
import torch, triton, triton.language as tl
@triton.jit
def ce_fwd(logits_ptr, tgt_ptr, loss_ptr, D, BLOCK_D: tl.constexpr):
    row=tl.program_id(0)
    offs=tl.arange(0, BLOCK_D); mask=offs<D
    x=tl.load(logits_ptr+row*D+offs, mask=mask, other=-1e30).to(tl.float32)
    m=tl.max(x, axis=0)
    e=tl.exp(x-m); e=tl.where(mask, e, 0.0)
    lse=m+tl.log(tl.sum(e, axis=0))
    tgt=tl.load(tgt_ptr+row)
    xt=tl.load(logits_ptr+row*D+tgt).to(tl.float32)
    loss=lse-xt
    tl.store(loss_ptr+row, loss)
def tri_ce(logits, target):
    M,D=logits.shape; loss=torch.empty((M,),device=logits.device,dtype=torch.float32)
    ce_fwd[(M,)](logits, target, loss, D, BLOCK_D=triton.next_power_of_2(D)); return loss
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
    for M,D in [(4096,4096),(8192,32000),(8192,128256),(4096,128256),(8192,4096)]:
        torch.manual_seed(0); logits=torch.randn(M,D,device="cuda",dtype=torch.float32)
        target=torch.randint(0,D,(M,),device="cuda")
        ref=torch.nn.functional.cross_entropy(logits, target, reduction='none')
        out=tri_ce(logits, target)
        err=(out-ref).abs().max().item()
        tri=lambda: tri_ce(logits,target); tor=lambda: torch.nn.functional.cross_entropy(logits,target,reduction='none')
        tm,_=time_fn(tri); tm_t,_=time_fn(tor)
        rows.append(dict(M=M,D=D,err=round(err,5),triton_ms=round(tm,4),torch_ms=round(tm_t,4),torch_over_triton=round(tm_t/tm,3)))
        print(f"M={M} D={D:6d} err={err:.3e} triton={tm:.4f}ms torch={tm_t:.4f}ms torch/triton={tm_t/tm:.2f}x")
    print("RESULT_JSON", json.dumps(rows))
if __name__=="__main__": main()
