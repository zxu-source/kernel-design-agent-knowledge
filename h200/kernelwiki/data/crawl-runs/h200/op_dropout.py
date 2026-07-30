#!/usr/bin/env python3
"""H200 Fused Dropout (Philox RNG + mask + scale) (Triton) vs torch.
y = x * mask / (1-p), mask ~ Bernoulli(1-p). Fused: RNG + compare + scale in one
kernel. Correctness: STATISTICAL (zero_fraction ~ p, nonzero = x/(1-p), mean
preserved) — torch uses a different RNG so exact values differ.
PURPOSE = speedup. H200, Triton 3.6.
"""
import json, statistics as st
import torch, triton, triton.language as tl
@triton.jit
def dropout(x_ptr, o_ptr, p, seed, total, BLOCK: tl.constexpr):
    pid=tl.program_id(0); offs=pid*BLOCK+tl.arange(0,BLOCK); mask=offs<total
    x=tl.load(x_ptr+offs, mask=mask, other=0.0).to(tl.float32)
    rand=tl.rand(seed, offs)                       # Philox uniform [0,1)
    keep = rand >= p                               # Bernoulli(1-p)
    scale = 1.0/(1.0-p)
    y = tl.where(keep, x*scale, 0.0)
    tl.store(o_ptr+offs, y.to(o_ptr.dtype.element_ty), mask=mask)
def tri_dropout(x, p, seed, BLOCK=4096):
    o=torch.empty_like(x); dropout[(triton.cdiv(x.numel(),BLOCK),)](x,o,p,seed,x.numel(),BLOCK=BLOCK); return o
def time_fn(fn,trials=50):
    for _ in range(10): fn(); torch.cuda.synchronize()
    ts=[]
    for _ in range(trials):
        s,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    return min(ts), st.median(ts)
def main():
    print(f"triton={triton.__version__} dev={torch.cuda.get_device_name(0)} SMs={torch.cuda.get_device_properties(0).multi_processor_count}")
    rows=[]
    for M,N in [(4096,4096),(8192,8192),(8192,14336),(16384,14336)]:
        for dt in (torch.float32, torch.bfloat16):
            torch.manual_seed(0); x=torch.randn(M,N,device="cuda",dtype=dt)*0.5
            p=0.1
            out=tri_dropout(x, p, seed=42)
            # statistical correctness
            zero_frac=(out==0).float().mean().item()
            nonzeros=out[out!=0]
            expected_val = x[out!=0] / (1.0-p)
            val_err=(nonzeros.float()-expected_val.float()).abs().max().item()
            mean_ratio=out.float().mean().item()/(x.float().mean().item()+1e-9)
            tri=lambda: tri_dropout(x, p, seed=42)
            tor=lambda: torch.nn.functional.dropout(x, p=p, training=True)
            tm,_=time_fn(tri); tm_t,_=time_fn(tor)
            rows.append(dict(M=M,N=N,dtype=str(dt).split(".")[-1],zero_frac=round(zero_frac,4),
                             val_err=round(val_err,4),mean_ratio=round(mean_ratio,3),
                             triton_ms=round(tm,4),torch_ms=round(tm_t,4),torch_over_triton=round(tm_t/tm,3)))
            print(f"M={M} N={N} {str(dt).split('.')[-1]:8s} zero_frac={zero_frac:.4f}(~p={p}) val_err={val_err:.2e} mean_ratio={mean_ratio:.3f} triton={tm:.4f}ms torch={tm_t:.4f}ms torch/triton={tm_t/tm:.2f}x")
    print("RESULT_JSON", json.dumps(rows))
if __name__=="__main__": main()
