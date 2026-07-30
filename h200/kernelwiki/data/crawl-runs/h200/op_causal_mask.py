#!/usr/bin/env python3
"""H200 Causal Mask Generation (Triton) vs torch.
out[i,j] = 0 if j<=i else -inf (lower-triangular attention mask).
PURPOSE = both (mask gen + robustness: -inf for masked). H200, Triton 3.6.
"""
import json, statistics
import torch, triton, triton.language as tl
@triton.jit
def causal_mask(o_ptr, N, BLOCK_N: tl.constexpr):
    row=tl.program_id(0)
    offs=tl.arange(0, BLOCK_N); mask=offs<N
    vals=tl.where(offs <= row, 0.0, float('-inf'))
    tl.store(o_ptr + row*N + offs, vals, mask=mask)
def tri_mask(N, BLOCK_N=4096):
    o=torch.empty((N,N),device="cuda",dtype=torch.float32)
    causal_mask[(N,)](o, N, BLOCK_N=triton.next_power_of_2(N)); return o
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
    for N in [1024,2048,4096,8192]:
        ref=torch.triu(torch.ones(N,N,device="cuda",dtype=torch.float32)*float('-inf'),diagonal=1)
        out=tri_mask(N)
        # `-inf - -inf` is NaN, so compare the causal-mask semantics rather
        # than subtracting sentinel values.
        correct = torch.equal(torch.isneginf(out), torch.isneginf(ref)) and torch.equal(out == 0, ref == 0)
        err = 0.0 if correct else (out[torch.isfinite(ref)] - ref[torch.isfinite(ref)]).abs().max().item()
        tri=lambda: tri_mask(N); tor=lambda: torch.triu(torch.ones(N,N,device="cuda",dtype=torch.float32)*float('-inf'),diagonal=1)
        tm,_=time_fn(tri); tm_t,_=time_fn(tor)
        rows.append(dict(N=N,correct=bool(correct),err=round(err,5),triton_ms=round(tm,4),torch_ms=round(tm_t,4),torch_over_triton=round(tm_t/tm,3)))
        print(f"N={N:5d} correct={correct} err={err:.3e} triton={tm:.4f}ms torch={tm_t:.4f}ms torch/triton={tm_t/tm:.2f}x")
    print("RESULT_JSON", json.dumps(rows))
if __name__=="__main__": main()
