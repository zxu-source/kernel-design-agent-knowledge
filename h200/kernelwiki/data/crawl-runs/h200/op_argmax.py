#!/usr/bin/env python3
"""H200 Per-Row Argmax (Triton) vs torch.argmax.
Sampling top-1: for each row, find the max value's index. Triton kernel does a
block-level argmax (tl.argmax) per row. PURPOSE = characterization (sampling
building block). vs torch.argmax on H200, Triton 3.6.
"""
import json, statistics
import torch, triton, triton.language as tl
@triton.jit
def argmax_row(x_ptr, idx_ptr, N, BLOCK_N: tl.constexpr):
    row=tl.program_id(0)
    offs=tl.arange(0, BLOCK_N); mask=offs<N
    x=tl.load(x_ptr+row*N+offs, mask=mask, other=-float('inf')).to(tl.float32)
    i=tl.argmax(x, axis=0)            # index of max within the row
    tl.store(idx_ptr+row, i.to(tl.int64))
def tri_argmax(x):
    M,N=x.shape; idx=torch.empty((M,),device=x.device,dtype=torch.int64)
    argmax_row[(M,)](x, idx, N, BLOCK_N=triton.next_power_of_2(N)); return idx
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
    for M,N in [(4096,4096),(8192,8192),(4096,32000),(8192,32000),(8192,128256)]:
        dt=torch.float32
        torch.manual_seed(0); x=torch.randn(M,N,device="cuda",dtype=dt)
        ref=torch.argmax(x,dim=-1)
        out=tri_argmax(x)
        nmatch=(out==ref).sum().item(); nacc=nmatch/M
        tri=lambda: tri_argmax(x); tor=lambda: torch.argmax(x,dim=-1)
        tm,_=time_fn(tri); tm_t,_=time_fn(tor)
        rows.append(dict(M=M,N=N,agree=round(nacc,5),triton_ms=round(tm,4),torch_ms=round(tm_t,4),torch_over_triton=round(tm_t/tm,3)))
        print(f"M={M} N={N:6d} agree={nacc:.4f} triton={tm:.4f}ms torch={tm_t:.4f}ms torch/triton={tm_t/tm:.2f}x")
    print("RESULT_JSON", json.dumps(rows))
if __name__=="__main__": main()
