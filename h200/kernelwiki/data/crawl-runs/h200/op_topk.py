#!/usr/bin/env python3
"""H200 Per-Row Top-K Selection (Triton) vs torch.topk.
k-pass selection: iteratively find row max + argmax, mask it (-inf), repeat k
times. O(k*N) per row; good for small k (top-k sampling/decoding). PURPOSE =
characterization. vs torch.topk(k) on H200, Triton 3.6.
"""
import json, statistics
import torch, triton, triton.language as tl
@triton.jit
def topk(x_ptr, val_ptr, idx_ptr, N, K: tl.constexpr, BLOCK_N: tl.constexpr):
    row=tl.program_id(0)
    offs=tl.arange(0, BLOCK_N); mask=offs<N
    x=tl.load(x_ptr+row*N+offs, mask=mask, other=-1e30).to(tl.float32)
    for k in tl.static_range(K):
        m=tl.max(x, axis=0)
        sel = x == m
        # argmax = first index where x==m (only one since unique after masking)
        idx=tl.argmax(tl.where(sel, offs.to(tl.float32), -1e30), axis=0)
        # store val[k], idx[k] for this row
        tl.store(val_ptr + row*K + k, m)
        tl.store(idx_ptr + row*K + k, idx.to(tl.int64))
        # mask out the picked element
        x=tl.where(sel, -1e30, x)
def tri_topk(x, K):
    M,N=x.shape
    val=torch.empty((M,K),device=x.device,dtype=torch.float32); idx=torch.empty((M,K),device=x.device,dtype=torch.int64)
    topk[(M,)](x,val,idx,N,K=K,BLOCK_N=triton.next_power_of_2(N))
    return val, idx
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
    K=5
    for M,N in [(4096,4096),(8192,8192),(4096,32000),(8192,32000)]:
        torch.manual_seed(0); x=torch.randn(M,N,device="cuda",dtype=torch.float32)
        rv,ri=torch.topk(x,K,dim=-1)
        tv,ti=tri_topk(x,K)
        # compare as sets (top-k may differ in tie order); compare sorted values
        val_match=(torch.sort(tv,dim=-1).values - torch.sort(rv,dim=-1).values).abs().max().item()
        # idx set match fraction
        def setmatch(a,b):
            mm=0; tot=0
            for r in range(a.shape[0]):
                mm += len(set(a[r].tolist()) & set(b[r].tolist())); tot+=K
            return mm/tot
        sm=setmatch(ti, ri)
        tri=lambda: tri_topk(x,K); tor=lambda: torch.topk(x,K,dim=-1)
        tm,_=time_fn(tri); tm_t,_=time_fn(tor)
        rows.append(dict(M=M,N=N,K=K,val_err=round(val_match,4),set_match=round(sm,4),triton_ms=round(tm,4),torch_ms=round(tm_t,4),torch_over_triton=round(tm_t/tm,3)))
        print(f"M={M} N={N:6d} K={K} val_err={val_match:.3e} set_match={sm:.4f} triton={tm:.4f}ms torch={tm_t:.4f}ms torch/triton={tm_t/tm:.2f}x")
    print("RESULT_JSON", json.dumps(rows))
if __name__=="__main__": main()
