#!/usr/bin/env python3
"""H200 Scatter-Add (index_add, embedding-grad style) (Triton) vs torch.
out[idx[i], :] += values[i, :] (scatter-add into [V,D] by index). One program per
row (token), atomic_add the D-vector. Used in embedding grad / MoE combine.
PURPOSE = characterization. vs torch.index_add on H200, Triton 3.6.
"""
import json, statistics
import torch, triton, triton.language as tl
@triton.jit
def scatter_add(val_ptr, idx_ptr, out_ptr, D, BLOCK_D: tl.constexpr):
    i=tl.program_id(0)
    dst=tl.load(idx_ptr+i)
    d=tl.arange(0, BLOCK_D); mask=d<D
    v=tl.load(val_ptr+i*D+d, mask=mask, other=0.0).to(tl.float32)
    tl.atomic_add(out_ptr + dst*D + d, v, mask=mask)
def tri_scatter(values, idx, V, D):
    out=torch.zeros((V,D),device=values.device,dtype=torch.float32)
    scatter_add[(values.shape[0],)](values, idx, out, D, BLOCK_D=triton.next_power_of_2(D)); return out
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
    for M,V,D in [(8192,128256,4096),(4096,128256,4096),(8192,32000,4096),(16384,128256,4096),(8192,128256,8192)]:
        torch.manual_seed(0)
        values=torch.randn(M,D,device="cuda",dtype=torch.float32)
        idx=torch.randint(0,V,(M,),device="cuda")
        ref=torch.zeros((V,D),device="cuda",dtype=torch.float32).index_add_(0, idx, values)
        out=tri_scatter(values, idx, V, D)
        err=(out-ref).abs().max().item(); rel=err/(ref.abs().max().item()+1e-9)
        tri=lambda: tri_scatter(values,idx,V,D); tor=lambda: torch.zeros((V,D),device="cuda",dtype=torch.float32).index_add_(0,idx,values)
        tm,_=time_fn(tri); tm_t,_=time_fn(tor)
        rows.append(dict(M=M,V=V,D=D,err=round(err,5),rel=round(rel,6),triton_ms=round(tm,4),torch_ms=round(tm_t,4),torch_over_triton=round(tm_t/tm,3)))
        print(f"M={M} V={V} D={D} err={err:.3e} rel={rel:.2e} triton={tm:.4f}ms torch={tm_t:.4f}ms torch/triton={tm_t/tm:.2f}x")
    print("RESULT_JSON", json.dumps(rows))
if __name__=="__main__": main()
