#!/usr/bin/env python3
"""H200 Embedding Lookup (gather) (Triton) vs torch.nn.functional.embedding.
out[i,:] = table[idx[i],:]. Memory-bound gather (one program per row).
PURPOSE = characterization. vs torch embedding on H200, Triton 3.6.
"""
import json, statistics
import torch, triton, triton.language as tl
@triton.jit
def embed_lookup(tbl_ptr, idx_ptr, o_ptr, V, D, BLOCK_D: tl.constexpr):
    i=tl.program_id(0)
    src=tl.load(idx_ptr+i)
    d=tl.arange(0, BLOCK_D); mask=d<D
    v=tl.load(tbl_ptr + src*D + d, mask=mask, other=0.0)
    tl.store(o_ptr + i*D + d, v, mask=mask)
def tri_embed(tbl, idx, D):
    M=idx.shape[0]; o=torch.empty((M,D),device=tbl.device,dtype=tbl.dtype)
    embed_lookup[(M,)](tbl, idx, o, tbl.shape[0], D, BLOCK_D=triton.next_power_of_2(D)); return o
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
    for M,V,D in [(8192,128256,4096),(4096,128256,4096),(8192,32000,4096),(8192,128256,8192),(16384,128256,4096)]:
        for dt in (torch.float32, torch.bfloat16):
            torch.manual_seed(0); tbl=torch.randn(V,D,device="cuda",dtype=dt)*0.1
            idx=torch.randint(0,V,(M,),device="cuda")
            ref=tbl[idx]; out=tri_embed(tbl,idx,D)
            err=(out-ref).abs().max().item()
            tri=lambda: tri_embed(tbl,idx,D); tor=lambda: tbl[idx]
            tm,_=time_fn(tri); tm_t,_=time_fn(tor)
            rows.append(dict(M=M,V=V,D=D,dtype=str(dt).split(".")[-1],err=round(err,5),triton_ms=round(tm,4),torch_ms=round(tm_t,4),torch_over_triton=round(tm_t/tm,3)))
            print(f"M={M} V={V} D={D} {str(dt).split('.')[-1]:8s} err={err:.2e} triton={tm:.4f}ms torch={tm_t:.4f}ms torch/triton={tm_t/tm:.2f}x")
    print("RESULT_JSON", json.dumps(rows))
if __name__=="__main__": main()
