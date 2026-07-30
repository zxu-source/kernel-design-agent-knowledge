#!/usr/bin/env python3
"""H200 Fused Concat (Triton) vs torch.cat.
Concatenate K tensors along the last dim into one output in a single kernel
(common: QKV-projection output [Q;K;V] -> concat). vs torch.cat (which may launch
per-tensor copies). PURPOSE = speedup (one fused launch). H200, Triton 3.6.
"""
import json, statistics
import torch, triton, triton.language as tl
def cdiv(a,b): return (a+b-1)//b
@triton.jit
def concat3(a_ptr,b_ptr,c_ptr,o_ptr, M, Dk, D, BLOCK: tl.constexpr):
    # one program per row: copy a/b/c [m,0:Dk] into out [m, 0:Dk|Dk:2Dk|2Dk:3Dk]
    row=tl.program_id(0)
    j=tl.arange(0, BLOCK); mask=j<Dk
    ar=tl.load(a_ptr+row*Dk+j, mask=mask)
    br=tl.load(b_ptr+row*Dk+j, mask=mask)
    cr=tl.load(c_ptr+row*Dk+j, mask=mask)
    tl.store(o_ptr+row*D + j,       ar, mask=mask)   # out[m, 0:Dk)
    tl.store(o_ptr+row*D + (Dk+j),  br, mask=mask)   # out[m, Dk:2Dk)
    tl.store(o_ptr+row*D + (2*Dk+j),cr, mask=mask)   # out[m, 2Dk:3Dk)
def tri_concat(a,b,c, BLOCK=None):
    M,Dk=a.shape; D=3*Dk
    o=torch.empty((M,D),device=a.device,dtype=a.dtype)
    concat3[(M,)](a,b,c,o,M,Dk,D,BLOCK=triton.next_power_of_2(Dk)); return o
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
    for M,Dk in [(4096,4096),(8192,4096),(8192,8192),(8192,14336),(16384,14336)]:
        for dt in (torch.float32, torch.bfloat16):
            a=torch.randn(M,Dk,device="cuda",dtype=dt); b=torch.randn(M,Dk,device="cuda",dtype=dt); c=torch.randn(M,Dk,device="cuda",dtype=dt)
            ref=torch.cat([a,b,c],dim=-1); out=tri_concat(a,b,c)
            err=(out-ref).abs().max().item()
            tri=lambda: tri_concat(a,b,c); tor=lambda: torch.cat([a,b,c],dim=-1)
            tm,_=time_fn(tri); tm_t,_=time_fn(tor)
            rows.append(dict(M=M,Dk=Dk,dtype=str(dt).split(".")[-1],err=round(err,5),triton_ms=round(tm,4),torch_ms=round(tm_t,4),torch_over_triton=round(tm_t/tm,3)))
            print(f"M={M} Dk={Dk} {str(dt).split('.')[-1]:8s} err={err:.2e} triton={tm:.4f}ms torch={tm_t:.4f}ms torch/triton={tm_t/tm:.2f}x")
    print("RESULT_JSON", json.dumps(rows))
if __name__=="__main__": main()
