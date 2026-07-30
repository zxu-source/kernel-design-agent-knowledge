#!/usr/bin/env python3
"""H200 Max Pooling (2x2 stride2) (Triton) vs torch.max_pool2d.
Each output (c,oh,ow) = max over the 2x2 window. PURPOSE = characterization.
vs torch.nn.functional.max_pool2d on H200, Triton 3.6.
"""
import json, statistics
import torch, triton, triton.language as tl
@triton.jit
def maxpool2d(in_ptr, o_ptr, C, H, W, OH, OW, BLOCK: tl.constexpr):
    c=tl.program_id(0); pid=tl.program_id(1)
    p=pid*BLOCK+tl.arange(0, BLOCK); mask=p<OH*OW
    oh=p//OW; ow=p%OW
    ih=oh*2; iw=ow*2
    base=c*H*W + ih*W + iw
    a=tl.load(in_ptr+base, mask=mask, other=-1e30)
    b=tl.load(in_ptr+base+1, mask=mask&(iw+1<W), other=-1e30)
    cc=tl.load(in_ptr+base+W, mask=mask&(ih+1<H), other=-1e30)
    d=tl.load(in_ptr+base+W+1, mask=mask&(ih+1<H)&(iw+1<W), other=-1e30)
    m=tl.maximum(tl.maximum(a,b), tl.maximum(cc,d))
    tl.store(o_ptr + c*OH*OW + p, m, mask=mask)
def tri_mp(x, BLOCK=4096):  # x: [C,H,W]
    C,H,W=x.shape; OH,OW=H//2,W//2; o=torch.empty((C,OH,OW),device=x.device,dtype=x.dtype)
    maxpool2d[(C, triton.cdiv(OH*OW,BLOCK))](x,o,C,H,W,OH,OW,BLOCK=BLOCK); return o
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
    for C,H,W in [(64,128,128),(128,128,128),(256,64,64),(128,256,256),(64,512,512)]:
        torch.manual_seed(0); x=torch.randn(1,C,H,W,device="cuda",dtype=torch.float32)
        ref=torch.nn.functional.max_pool2d(x,2,2)[0]
        out=tri_mp(x[0])
        err=(out-ref).abs().max().item()
        tri=lambda: tri_mp(x[0]); tor=lambda: torch.nn.functional.max_pool2d(x,2,2)
        tm,_=time_fn(tri); tm_t,_=time_fn(tor)
        rows.append(dict(C=C,H=H,W=W,err=round(err,4),triton_ms=round(tm,4),torch_ms=round(tm_t,4),torch_over_triton=round(tm_t/tm,3)))
        print(f"C={C} H={H} W={W} err={err:.3e} triton={tm:.4f}ms torch={tm_t:.4f}ms torch/triton={tm_t/tm:.2f}x")
    print("RESULT_JSON", json.dumps(rows))
if __name__=="__main__": main()
