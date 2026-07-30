#!/usr/bin/env python3
"""H200 SiLU Backward (Triton) vs torch.autograd.
silu'(x) = sigmoid(x)*(1 + x*(1-sigmoid(x))); grad_x = grad_y*silu'(x).
PURPOSE = speedup. H200, Triton 3.6.
"""
import json, statistics
import torch, triton, triton.language as tl
@triton.jit
def silu_bwd(x_ptr, gy_ptr, gx_ptr, total, BLOCK: tl.constexpr):
    pid=tl.program_id(0); offs=pid*BLOCK+tl.arange(0,BLOCK); mask=offs<total
    x=tl.load(x_ptr+offs, mask=mask, other=0.0).to(tl.float32)
    gy=tl.load(gy_ptr+offs, mask=mask, other=0.0).to(tl.float32)
    sig=1.0/(1.0+tl.exp(-x))
    dsilu=sig*(1.0 + x*(1.0-sig))
    tl.store(gx_ptr+offs, (gy*dsilu).to(gx_ptr.dtype.element_ty), mask=mask)
def tri_sbwd(x, gy, BLOCK=4096):
    o=torch.empty_like(x); silu_bwd[(triton.cdiv(x.numel(),BLOCK),)](x,gy,o,x.numel(),BLOCK=BLOCK); return o
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
    for M,N in [(4096,4096),(8192,8192),(8192,11008),(8192,14336),(16384,14336)]:
        for dt in (torch.float32, torch.bfloat16):
            torch.manual_seed(0); x=(torch.randn(M,N,device="cuda",dtype=dt)*0.5).detach().requires_grad_(True)
            y=torch.nn.functional.silu(x); gy=torch.randn_like(y)
            y.backward(gy); rxg=x.grad.detach().clone(); x.grad=None
            out=tri_sbwd(x.detach(), gy)
            err=(out.float()-rxg.float()).abs().max().item()
            tri=lambda: tri_sbwd(x.detach(),gy)
            xt=x.detach().clone().requires_grad_(True); yt=torch.nn.functional.silu(xt)
            def tor():
                if xt.grad is not None: xt.grad=None
                yt.backward(gy, retain_graph=True)
            tm,_=time_fn(tri); tm_t,_=time_fn(tor)
            rows.append(dict(M=M,N=N,dtype=str(dt).split(".")[-1],err=round(err,5),triton_ms=round(tm,4),torch_ms=round(tm_t,4),torch_over_triton=round(tm_t/tm,3)))
            print(f"M={M} N={N} {str(dt).split('.')[-1]:8s} err={err:.3e} triton={tm:.4f}ms torch={tm_t:.4f}ms torch/triton={tm_t/tm:.2f}x")
    print("RESULT_JSON", json.dumps(rows))
if __name__=="__main__": main()
