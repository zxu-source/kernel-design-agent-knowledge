#!/usr/bin/env python3
"""H200 GELU (tanh) Backward (Triton) vs torch.autograd.
gelu'(x) = 0.5*(1+tanh(g)) + 0.5*x*(1-tanh(g)^2)*g'(x),
g(x)=c*(x+0.044715*x^3), c=sqrt(2/pi), g'(x)=c*(1+0.134145*x^2).
grad_x = grad_y * gelu'(x). PURPOSE = speedup. H200, Triton 3.6.
"""
import json, statistics
import torch, triton, triton.language as tl
@triton.jit
def gelu_bwd(x_ptr, gy_ptr, gx_ptr, total, BLOCK: tl.constexpr):
    pid=tl.program_id(0); offs=pid*BLOCK+tl.arange(0,BLOCK); mask=offs<total
    x=tl.load(x_ptr+offs, mask=mask, other=0.0).to(tl.float32)
    gy=tl.load(gy_ptr+offs, mask=mask, other=0.0).to(tl.float32)
    c=0.7978845608028654
    inner=c*(x+0.044715*x*x*x)
    e2=tl.exp(2.0*inner); tanh=(e2-1.0)/(e2+1.0)
    gp=c*(1.0+0.134145*x*x)
    dgelu=0.5*(1.0+tanh) + 0.5*x*(1.0-tanh*tanh)*gp
    gx=gy*dgelu
    tl.store(gx_ptr+offs, gx.to(gx_ptr.dtype.element_ty), mask=mask)
def tri_gbwd(x, gy):
    o=torch.empty_like(x); total=x.numel(); BLOCK=4096
    gelu_bwd[(triton.cdiv(total,BLOCK),)](x,gy,o,total,BLOCK=BLOCK); return o
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
            torch.manual_seed(0); x=(torch.randn(M,N,device="cuda",dtype=dt)*0.7).detach().requires_grad_(True)
            y=torch.nn.functional.gelu(x, approximate='tanh'); gy=torch.randn_like(y)
            y.backward(gy); rxg=x.grad.detach().clone(); x.grad=None
            out=tri_gbwd(x.detach(), gy)
            err=(out.float()-rxg.float()).abs().max().item()
            tri=lambda: tri_gbwd(x.detach(),gy)
            xt=x.detach().clone().requires_grad_(True)
            yt=torch.nn.functional.gelu(xt, approximate='tanh')
            def tor():
                if xt.grad is not None: xt.grad=None
                yt.backward(gy, retain_graph=True)
            tm,_=time_fn(tri); tm_t,_=time_fn(tor)
            rows.append(dict(M=M,N=N,dtype=str(dt).split(".")[-1],err=round(err,5),triton_ms=round(tm,4),torch_ms=round(tm_t,4),torch_over_triton=round(tm_t/tm,3)))
            print(f"M={M} N={N} {str(dt).split('.')[-1]:8s} err={err:.3e} triton={tm:.4f}ms torch={tm_t:.4f}ms torch/triton={tm_t/tm:.2f}x")
    print("RESULT_JSON", json.dumps(rows))
if __name__=="__main__": main()
