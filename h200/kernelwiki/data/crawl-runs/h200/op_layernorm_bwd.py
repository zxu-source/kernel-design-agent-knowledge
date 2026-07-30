#!/usr/bin/env python3
"""H200 LayerNorm Backward (Triton) vs torch.autograd.
grad_x = rrms*(grad_y*w - c1 - (x-mean)*c2*rrms^2), c1=mean(grad_y*w),
c2=mean(grad_y*w*(x-mean)); grad_w += sum(grad_y*(x-mean)*rrms); grad_b += sum(grad_y).
PURPOSE = speedup. vs torch autograd on H200, Triton 3.6.
"""
import json, statistics
import torch, triton, triton.language as tl
@triton.jit
def layernorm_bwd(x_ptr, w_ptr, b_ptr, gy_ptr, gx_ptr, gw_ptr, gb_ptr, eps, M, N, BLOCK_N: tl.constexpr):
    row=tl.program_id(0)
    offs=tl.arange(0, BLOCK_N); mask=offs<N
    x=tl.load(x_ptr+row*N+offs, mask=mask, other=0.0).to(tl.float32)
    gy=tl.load(gy_ptr+row*N+offs, mask=mask, other=0.0).to(tl.float32)
    w=tl.load(w_ptr+offs, mask=mask, other=0.0).to(tl.float32)
    mean=tl.sum(x,axis=0)/N
    xm=x-mean
    var=tl.sum(xm*xm,axis=0)/N
    rrms=tl.rsqrt(var+eps)
    gyw=gy*w
    c1=tl.sum(gyw,axis=0)/N
    c2=tl.sum(gyw*xm,axis=0)/N
    gx=rrms*(gyw - c1 - xm*c2*rrms*rrms)
    tl.store(gx_ptr+row*N+offs, gx.to(gx_ptr.dtype.element_ty), mask=mask)
    tl.atomic_add(gw_ptr+offs, (gy*xm*rrms), mask=mask)
    tl.atomic_add(gb_ptr+offs, gy, mask=mask)
def tri_lnbwd(x,w,b,gy,eps):
    M,N=x.shape; gx=torch.empty_like(x); gw=torch.zeros(N,device=x.device,dtype=torch.float32); gb=torch.zeros(N,device=x.device,dtype=torch.float32)
    layernorm_bwd[(M,)](x,w,b,gy,gx,gw,gb,eps,M,N,BLOCK_N=triton.next_power_of_2(N))
    return gx,gw,gb
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
        torch.manual_seed(0)
        x=torch.randn(M,N,device="cuda",dtype=torch.float32,requires_grad=True)
        w=torch.randn(N,device="cuda",dtype=torch.float32,requires_grad=True)
        bb=torch.zeros(N,device="cuda",dtype=torch.float32,requires_grad=True)
        eps=1e-5
        y=torch.nn.functional.layer_norm(x,(N,),w,bb,eps); gy=torch.randn_like(y)
        y.backward(gy); rxg=x.grad.detach().clone(); rwg=w.grad.detach().clone()
        x.grad=None; w.grad=None
        gx,gw,gd=tri_lnbwd(x.detach(),w.detach(),bb.detach(),gy,eps)
        ex=(gx-rxg).abs().max().item(); ew=(gw-rwg).abs().max().item()
        tri=lambda: tri_lnbwd(x.detach(),w.detach(),bb.detach(),gy,eps)
        tor=lambda: (lambda xx,ww:(torch.nn.functional.layer_norm(xx,(N,),ww,bb.detach(),eps).backward(gy), xx.grad)[1])(x.detach().clone().requires_grad_(True), w.detach().clone().requires_grad_(True))
        tm,_=time_fn(tri); tm_t,_=time_fn(tor)
        rows.append(dict(M=M,N=N,err_x=round(ex,5),err_w=round(ew,5),triton_ms=round(tm,4),torch_ms=round(tm_t,4),torch_over_triton=round(tm_t/tm,3)))
        print(f"M={M} N={N} err_x={ex:.3e} err_w={ew:.3e} triton={tm:.4f}ms torch={tm_t:.4f}ms torch/triton={tm_t/tm:.2f}x")
    print("RESULT_JSON", json.dumps(rows))
if __name__=="__main__": main()
