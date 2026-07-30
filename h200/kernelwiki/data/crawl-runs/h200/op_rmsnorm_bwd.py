#!/usr/bin/env python3
"""H200 RMSNorm Backward (Triton) vs torch.autograd.
grad_x = rrms*w*(grad_y - x*rrms^2 * mean(grad_y*w*x)); grad_w = sum_rows(grad_y*x*rrms).
One fused pass per row (grad_x) + atomic-add grad_w. PURPOSE = speedup.
vs torch autograd on H200, Triton 3.6.
"""
import json, statistics
import torch, triton, triton.language as tl
@triton.jit
def rmsnorm_bwd(x_ptr, w_ptr, gy_ptr, gx_ptr, gw_ptr, eps, M, N, BLOCK_N: tl.constexpr):
    row=tl.program_id(0)
    offs=tl.arange(0, BLOCK_N); mask=offs<N
    x=tl.load(x_ptr+row*N+offs, mask=mask, other=0.0).to(tl.float32)
    gy=tl.load(gy_ptr+row*N+offs, mask=mask, other=0.0).to(tl.float32)
    w=tl.load(w_ptr+offs, mask=mask, other=0.0).to(tl.float32)
    var=tl.sum(x*x, axis=0)/N
    rrms=tl.rsqrt(var+eps)
    g2w=gy*w
    c=tl.sum(g2w*x, axis=0)/N
    gx=rrms*(w*gy - x*rrms*rrms*c)   # rrms*w*gy - x*rrms^3*c (variance term has NO w)
    tl.store(gx_ptr+row*N+offs, gx.to(gx_ptr.dtype.element_ty), mask=mask)
    tl.atomic_add(gw_ptr+offs, (gy*x*rrms).to(tl.float32), mask=mask)
def tri_rmsbwd(x,w,gy,eps):
    M,N=x.shape; gx=torch.empty_like(x); gw=torch.zeros_like(w, dtype=torch.float32)
    rmsnorm_bwd[(M,)](x,w,gy,gx,gw,eps,M,N,BLOCK_N=triton.next_power_of_2(N))
    return gx, gw
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
        eps=1e-6
        y=torch.nn.functional.rms_norm(x,(N,),w,eps); gy=torch.randn_like(y)
        y.backward(gy); rxg=x.grad.detach().clone(); rwg=w.grad.detach().clone()
        x.grad=None; w.grad=None
        gx,gw=tri_rmsbwd(x.detach(),w.detach(),gy,eps)
        ex=(gx-rxg).abs().max().item(); ew=(gw-rwg).abs().max().item()
        tri=lambda: tri_rmsbwd(x.detach(),w.detach(),gy,eps)
        tor=lambda: (lambda xx,ww:(torch.nn.functional.rms_norm(xx,(N,),ww,eps).backward(gy), (xx.grad,ww.grad))[0])(x.detach().clone().requires_grad_(True), w.detach().clone().requires_grad_(True))
        tm,_=time_fn(tri); tm_t,_=time_fn(tor)
        rows.append(dict(M=M,N=N,err_x=round(ex,5),err_w=round(ew,5),triton_ms=round(tm,4),torch_ms=round(tm_t,4),torch_over_triton=round(tm_t/tm,3)))
        print(f"M={M} N={N} err_x={ex:.3e} err_w={ew:.3e} triton={tm:.4f}ms torch={tm_t:.4f}ms torch/triton={tm_t/tm:.2f}x")
    print("RESULT_JSON", json.dumps(rows))
if __name__=="__main__": main()
