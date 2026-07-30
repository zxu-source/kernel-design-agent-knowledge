#!/usr/bin/env python3
"""H200 Softmax Backward (Triton) vs torch.autograd.
grad_x = (grad_y - sum(grad_y * y)) * y, where y = softmax output. One fused pass
per row. PURPOSE = speedup (fused backward). vs torch autograd on H200, Triton 3.6.
"""
import json, statistics
import torch, triton, triton.language as tl
@triton.jit
def softmax_bwd(y_ptr, gy_ptr, gx_ptr, N, BLOCK_N: tl.constexpr):
    row=tl.program_id(0)
    offs=tl.arange(0, BLOCK_N); mask=offs<N
    y=tl.load(y_ptr+row*N+offs, mask=mask, other=0.0).to(tl.float32)
    gy=tl.load(gy_ptr+row*N+offs, mask=mask, other=0.0).to(tl.float32)
    s=tl.sum(gy*y, axis=0)
    gx=(gy - s)*y
    tl.store(gx_ptr+row*N+offs, gx.to(gx_ptr.dtype.element_ty), mask=mask)
def tri_smbwd(y, gy):
    M,N=y.shape; gx=torch.empty_like(y)
    softmax_bwd[(M,)](y, gy, gx, N, BLOCK_N=triton.next_power_of_2(N)); return gx
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
    for M,N in [(4096,4096),(8192,8192),(4096,32000),(8192,32000)]:
        torch.manual_seed(0)
        logits=torch.randn(M,N,device="cuda",dtype=torch.float32,requires_grad=True)
        y=torch.softmax(logits,dim=-1)
        gy=torch.randn_like(y)
        # torch autograd backward
        ref=torch.empty_like(y)
        logits2=logits.detach().clone().requires_grad_(True)
        y2=torch.softmax(logits2,dim=-1); y2.backward(gy); ref=logits2.grad
        out=tri_smbwd(y.detach(), gy)
        err=(out-ref).abs().max().item(); rel=err/(ref.abs().max().item()+1e-9)
        tri=lambda: tri_smbwd(y.detach(),gy)
        tor=lambda: (lambda l: (torch.softmax(l,-1).backward(gy), l.grad)[1])(logits.detach().clone().requires_grad_(True))
        tm,_=time_fn(tri); tm_t,_=time_fn(tor)
        rows.append(dict(M=M,N=N,err=round(err,5),rel=round(rel,6),triton_ms=round(tm,4),torch_ms=round(tm_t,4),torch_over_triton=round(tm_t/tm,3)))
        print(f"M={M} N={N:6d} err={err:.3e} rel={rel:.2e} triton={tm:.4f}ms torch={tm_t:.4f}ms torch/triton={tm_t/tm:.2f}x")
    print("RESULT_JSON", json.dumps(rows))
if __name__=="__main__": main()
