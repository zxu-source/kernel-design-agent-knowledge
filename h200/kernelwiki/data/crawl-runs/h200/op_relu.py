#!/usr/bin/env python3
"""H200 ReLU Forward + Backward (Triton) vs torch.
ReLU fwd: y=max(x,0). ReLU bwd: grad_x=grad_y*(x>0). Fused elementwise.
PURPOSE = speedup (fused). vs torch.relu / autograd on H200, Triton 3.6.
"""
import json, statistics
import torch, triton, triton.language as tl
@triton.jit
def relu_fwd(x_ptr, o_ptr, total, BLOCK: tl.constexpr):
    pid=tl.program_id(0); offs=pid*BLOCK+tl.arange(0,BLOCK); mask=offs<total
    x=tl.load(x_ptr+offs, mask=mask, other=0.0)
    tl.store(o_ptr+offs, tl.maximum(x, 0.0), mask=mask)
@triton.jit
def relu_bwd(x_ptr, gy_ptr, gx_ptr, total, BLOCK: tl.constexpr):
    pid=tl.program_id(0); offs=pid*BLOCK+tl.arange(0,BLOCK); mask=offs<total
    x=tl.load(x_ptr+offs, mask=mask, other=0.0)
    gy=tl.load(gy_ptr+offs, mask=mask, other=0.0)
    tl.store(gx_ptr+offs, gy * (x > 0.0).to(gy.dtype), mask=mask)
def tri_relu_fwd(x, BLOCK=4096):
    o=torch.empty_like(x); relu_fwd[(triton.cdiv(x.numel(),BLOCK),)](x,o,x.numel(),BLOCK=BLOCK); return o
def tri_relu_bwd(x, gy, BLOCK=4096):
    o=torch.empty_like(x); relu_bwd[(triton.cdiv(x.numel(),BLOCK),)](x,gy,o,x.numel(),BLOCK=BLOCK); return o
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
    for M,N in [(4096,4096),(8192,8192),(8192,14336),(16384,14336)]:
        for dt in (torch.float32, torch.bfloat16):
            torch.manual_seed(0); x=(torch.randn(M,N,device="cuda",dtype=dt)*0.5).requires_grad_(True)
            # fwd
            ref=torch.relu(x.detach()); out=tri_relu_fwd(x.detach())
            err=(out-ref).abs().max().item()
            # bwd
            gy=torch.randn(M,N,device="cuda",dtype=dt)
            y=torch.relu(x); y.backward(gy); rxg=x.grad.detach().clone(); x.grad=None
            obwd=tri_relu_bwd(x.detach(),gy)
            errb=(obwd-rxg).abs().max().item()
            tri_f=lambda: tri_relu_fwd(x.detach())
            tor_f=lambda: torch.relu(x.detach())
            x2=x.detach().clone().requires_grad_(True); y2=torch.relu(x2)
            def tor_b():
                if x2.grad is not None: x2.grad=None
                y2.backward(gy, retain_graph=True)
            tf,_=time_fn(tri_f); tf_t,_=time_fn(tor_f)
            tb,_=time_fn(lambda: tri_relu_bwd(x.detach(),gy)); tb_t,_=time_fn(tor_b)
            rows.append(dict(M=M,N=N,dtype=str(dt).split(".")[-1],err_f=round(err,5),err_b=round(errb,5),
                             fwd_triton=round(tf,4),fwd_torch=round(tf_t,4),fwd_ratio=round(tf_t/tf,3),
                             bwd_triton=round(tb,4),bwd_torch=round(tb_t,4),bwd_ratio=round(tb_t/tb,3)))
            print(f"M={M} N={N} {str(dt).split('.')[-1]:8s} fwd: err={err:.2e} tri={tf:.4f}ms torch={tf_t:.4f}ms ratio={tf_t/tf:.2f}x | bwd: err={errb:.2e} tri={tb:.4f}ms torch={tb_t:.4f}ms ratio={tb_t/tb:.2f}x")
    print("RESULT_JSON", json.dumps(rows))
if __name__=="__main__": main()
