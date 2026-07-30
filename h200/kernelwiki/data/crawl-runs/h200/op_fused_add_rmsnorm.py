#!/usr/bin/env python3
"""H200 Fused Add+RMSNorm (Triton) validation.
LLM residual-stream kernel: residual_out = residual + x; out = rmsnorm(residual_out).
Fusing add+norm in one kernel reads residual & x once, writes residual_out & out once
(vs torch: add->tmp, rmsnorm(tmp)->out, copy tmp->residual_out = 3 reads/3 writes).
PURPOSE = speedup (fusion). vs torch reference on H200 (SM90), Triton 3.6.
"""
import json, statistics
import torch, triton, triton.language as tl

@triton.jit
def fused_add_rmsnorm(res_ptr, x_ptr, o_ptr, ro_ptr, w_ptr, eps, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N); mask = offs < N
    res = tl.load(res_ptr + row*N + offs, mask=mask, other=0.0).to(tl.float32)
    xx = tl.load(x_ptr + row*N + offs, mask=mask, other=0.0).to(tl.float32)
    s = res + xx                                   # summed residual
    var = tl.sum(s*s, axis=0) / N
    rrms = tl.rsqrt(var + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(ro_ptr + row*N + offs, s.to(ro_ptr.dtype.element_ty), mask=mask)              # residual_out
    tl.store(o_ptr  + row*N + offs, (s*rrms*w).to(o_ptr.dtype.element_ty), mask=mask)       # out

def triton_f(res, x, w, eps):
    M,N = res.shape
    o=torch.empty_like(x); ro=torch.empty_like(res)
    fused_add_rmsnorm[(M,)](res, x, o, ro, w, eps, N, BLOCK_N=triton.next_power_of_2(N))
    return o, ro

def torch_ref(res, x, w, eps):
    s = res + x
    o = torch.nn.functional.rms_norm(s, (s.shape[-1],), weight=w, eps=eps)
    return o, s

def time_fn(fn, trials=30):
    for _ in range(8): fn(); torch.cuda.synchronize()
    ts=[]
    for _ in range(trials):
        s,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    return min(ts), statistics.median(ts)

def main():
    print(f"triton={triton.__version__} dev={torch.cuda.get_device_name(0)} SMs={torch.cuda.get_device_properties(0).multi_processor_count}")
    eps=1e-6
    shapes=[(4096,4096),(8192,4096),(4096,8192),(8192,8192),(8192,11008),(8192,14336)]
    rows=[]
    for M,N in shapes:
        dt=torch.bfloat16
        torch.manual_seed(0)
        res=torch.randn(M,N,device="cuda",dtype=dt)*0.3; x=torch.randn(M,N,device="cuda",dtype=dt)*0.3
        w=torch.randn(N,device="cuda",dtype=dt)*0.1+1.0
        o,ro=triton_f(res,x,w,eps)
        ot,rot=torch_ref(res,x,w,eps)
        err=(o.float()-ot.float()).abs().max().item(); errr=(ro.float()-rot.float()).abs().max().item()
        tri=lambda: triton_f(res,x,w,eps)
        tor=lambda: torch_ref(res,x,w,eps)
        tm,_=time_fn(tri); tm_t,_=time_fn(tor)
        rows.append(dict(M=M,N=N,err=round(err,5),err_ro=round(errr,5),
                         triton_ms=round(tm,4),torch_ms=round(tm_t,4),torch_over_triton=round(tm_t/tm,3)))
        print(f"M={M:5d} N={N:5d} bf16 err_out={err:.3e} err_res={errr:.3e} triton={tm:.4f}ms torch={tm_t:.4f}ms torch/triton={tm_t/tm:.2f}x")
    print("RESULT_JSON", json.dumps(rows)); print("max_err=", max(r["err"] for r in rows))

if __name__=="__main__": main()
