#!/usr/bin/env python3
"""H200 Gradient Norm Clipping (Triton) vs torch.nn.utils.clip_grad_norm_.
Step 1: compute total grad norm = sqrt(sum(g^2)). Step 2: scale g *= min(1, max_norm/(norm+1e-6)).
PURPOSE = robustness (prevents gradient explosion) + speedup (fused).
H200, Triton 3.6.
"""
import json, statistics
import torch, triton, triton.language as tl
def cdiv(a,b): return (a+b-1)//b
@triton.jit
def sum_sq_kernel(g_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid=tl.program_id(0); offs=pid*BLOCK+tl.arange(0,BLOCK); mask=offs<N
    g=tl.load(g_ptr+offs, mask=mask, other=0.0).to(tl.float32)
    s=tl.sum(g*g, axis=0)
    tl.atomic_add(out_ptr, s)
@triton.jit
def scale_kernel(g_ptr, scale, N, BLOCK: tl.constexpr):
    pid=tl.program_id(0); offs=pid*BLOCK+tl.arange(0,BLOCK); mask=offs<N
    g=tl.load(g_ptr+offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(g_ptr+offs, (g*scale).to(g_ptr.dtype.element_ty), mask=mask)
def tri_clip(g, max_norm=1.0, BLOCK=4096):
    N=g.numel()
    ss=torch.zeros((1,),device=g.device,dtype=torch.float32)
    sum_sq_kernel[(cdiv(N,BLOCK),)](g, ss, N, BLOCK=BLOCK)
    norm=ss.item()**0.5
    scale=min(1.0, max_norm/(norm+1e-6))
    if scale < 1.0:
        scale_kernel[(cdiv(N,BLOCK),)](g, scale, N, BLOCK=BLOCK)
    return norm, scale
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
    for N in [1<<20, 1<<22, 1<<24, 1<<25]:
        dt=torch.float32
        torch.manual_seed(0)
        g_orig=torch.randn(N,device="cuda",dtype=dt)*10.0  # large grads -> will clip
        # torch ref (manual, since clip_grad_norm_ needs Parameters)
        g_ref=g_orig.clone()
        ref_norm=g_ref.float().norm(2).item()
        ref_scale=min(1.0, 1.0/(ref_norm+1e-6))
        if ref_scale < 1.0: g_ref *= ref_scale
        # triton
        g_tri=g_orig.clone()
        tri_norm, tri_scale = tri_clip(g_tri, max_norm=1.0)
        err_norm=abs(tri_norm-ref_norm)/max(ref_norm,1e-9)
        err_g=(g_tri-g_ref).abs().max().item()
        tri=lambda: tri_clip(g_orig.clone(), max_norm=1.0)
        def tor():
            gg=g_orig.clone(); torch.nn.utils.clip_grad_norm_([gg], max_norm=1.0); return gg
        tm,_=time_fn(tri); tm_t,_=time_fn(tor)
        rows.append(dict(N=N,norm_err=round(err_norm,6),grad_err=round(err_g,5),
                         tri_norm=round(tri_norm,2),tri_scale=round(tri_scale,5),
                         triton_ms=round(tm,4),torch_ms=round(tm_t,4),torch_over_triton=round(tm_t/tm,3)))
        print(f"N={N:8d} norm_err={err_norm:.2e} grad_err={err_g:.3e} norm={tri_norm:.1f} scale={tri_scale:.5f} triton={tm:.4f}ms torch={tm_t:.4f}ms torch/triton={tm_t/tm:.2f}x")
    print("RESULT_JSON", json.dumps(rows))
if __name__=="__main__": main()
