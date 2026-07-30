#!/usr/bin/env python3
"""H200 Fused AdamW Optimizer Step (Triton) vs torch.
Fused: m=β1*m+(1-β1)g; v=β2*v+(1-β2)g²; p -= lr*(m_hat/(sqrt(v_hat)+eps)+wd*p).
All elementwise in one kernel (3 reads m,v,g + param, 3 writes m,v,p).
vs torch: ~8 separate elementwise ops + intermediates.
PURPOSE = speedup (fusion). H200, Triton 3.6.
"""
import json, statistics
import torch, triton, triton.language as tl
@triton.jit
def adam_w(p_ptr, g_ptr, m_ptr, v_ptr, lr, beta1, beta2, eps, wd, bias_c1, bias_c2,
           total, BLOCK: tl.constexpr):
    pid=tl.program_id(0); offs=pid*BLOCK+tl.arange(0,BLOCK); mask=offs<total
    p=tl.load(p_ptr+offs, mask=mask, other=0.0).to(tl.float32)
    g=tl.load(g_ptr+offs, mask=mask, other=0.0).to(tl.float32)
    m=tl.load(m_ptr+offs, mask=mask, other=0.0).to(tl.float32)
    v=tl.load(v_ptr+offs, mask=mask, other=0.0).to(tl.float32)
    m = beta1*m + (1.0-beta1)*g
    v = beta2*v + (1.0-beta2)*g*g
    m_hat = m * bias_c1            # 1/(1-β1^t)
    v_hat = v * bias_c2            # 1/(1-β2^t)
    update = m_hat / (tl.sqrt(v_hat) + eps) + wd*p
    p_new = p - lr*update
    tl.store(m_ptr+offs, m.to(m_ptr.dtype.element_ty), mask=mask)
    tl.store(v_ptr+offs, v.to(v_ptr.dtype.element_ty), mask=mask)
    tl.store(p_ptr+offs, p_new.to(p_ptr.dtype.element_ty), mask=mask)
def tri_adam(p,g,m,v,lr=1e-3,b1=0.9,b2=0.999,eps=1e-8,wd=0.01,t=1,BLOCK=4096):
    bc1=1.0/(1.0-b1**t); bc2=1.0/(1.0-b2**t)
    total=p.numel()
    adam_w[(triton.cdiv(total,BLOCK),)](p,g,m,v,lr,b1,b2,eps,wd,bc1,bc2,total,BLOCK=BLOCK)
def torch_adam(p,g,m,v,lr=1e-3,b1=0.9,b2=0.999,eps=1e-8,wd=0.01,t=1):
    bc1=1.0/(1.0-b1**t); bc2=1.0/(1.0-b2**t)
    m2=b1*m+(1-b1)*g; v2=b2*v+(1-b2)*g*g
    m_hat=m2*bc1; v_hat=v2*bc2
    p2=p-lr*(m_hat/(v_hat.sqrt()+eps)+wd*p)
    return p2,m2,v2
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
        for dt in (torch.float32, torch.bfloat16):
            torch.manual_seed(0)
            p=torch.randn(N,device="cuda",dtype=dt)*0.1
            g=torch.randn(N,device="cuda",dtype=dt)*0.01
            m=torch.zeros(N,device="cuda",dtype=dt); v=torch.zeros(N,device="cuda",dtype=dt)
            # torch ref
            p_ref,m_ref,v_ref=torch_adam(p.clone(),g.clone(),m.clone(),v.clone())
            # triton
            pc,gc,mc,vc=p.clone(),g.clone(),m.clone(),v.clone()
            tri_adam(pc,gc,mc,vc)
            err_p=(pc.float()-p_ref.float()).abs().max().item()
            tri=lambda: tri_adam(p.clone(),g.clone(),m.clone(),v.clone())
            tor=lambda: torch_adam(p.clone(),g.clone(),m.clone(),v.clone())
            tm,_=time_fn(tri); tm_t,_=time_fn(tor)
            rows.append(dict(N=N,dtype=str(dt).split(".")[-1],err_p=round(err_p,5),
                             triton_ms=round(tm,4),torch_ms=round(tm_t,4),torch_over_triton=round(tm_t/tm,3)))
            print(f"N={N:8d} {str(dt).split('.')[-1]:8s} err_p={err_p:.3e} triton={tm:.4f}ms torch={tm_t:.4f}ms torch/triton={tm_t/tm:.2f}x")
    print("RESULT_JSON", json.dumps(rows))
if __name__=="__main__": main()
