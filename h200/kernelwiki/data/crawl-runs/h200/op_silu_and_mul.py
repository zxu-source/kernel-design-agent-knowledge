#!/usr/bin/env python3
"""H200 SiLU-and-Mul (Triton) validation.
LLM MLP gate/up: y = silu(gate) * up = (gate/(1+e^-gate)) * up. Fused elementwise
(vs torch: silu(gate) then * up = 2 elementwise launches + intermediate).
PURPOSE = speedup (fusion). vs torch on H200 (SM90), Triton 3.6.
"""
import json, statistics
import torch, triton, triton.language as tl

@triton.jit
def silu_and_mul(g_ptr, u_ptr, o_ptr, total, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid*BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    g = tl.load(g_ptr+offs, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(u_ptr+offs, mask=mask, other=0.0).to(tl.float32)
    silu = g / (1.0 + tl.exp(-g))
    tl.store(o_ptr+offs, (silu * u).to(o_ptr.dtype.element_ty), mask=mask)

def triton_sm(g, u):
    o = torch.empty_like(g)
    total = g.numel()
    BLOCK = 4096
    silu_and_mul[(triton.cdiv(total,BLOCK),)](g, u, o, total, BLOCK=BLOCK)
    return o

def torch_ref(g, u):
    return torch.nn.functional.silu(g) * u

def time_fn(fn, trials=50):
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
        for dt in (torch.bfloat16, torch.float16):
            torch.manual_seed(0)
            g=torch.randn(M,N,device="cuda",dtype=dt)*0.5; u=torch.randn(M,N,device="cuda",dtype=dt)*0.5
            ref=torch_ref(g,u); out=triton_sm(g,u)
            err=(out.float()-ref.float()).abs().max().item()
            tri=lambda: triton_sm(g,u); tor=lambda: torch_ref(g,u)
            tm,_=time_fn(tri); tm_t,_=time_fn(tor)
            rows.append(dict(M=M,N=N,dtype=str(dt).split(".")[-1],err=round(err,5),
                             triton_ms=round(tm,4),torch_ms=round(tm_t,4),torch_over_triton=round(tm_t/tm,3)))
            print(f"M={M:5d} N={N:5d} {str(dt).split('.')[-1]:8s} err={err:.3e} triton={tm:.4f}ms torch={tm_t:.4f}ms torch/triton={tm_t/tm:.2f}x")
    print("RESULT_JSON", json.dumps(rows)); print("max_err=", max(r["err"] for r in rows))

if __name__=="__main__": main()
