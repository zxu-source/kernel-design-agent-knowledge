#!/usr/bin/env python3
"""H200 Rotary Position Embedding (RoPE) (Triton) vs torch.
Llama-style rotate-half RoPE: for position p, split head_dim into halves, rotate.
q_out[..., :D/2] = q[..., :D/2]*cos - q[..., D/2:]*sin
q_out[..., D/2:] = q[..., D/2:]*cos + q[..., :D/2]*sin
PURPOSE = speedup (fused RoPE). vs torch reference on H200, Triton 3.6.
"""
import json, statistics
import torch, triton, triton.language as tl
@triton.jit
def rope_fwd(q_ptr, cos_ptr, sin_ptr, o_ptr, NH, S, D, BLOCK_D: tl.constexpr):
    # grid: (B*H*S,); each program = one (b,h,s) row
    row = tl.program_id(0)
    d = tl.arange(0, BLOCK_D); mask = d < D
    half = D // 2
    q = tl.load(q_ptr + row*D + d, mask=mask, other=0.0).to(tl.float32)
    s = row % S                           # sequence position
    c = tl.load(cos_ptr + s*half + (d % half), mask=mask, other=0.0).to(tl.float32)
    sn = tl.load(sin_ptr + s*half + (d % half), mask=mask, other=0.0).to(tl.float32)
    # partner index (rotate-half): always in [0,D)
    lo = d < half
    partner = tl.where(lo, d + half, d - half)
    qp = tl.load(q_ptr + row*D + partner, mask=mask, other=0.0).to(tl.float32)
    out = tl.where(lo, q*c - qp*sn, q*c + qp*sn)
    tl.store(o_ptr + row*D + d, out.to(o_ptr.dtype.element_ty), mask=mask)
def tri_rope(q, cos, sin):
    B,H,S,D = q.shape
    q2=q.reshape(B*H*S, D).contiguous(); o=torch.empty_like(q2)
    rope_fwd[(B*H*S,)](q2, cos, sin, o, H, S, D, BLOCK_D=triton.next_power_of_2(D))
    return o.reshape(B,H,S,D)
def torch_rope(q, cos, sin):
    # cos,sin: [S, D/2]
    d=q.shape[-1]//2
    q1=q[...,:d]; q2=q[...,d:]
    c=cos[None,None,:,:].expand_as(q1)  # but cos indexed by position
    # cos/sin shape [S,d]; q [B,H,S,d]
    c=cos.unsqueeze(0).unsqueeze(0); s=sin.unsqueeze(0).unsqueeze(0)
    o1=q1*c - q2*s; o2=q2*c + q1*s
    return torch.cat([o1,o2],dim=-1)
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
    for B,H,S,D in [(1,32,4096,128),(2,32,2048,128),(1,32,8192,128),(1,8,4096,64),(1,32,4096,256)]:
        torch.manual_seed(0); q=torch.randn(B,H,S,D,device="cuda",dtype=torch.float16)*0.3
        half=D//2
        inv_freq=1.0/(10000.0**(torch.arange(0,half,device="cuda",dtype=torch.float32)/half))
        pos=torch.arange(S,device="cuda",dtype=torch.float32)
        cos=(pos[:,None]*inv_freq[None,:]).cos().to(torch.float32)
        sin=(pos[:,None]*inv_freq[None,:]).sin().to(torch.float32)
        ref=torch_rope(q, cos, sin); out=tri_rope(q, cos, sin)
        err=(out.float()-ref.float()).abs().max().item()
        tri=lambda: tri_rope(q,cos,sin); tor=lambda: torch_rope(q,cos,sin)
        tm,_=time_fn(tri); tm_t,_=time_fn(tor)
        rows.append(dict(B=B,H=H,S=S,D=D,err=round(err,4),triton_ms=round(tm,4),torch_ms=round(tm_t,4),torch_over_triton=round(tm_t/tm,3)))
        print(f"B={B} H={H} S={S} D={D} err={err:.3e} triton={tm:.4f}ms torch={tm_t:.4f}ms torch/triton={tm_t/tm:.2f}x")
    print("RESULT_JSON", json.dumps(rows))
if __name__=="__main__": main()
