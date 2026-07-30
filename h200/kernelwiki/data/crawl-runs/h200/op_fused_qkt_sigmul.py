#!/usr/bin/env python3
"""H200 Fused QK^T*scale+mask + Fused Sigmoid*Mul (Triton) vs torch.
QK^T+scale+mask: attention score = (Q@K^T)*scale + mask (GEMM epilogue fusion).
Sigmoid*Mul: out = x * sigmoid(gate) (GRU/LSTM gate mechanism).
PURPOSE = speedup (fusion). H200, Triton 3.6.
"""
import json, statistics
import torch, triton, triton.language as tl
def cdiv(a,b): return (a+b-1)//b

# ---- Fused QK^T * scale + mask (attention score pre-softmax) ----
@triton.jit
def qkt_scale_mask(q_ptr, k_ptr, m_ptr, o_ptr, M, N, D, scale,
                   sqm, sqd, skn, skd, som, son,
                   BM:tl.constexpr, BN:tl.constexpr, BD:tl.constexpr):
    pm=tl.program_id(0); pn=tl.program_id(1)
    om=pm*BM+tl.arange(0,BM); on=pn*BN+tl.arange(0,BN)
    acc=tl.zeros((BM,BN),dtype=tl.float32)
    for d0 in range(0,D,BD):
        od=d0+tl.arange(0,BD)
        q=tl.load(q_ptr+om[:,None]*sqm+od[None,:]*sqd,mask=(om[:,None]<M)&(od[None,:]<D),other=0.0)
        k=tl.load(k_ptr+on[:,None]*skn+od[None,:]*skd,mask=(on[:,None]<N)&(od[None,:]<D),other=0.0)
        acc+=tl.dot(q, tl.trans(k))   # Q[M,D] @ K[N,D]^T -> [M,N]
    acc=acc*scale                      # epilogue: scale
    mask=tl.load(m_ptr+om[:,None]*som+on[None,:]*son,mask=(om[:,None]<M)&(on[None,:]<N),other=0.0).to(tl.float32)
    acc=acc+mask                       # epilogue: additive mask (0 or -inf)
    tl.store(o_ptr+om[:,None]*som+on[None,:]*son, acc, mask=(om[:,None]<M)&(on[None,:]<N))

# ---- Fused Sigmoid * Mul (gate) ----
@triton.jit
def fused_sigmoid_mul(x_ptr, g_ptr, o_ptr, total, BLOCK:tl.constexpr):
    pid=tl.program_id(0); offs=pid*BLOCK+tl.arange(0,BLOCK); mask=offs<total
    x=tl.load(x_ptr+offs,mask=mask,other=0.0).to(tl.float32)
    g=tl.load(g_ptr+offs,mask=mask,other=0.0).to(tl.float32)
    tl.store(o_ptr+offs, (x*(1.0/(1.0+tl.exp(-g)))).to(o_ptr.dtype.element_ty), mask=mask)

def time_fn(fn, trials=50):
    for _ in range(10): fn(); torch.cuda.synchronize()
    ts=[]
    for _ in range(trials):
        s,e=torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    return min(ts), statistics.median(ts)

def main():
    print(f"triton={triton.__version__} dev={torch.cuda.get_device_name(0)} SMs={torch.cuda.get_device_properties(0).multi_processor_count}")
    print("--- Fused QK^T*scale+mask ---")
    for M,N,D in [(512,512,64),(1024,1024,128),(2048,2048,128),(4096,4096,128)]:
        dt=torch.bfloat16; torch.manual_seed(0)
        q=torch.randn(M,D,device="cuda",dtype=dt)*0.2; k=torch.randn(N,D,device="cuda",dtype=dt)*0.2
        mask=torch.zeros(M,N,device="cuda",dtype=dt)
        # causal mask: lower triangle 0, upper -inf
        for i in range(min(M,N)): mask[i,i+1:]=float('-inf')
        scale=1.0/(D**0.5)
        ref=(q.float()@k.float().T)*scale + mask.float()
        o=torch.empty((M,N),device="cuda",dtype=torch.float32)
        qkt_scale_mask[(cdiv(M,128),cdiv(N,128))](q,k,mask,o,M,N,D,scale,
            q.stride(0),q.stride(1),k.stride(0),k.stride(1),o.stride(0),o.stride(1),
            BM=128,BN=128,BD=64,num_warps=8,num_stages=3)
        err=(o-ref).abs().max().item()
        # exclude -inf positions from err
        finite=(ref!=-float('inf'))
        err_finite=(o[finite]-ref[finite]).abs().max().item() if finite.any() else 0
        tri=lambda: qkt_scale_mask[(cdiv(M,128),cdiv(N,128))](q,k,mask,torch.empty_like(o),M,N,D,scale,
            q.stride(0),q.stride(1),k.stride(0),k.stride(1),o.stride(0),o.stride(1),BM=128,BN=128,BD=64,num_warps=8,num_stages=3)
        tor=lambda: (q.float()@k.float().T)*scale+mask.float()
        tm,_=time_fn(tri); tm_t,_=time_fn(tor)
        print(f"  M={M} N={N} D={D} err_finite={err_finite:.3e} triton={tm:.4f}ms torch={tm_t:.4f}ms torch/triton={tm_t/tm:.2f}x")
    print("--- Fused Sigmoid*Mul ---")
    for M,N in [(4096,4096),(8192,8192),(8192,14336)]:
        for dt in (torch.float32, torch.bfloat16):
            torch.manual_seed(0)
            x=torch.randn(M,N,device="cuda",dtype=dt)*0.3; g=torch.randn(M,N,device="cuda",dtype=dt)*0.3
            ref=x*torch.sigmoid(g)
            o=torch.empty_like(x); fused_sigmoid_mul[(triton.cdiv(M*N,4096),)](x,g,o,M*N,BLOCK=4096)
            err=(o.float()-ref.float()).abs().max().item()
            tri=lambda: fused_sigmoid_mul[(triton.cdiv(M*N,4096),)](x,g,torch.empty_like(x),M*N,BLOCK=4096)
            tor=lambda: x*torch.sigmoid(g)
            tm,_=time_fn(tri); tm_t,_=time_fn(tor)
            print(f"  M={M} N={N} {str(dt).split('.')[-1]:8s} err={err:.3e} triton={tm:.4f}ms torch={tm_t:.4f}ms torch/triton={tm_t/tm:.2f}x")
if __name__=="__main__": main()
