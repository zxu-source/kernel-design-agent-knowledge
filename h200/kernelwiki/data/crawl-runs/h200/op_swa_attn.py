#!/usr/bin/env python3
"""H200 Sliding-Window Attention (Triton) validation.
Attention with a sliding-window causal mask: query i attends to keys in
[max(0, i-window+1), i]. Common in long-context LLMs (Mistral/Qwen sliding
window). Extends FA-2 with a windowed mask. PURPOSE = speedup (fused windowed
attention; fewer N tiles for small window). vs torch SDPA reference (manual
windowed mask) on H200, Triton 3.6. fp16.
"""
import json, statistics
import torch, triton, triton.language as tl

@triton.jit
def swa_attn_fwd(Q,K,V,O,sm_scale,stride_bh,stride_row,M,N,WINDOW:tl.constexpr,
                 BLOCK_M:tl.constexpr,BLOCK_N:tl.constexpr,HEAD_DIM:tl.constexpr):
    pid_m=tl.program_id(0); pid_bh=tl.program_id(1)
    offs_m=pid_m*BLOCK_M+tl.arange(0,BLOCK_M)
    Qb=tl.make_block_ptr(base=Q+pid_bh*stride_bh,shape=(M,HEAD_DIM),strides=(stride_row,1),
                         offsets=(pid_m*BLOCK_M,0),block_shape=(BLOCK_M,HEAD_DIM),order=(1,0))
    q=tl.load(Qb,boundary_check=(0,))
    m_i=tl.full([BLOCK_M],-1e30,tl.float32); l_i=tl.full([BLOCK_M],1.0,tl.float32)
    acc=tl.zeros([BLOCK_M,HEAD_DIM],tl.float32)
    for start_n in range(0, M, BLOCK_N):       # full N range; mask applies causal+window
        Kb=tl.make_block_ptr(base=K+pid_bh*stride_bh,shape=(HEAD_DIM,N),strides=(1,stride_row),
                             offsets=(0,start_n),block_shape=(HEAD_DIM,BLOCK_N),order=(0,1))
        Vb=tl.make_block_ptr(base=V+pid_bh*stride_bh,shape=(N,HEAD_DIM),strides=(stride_row,1),
                             offsets=(start_n,0),block_shape=(BLOCK_N,HEAD_DIM),order=(1,0))
        k=tl.load(Kb,boundary_check=(1,))
        qk=tl.dot(q,k)*sm_scale
        n_offs=start_n+tl.arange(0,BLOCK_N)
        # causal AND within window: col <= row AND col > row-WINDOW
        mask=(n_offs[None,:] <= offs_m[:,None]) & (n_offs[None,:] > (offs_m[:,None] - WINDOW))
        qk=tl.where(mask, qk, -1e30)
        m_ij=tl.maximum(m_i,tl.max(qk,1)); p=tl.exp(qk-m_ij[:,None])
        alpha=tl.exp(m_i-m_ij); l_i=l_i*alpha+tl.sum(p,1); acc=acc*alpha[:,None]
        v=tl.load(Vb,boundary_check=(0,)); acc+=tl.dot(p.to(v.dtype),v)
        m_i=m_ij
    acc=acc/l_i[:,None]
    Ob=tl.make_block_ptr(base=O+pid_bh*stride_bh,shape=(M,HEAD_DIM),strides=(stride_row,1),
                         offsets=(pid_m*BLOCK_M,0),block_shape=(BLOCK_M,HEAD_DIM),order=(1,0))
    tl.store(Ob,acc.to(O.dtype.element_ty),boundary_check=(0,))

def triton_swa(q,k,v,sm_scale,window,BLOCK_M=128,BLOCK_N=64):
    B,H,M,D=q.shape; N=M
    q2=q.reshape(B*H,M,D).contiguous(); k2=k.reshape(B*H,N,D).contiguous(); v2=v.reshape(B*H,N,D).contiguous()
    o=torch.empty((B*H,M,D),device=q.device,dtype=q.dtype)
    swa_attn_fwd[(triton.cdiv(M,BLOCK_M),B*H)](q2,k2,v2,o,sm_scale,q2.stride(0),q2.stride(1),M,N,WINDOW=window,BLOCK_M=BLOCK_M,BLOCK_N=BLOCK_N,HEAD_DIM=D)
    return o.reshape(B,H,M,D)

def torch_swa_ref(q,k,v,sm_scale,window):
    B,H,M,D=q.shape
    attn=torch.einsum('bhid,bhjd->bhij', q.float(), k.float())*sm_scale   # [B,H,M,M]
    i=torch.arange(M,device=q.device)
    mask=(i[None,:] <= i[:,None]) & (i[None,:] > (i[:,None]-window))       # [M,M]
    attn=attn.masked_fill(~mask[None,None,:,:], float('-inf'))
    p=torch.softmax(attn,dim=-1)
    out=torch.einsum('bhij,bhjd->bhid', p, v.float()).to(q.dtype)
    return out

def time_fn(fn,trials=20):
    for _ in range(5): fn(); torch.cuda.synchronize()
    ts=[]
    for _ in range(trials):
        s,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    return min(ts), statistics.median(ts)

def main():
    print(f"triton={triton.__version__} dev={torch.cuda.get_device_name(0)} SMs={torch.cuda.get_device_properties(0).multi_processor_count}")
    rows=[]
    for B,H,M,D,W in [(1,8,8192,64,512),(1,8,8192,128,1024),(4,8,4096,64,512),(1,4,16384,64,1024),(2,16,2048,128,512)]:
        torch.manual_seed(0)
        q=torch.randn(B,H,M,D,device="cuda",dtype=torch.float16)*0.2
        k=torch.randn(B,H,M,D,device="cuda",dtype=torch.float16)*0.2
        v=torch.randn(B,H,M,D,device="cuda",dtype=torch.float16)*0.2
        sm_scale=1.0/(D**0.5)
        ref=torch_swa_ref(q,k,v,sm_scale,W); out=triton_swa(q,k,v,sm_scale,W)
        err=(out.float()-ref.float()).abs().max().item()
        tri=lambda: triton_swa(q,k,v,sm_scale,W)
        ref_fn=lambda: torch_swa_ref(q,k,v,sm_scale,W)
        tm,_=time_fn(tri); rm,_=time_fn(ref_fn)
        rows.append(dict(B=B,H=H,M=M,D=D,W=W,err=round(err,4),triton_ms=round(tm,4),ref_ms=round(rm,4),ref_over_triton=round(rm/tm,3)))
        print(f"B={B} H={H} M={M} D={D} W={W} err={err:.3e} triton={tm:.4f}ms naive_ref={rm:.4f}ms ref/triton={rm/tm:.2f}x")
    print("RESULT_JSON", json.dumps(rows))
if __name__=="__main__": main()
