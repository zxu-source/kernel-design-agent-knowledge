#!/usr/bin/env python3
"""H200 Flash-Attention-2 CAUSAL forward (Triton) validation.
FA-2 forward with lower-triangular causal mask vs torch SDPA(is_causal=True).
PURPOSE = speedup (fused causal attention). Extends triton6660_flash_attention
with a causal row/col mask in the softmax. H200, Triton 3.6.
"""
import json, statistics
import torch, triton, triton.language as tl

@triton.jit
def attn_fwd_causal(Q, K, V, O, sm_scale, stride_bh, stride_row,
                    M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr):
    pid_m=tl.program_id(0); pid_bh=tl.program_id(1)
    offs_m=pid_m*BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d=tl.arange(0, HEAD_DIM)
    Qb=tl.make_block_ptr(base=Q+pid_bh*stride_bh, shape=(M,HEAD_DIM), strides=(stride_row,1),
                         offsets=(pid_m*BLOCK_M,0), block_shape=(BLOCK_M,HEAD_DIM), order=(1,0))
    q=tl.load(Qb, boundary_check=(0,))
    m_i=tl.full([BLOCK_M], -float('inf'), tl.float32)
    l_i=tl.full([BLOCK_M], 1.0, tl.float32)
    acc=tl.zeros([BLOCK_M,HEAD_DIM], tl.float32)
    # causal: only need N tiles up to the diagonal of this M-block
    n_end=tl.minimum( (pid_m+1)*BLOCK_M, N )
    for start_n in range(0, n_end, BLOCK_N):
        Kb=tl.make_block_ptr(base=K+pid_bh*stride_bh, shape=(HEAD_DIM,N), strides=(1,stride_row),
                             offsets=(0,start_n), block_shape=(HEAD_DIM,BLOCK_N), order=(0,1))
        Vb=tl.make_block_ptr(base=V+pid_bh*stride_bh, shape=(N,HEAD_DIM), strides=(stride_row,1),
                             offsets=(start_n,0), block_shape=(BLOCK_N,HEAD_DIM), order=(1,0))
        k=tl.load(Kb, boundary_check=(1,))
        qk=tl.dot(q,k)*sm_scale
        # causal mask: key col (start_n+j) must be <= query row (pid_m*BLOCK_M+i)
        n_offs=start_n + tl.arange(0,BLOCK_N)
        qk=tl.where(n_offs[None,:] <= offs_m[:,None], qk, -float('inf'))
        m_ij=tl.maximum(m_i, tl.max(qk,1))
        p=tl.exp(qk - m_ij[:,None])
        alpha=tl.exp(m_i - m_ij)
        l_i=l_i*alpha + tl.sum(p,1)
        acc=acc*alpha[:,None]
        v=tl.load(Vb, boundary_check=(0,))
        acc += tl.dot(p.to(v.dtype), v)
        m_i=m_ij
    acc=acc/l_i[:,None]
    Ob=tl.make_block_ptr(base=O+pid_bh*stride_bh, shape=(M,HEAD_DIM), strides=(stride_row,1),
                         offsets=(pid_m*BLOCK_M,0), block_shape=(BLOCK_M,HEAD_DIM), order=(1,0))
    tl.store(Ob, acc.to(O.dtype.element_ty), boundary_check=(0,))

def triton_fa2_causal(q,k,v,sm_scale,BLOCK_M=128,BLOCK_N=64):
    B,H,M,D=q.shape; N=k.shape[2]
    q2=q.reshape(B*H,M,D).contiguous(); k2=k.reshape(B*H,N,D).contiguous(); v2=v.reshape(B*H,N,D).contiguous()
    o=torch.empty((B*H,M,D),device=q.device,dtype=q.dtype)
    attn_fwd_causal[(triton.cdiv(M,BLOCK_M),B*H)](q2,k2,v2,o,sm_scale,q2.stride(0),q2.stride(1),M,N,BLOCK_M=BLOCK_M,BLOCK_N=BLOCK_N,HEAD_DIM=D)
    return o.reshape(B,H,M,D)
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
    for B,H,M,D in [(1,8,8192,64),(1,8,8192,128),(4,8,4096,64),(1,4,16384,64),(2,16,2048,128)]:
        torch.manual_seed(0)
        q=torch.randn(B,H,M,D,device="cuda",dtype=torch.float16)*0.2
        k=torch.randn(B,H,M,D,device="cuda",dtype=torch.float16)*0.2
        v=torch.randn(B,H,M,D,device="cuda",dtype=torch.float16)*0.2
        sm_scale=1.0/(D**0.5)
        ref=torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=True,scale=sm_scale)
        out=triton_fa2_causal(q,k,v,sm_scale)
        err=(out.float()-ref.float()).abs().max().item()
        BM=128 if D==64 else 64
        tri=lambda: triton_fa2_causal(q,k,v,sm_scale,BLOCK_M=BM)
        sd=lambda: torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=True,scale=sm_scale)
        tm,_=time_fn(tri); sdms,_=time_fn(sd)
        rows.append(dict(B=B,H=H,M=M,D=D,err=round(err,4),triton_ms=round(tm,4),sdpa_ms=round(sdms,4),sdpa_over_triton=round(sdms/tm,3)))
        print(f"B={B} H={H} M={M} D={D} err={err:.3e} triton={tm:.4f}ms sdpa={sdms:.4f}ms sdpa/triton={sdms/tm:.2f}x")
    print("RESULT_JSON", json.dumps(rows))
if __name__=="__main__": main()
