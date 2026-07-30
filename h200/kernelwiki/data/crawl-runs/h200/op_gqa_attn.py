#!/usr/bin/env python3
"""H200 Grouped-Query Attention (GQA) Flash-Forward (Triton) vs torch.
GQA: Hq query heads share Hkv = Hq//n_rep KV-head groups. FA kernel maps each Q
head to its KV group (head_kv = head_q // n_rep), avoiding expanding K/V.
PURPOSE = speedup (no KV replication -> less memory traffic). vs torch SDPA with
repeat_interleave'd K,V on H200, Triton 3.6. fp16.
"""
import json, statistics
import torch, triton, triton.language as tl
@triton.jit
def gqa_attn_fwd(Q, K, V, O, sm_scale, B, Hq, Hkv, M, N,
                 sQB, sQH, sQM, sKB, sKH, sKM,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr, NREP: tl.constexpr):
    pid_m=tl.program_id(0); pid_bh=tl.program_id(1)
    b=pid_bh // Hq; hq=pid_bh % Hq
    hkv=hq // NREP                          # GQA group mapping
    offs_m=pid_m*BLOCK_M+tl.arange(0,BLOCK_M)
    offs_d=tl.arange(0,HEAD_DIM)
    Qb=tl.make_block_ptr(base=Q+b*sQB+hq*sQH, shape=(M,HEAD_DIM), strides=(sQM,1),
                         offsets=(pid_m*BLOCK_M,0), block_shape=(BLOCK_M,HEAD_DIM), order=(1,0))
    q=tl.load(Qb, boundary_check=(0,))
    m_i=tl.full([BLOCK_M],-float('inf'),tl.float32); l_i=tl.full([BLOCK_M],1.0,tl.float32)
    acc=tl.zeros([BLOCK_M,HEAD_DIM],tl.float32)
    for start_n in range(0, N, BLOCK_N):
        Kb=tl.make_block_ptr(base=K+b*sKB+hkv*sKH, shape=(HEAD_DIM,N), strides=(1,sKM),
                             offsets=(0,start_n), block_shape=(HEAD_DIM,BLOCK_N), order=(0,1))
        Vb=tl.make_block_ptr(base=V+b*sKB+hkv*sKH, shape=(N,HEAD_DIM), strides=(sKM,1),
                             offsets=(start_n,0), block_shape=(BLOCK_N,HEAD_DIM), order=(1,0))
        k=tl.load(Kb, boundary_check=(1,))
        qk=tl.dot(q,k)*sm_scale
        m_ij=tl.maximum(m_i, tl.max(qk,1)); p=tl.exp(qk-m_ij[:,None])
        alpha=tl.exp(m_i-m_ij); l_i=l_i*alpha+tl.sum(p,1); acc=acc*alpha[:,None]
        v=tl.load(Vb, boundary_check=(0,)); acc+=tl.dot(p.to(v.dtype),v)
        m_i=m_ij
    acc=acc/l_i[:,None]
    Ob=tl.make_block_ptr(base=O+b*sQB+hq*sQH, shape=(M,HEAD_DIM), strides=(sQM,1),
                         offsets=(pid_m*BLOCK_M,0), block_shape=(BLOCK_M,HEAD_DIM), order=(1,0))
    tl.store(Ob, acc.to(O.dtype.element_ty), boundary_check=(0,))
def triton_gqa(q,k,v,sm_scale,BLOCK_M=128,BLOCK_N=64):
    B,Hq,M,D=q.shape; Hkv=k.shape[1]; N=k.shape[2]; nrep=Hq//Hkv
    o=torch.empty_like(q)
    gqa_attn_fwd[(triton.cdiv(M,BLOCK_M), B*Hq)](q,k,v,o,sm_scale,B,Hq,Hkv,M,N,
        q.stride(0),q.stride(1),q.stride(2), k.stride(0),k.stride(1),k.stride(2),
        BLOCK_M=BLOCK_M,BLOCK_N=BLOCK_N,HEAD_DIM=D,NREP=nrep)
    return o
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
    # (B, Hq, Hkv, M, D) — LLM-typical GQA ratios
    for B,Hq,Hkv,M,D in [(1,32,8,8192,128),(2,32,8,4096,128),(1,32,8,16384,128),(1,16,8,8192,64),(1,8,1,4096,128)]:  # last = MQA
        torch.manual_seed(0)
        q=torch.randn(B,Hq,M,D,device="cuda",dtype=torch.float16)*0.2
        k=torch.randn(B,Hkv,M,D,device="cuda",dtype=torch.float16)*0.2
        v=torch.randn(B,Hkv,M,D,device="cuda",dtype=torch.float16)*0.2
        sm_scale=1.0/(D**0.5)
        ke=k.repeat_interleave(Hq//Hkv,dim=1); ve=v.repeat_interleave(Hq//Hkv,dim=1)
        ref=torch.nn.functional.scaled_dot_product_attention(q,ke,ve,is_causal=False,scale=sm_scale)
        out=triton_gqa(q,k,v,sm_scale)
        err=(out.float()-ref.float()).abs().max().item()
        tri=lambda: triton_gqa(q,k,v,sm_scale)
        sd=lambda: torch.nn.functional.scaled_dot_product_attention(q,ke,ve,is_causal=False,scale=sm_scale)
        tm,_=time_fn(tri); sdms,_=time_fn(sd)
        rows.append(dict(B=B,Hq=Hq,Hkv=Hkv,M=M,D=D,err=round(err,4),triton_ms=round(tm,4),sdpa_ms=round(sdms,4),sdpa_over_triton=round(sdms/tm,3)))
        tag="MQA" if Hkv==1 else "GQA"
        print(f"{tag} B={B} Hq={Hq} Hkv={Hkv} M={M} D={D} err={err:.3e} triton={tm:.4f}ms sdpa={sdms:.4f}ms sdpa/triton={sdms/tm:.2f}x")
    print("RESULT_JSON", json.dumps(rows))
if __name__=="__main__": main()
