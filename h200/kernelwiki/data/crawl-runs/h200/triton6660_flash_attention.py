#!/usr/bin/env python3
"""H200 validation for Triton PR #6660 (software-pipelined attention).

PR #6660 improves Triton's compiler Pipeliner + warp-specialization for
attention. Those are compiler-internal passes with no user toggle, so this
harness characterizes the kernel CLASS they optimize: a standard Triton
flash-attention-2 forward kernel on H200 (SM90). Validates correctness against
torch SDPA and measures Triton FA-2 throughput (TFLOPS) vs torch SDPA
(FlashAttention-3 / cuDNN backend on cu130).

Hardware: NVIDIA H200, 132 SMs, cc 9.0. Triton 3.6.0, PyTorch 2.11.0+cu130.
From-scratch FA-2 forward (not copied from upstream); the PR's specific
pipeliner changes are not individually isolated here.
"""
import json, statistics
import torch, triton, triton.language as tl

@triton.jit
def attn_fwd(Q, K, V, O, sm_scale, stride_bh, stride_row,
             M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    Qb = tl.make_block_ptr(base=Q + pid_bh*stride_bh, shape=(M, HEAD_DIM), strides=(stride_row, 1),
                           offsets=(pid_m*BLOCK_M, 0), block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0))
    Kb = tl.make_block_ptr(base=K + pid_bh*stride_bh, shape=(HEAD_DIM, N), strides=(1, stride_row),
                           offsets=(0, 0), block_shape=(HEAD_DIM, BLOCK_N), order=(0, 1))
    Vb = tl.make_block_ptr(base=V + pid_bh*stride_bh, shape=(N, HEAD_DIM), strides=(stride_row, 1),
                           offsets=(0, 0), block_shape=(BLOCK_N, HEAD_DIM), order=(1, 0))
    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.full([BLOCK_M], 1.0, tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    q = tl.load(Qb, boundary_check=(0,))
    for start_n in range(0, N, BLOCK_N):
        k = tl.load(Kb, boundary_check=(1,))
        qk = tl.dot(q, k) * sm_scale
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_ij[:, None])
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v = tl.load(Vb, boundary_check=(0,))
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_ij
        Kb = tl.advance(Kb, (0, BLOCK_N))
        Vb = tl.advance(Vb, (BLOCK_N, 0))
    acc = acc / l_i[:, None]
    Ob = tl.make_block_ptr(base=O + pid_bh*stride_bh, shape=(M, HEAD_DIM), strides=(stride_row, 1),
                           offsets=(pid_m*BLOCK_M, 0), block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0))
    tl.store(Ob, acc.to(O.dtype.element_ty), boundary_check=(0,))

def triton_fa2(q, k, v, sm_scale, BLOCK_M=64, BLOCK_N=64):
    B,H,M,D = q.shape; N = k.shape[2]
    q2=q.reshape(B*H,M,D).contiguous(); k2=k.reshape(B*H,N,D).contiguous(); v2=v.reshape(B*H,N,D).contiguous()
    o=torch.empty((B*H,M,D),device=q.device,dtype=q.dtype)
    grid=(triton.cdiv(M,BLOCK_M), B*H)
    attn_fwd[grid](q2,k2,v2,o, sm_scale, q2.stride(0), q2.stride(1), M,N, BLOCK_M=BLOCK_M,BLOCK_N=BLOCK_N,HEAD_DIM=D)
    return o.reshape(B,H,M,D)

def time_fn(fn, trials=20):
    for _ in range(5): fn(); torch.cuda.synchronize()
    ts=[]
    for _ in range(trials):
        s,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    return min(ts), statistics.median(ts)

def attn_flops(B,H,M,N,D): return 4*B*H*M*N*D

def main():
    print(f"triton={triton.__version__} dev={torch.cuda.get_device_name(0)} SMs={torch.cuda.get_device_properties(0).multi_processor_count}")
    shapes=[(1,8,8192,8192,64),(1,8,8192,8192,128),(4,8,4096,4096,64),(1,4,16384,16384,64),(2,16,2048,2048,128)]
    rows=[]
    for B,H,M,N,D in shapes:
        torch.manual_seed(0)
        q=torch.randn(B,H,M,D,device="cuda",dtype=torch.float16)*0.2
        k=torch.randn(B,H,N,D,device="cuda",dtype=torch.float16)*0.2
        v=torch.randn(B,H,N,D,device="cuda",dtype=torch.float16)*0.2
        sm_scale=1.0/(D**0.5)
        ref=torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=False,scale=sm_scale)
        BM = 128 if D==64 else 64
        out=triton_fa2(q,k,v,sm_scale,BLOCK_M=BM,BLOCK_N=64)
        err=(out.float()-ref.float()).abs().max().item()
        tri=lambda: triton_fa2(q,k,v,sm_scale,BLOCK_M=BM,BLOCK_N=64)
        sd=lambda: torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=False,scale=sm_scale)
        tr_min,_=time_fn(tri); sd_min,_=time_fn(sd)
        fl=attn_flops(B,H,M,N,D)
        rows.append(dict(B=B,H=H,M=M,N=N,D=D,err=round(err,4),triton_ms=round(tr_min,4),
                         sdpa_ms=round(sd_min,4),ratio=round(sd_min/tr_min,3),
                         triton_tflops=round(fl/(tr_min*1e-3)/1e12,1),sdpa_tflops=round(fl/(sd_min*1e-3)/1e12,1)))
        print(f"B={B} H={H} M={M} N={N} D={D} err={err:.3e} triton={tr_min:.4f}ms({fl/(tr_min*1e-3)/1e12:.0f}TF) sdpa={sd_min:.4f}ms({fl/(sd_min*1e-3)/1e12:.0f}TF) sdpa/triton={sd_min/tr_min:.2f}x")
    print("RESULT_JSON",json.dumps(rows)); print("max_err=",max(r["err"] for r in rows))

if __name__=="__main__": main()
