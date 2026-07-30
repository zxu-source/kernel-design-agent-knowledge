#!/usr/bin/env python3
"""H200 Matmul Backward (Triton) vs torch.autograd.
C = A@B (A[M,K], B[K,N]). grad_a = grad_c @ B^T [M,K]; grad_b = A^T @ grad_c [K,N].
Two strided GEMMs (one generic kernel handling transpose via strides).
PURPOSE = characterization. vs torch autograd on H200, Triton 3.6.
"""
import json, statistics
import torch, triton, triton.language as tl
def cdiv(a,b): return (a+b-1)//b
@triton.jit
def gemm_g(a_ptr,b_ptr,c_ptr,OI,OJ,R,a_si,a_sr,b_sr,b_sj,
           BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    pi=tl.program_id(0); pj=tl.program_id(1)
    oi=pi*BM+tl.arange(0,BM); oj=pj*BN+tl.arange(0,BN)
    acc=tl.zeros((BM,BN),dtype=tl.float32)
    for r0 in range(0,R,BK):
        r=r0+tl.arange(0,BK)
        a=tl.load(a_ptr+oi[:,None]*a_si+r[None,:]*a_sr,mask=(oi[:,None]<OI)&(r[None,:]<R),other=0.0)
        b=tl.load(b_ptr+r[:,None]*b_sr+oj[None,:]*b_sj,mask=(r[:,None]<R)&(oj[None,:]<OJ),other=0.0)
        acc+=tl.dot(a,b)
    tl.store(c_ptr+oi[:,None]*OJ+oj[None,:],acc,mask=(oi[:,None]<OI)&(oj[None,:]<OJ))
def tri_gemm(a,b,OI,OJ,R,a_si,a_sr,b_sr,b_sj,BM=128,BN=128,BK=64,nw=8,ns=3):
    c=torch.empty((OI,OJ),device=a.device,dtype=torch.float32)
    gemm_g[(cdiv(OI,BM),cdiv(OJ,BN))](a,b,c,OI,OJ,R,a_si,a_sr,b_sr,b_sj,BM=BM,BN=BN,BK=BK,num_warps=nw,num_stages=ns); return c
def tri_mmbwd(A,B,gradc):  # A[M,K] bf16, B[K,N] bf16, gradc[M,N] fp32
    M,K=A.shape; _,N=B.shape
    # grad_a[M,K] = grad_c[M,N] @ B[K,N]^T : a2=gradc(out m, r n) strides N,1; b2=B(r n, out k) strides 1,N
    ga=tri_gemm(gradc, B, M, K, N, N, 1, 1, N)
    # grad_b[K,N] = A[M,K]^T @ grad_c[M,N] : a2=A(out k, r m) strides 1,K; b2=gradc(r m, out n) strides N,1
    gb=tri_gemm(A, gradc, K, N, M, 1, K, N, 1)
    return ga, gb
def time_fn(fn,trials=20):
    for _ in range(5): fn(); torch.cuda.synchronize()
    ts=[]
    for _ in range(trials):
        s,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    return min(ts), statistics.median(ts)
def main():
    torch.backends.cuda.matmul.allow_tf32=False
    print(f"triton={triton.__version__} dev={torch.cuda.get_device_name(0)} SMs={torch.cuda.get_device_properties(0).multi_processor_count}")
    rows=[]
    for M,K,N in [(2048,2048,2048),(4096,4096,4096),(8192,8192,4096),(8192,4096,8192),(4096,4096,8192)]:
        A=torch.randn(M,K,device="cuda",dtype=torch.bfloat16,requires_grad=True)*0.3
        B=torch.randn(K,N,device="cuda",dtype=torch.bfloat16,requires_grad=True)*0.3
        C=A@B; gradc=torch.randn_like(C)
        A.retain_grad(); B.retain_grad(); C.backward(gradc)
        ra=A.grad.clone(); rb=B.grad.clone(); A.grad=None; B.grad=None
        ga,gb=tri_mmbwd(A.detach(),B.detach(),gradc)
        ea=(ga-ra).abs().max().item(); eb=(gb-rb).abs().max().item()
        tri=lambda: tri_mmbwd(A.detach(),B.detach(),gradc)
        # torch backward-only timing: build graph once (forward not timed), time backward
        At=A.detach().requires_grad_(True); Bt=B.detach().requires_grad_(True); Ct=At@Bt
        def tor():
            if At.grad is not None: At.grad=None
            if Bt.grad is not None: Bt.grad=None
            Ct.backward(gradc, retain_graph=True)
        tm,_=time_fn(tri); tm_t,_=time_fn(tor)
        rows.append(dict(M=M,K=K,N=N,err_a=round(ea,3),err_b=round(eb,3),triton_ms=round(tm,4),torch_ms=round(tm_t,4),torch_over_triton=round(tm_t/tm,3)))
        print(f"M={M} K={K} N={N} err_a={ea:.3f} err_b={eb:.3f} triton={tm:.4f}ms torch={tm_t:.4f}ms torch/triton={tm_t/tm:.2f}x")
    print("RESULT_JSON", json.dumps(rows))
if __name__=="__main__": main()
