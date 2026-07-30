#!/usr/bin/env python3
"""H200 Block-Sparse Matmul (spMM, Triton) vs dense-masked.
Only compute the nonzero output tiles given a block-sparsity mask (grid over
nonzero mask entries). vs dense matmul over all tiles then zero masked tiles
(same result). PURPOSE = speedup (skip zero tiles; ~total/nonzero work).
H200, Triton 3.6.
"""
import json, statistics
import torch, triton, triton.language as tl
def cdiv(a,b): return (a+b-1)//b
@triton.jit
def bs_gemm(a_ptr,b_ptr,o_ptr,mb_ptr,nb_ptr, M,N,K,sam,sbk,sbn,scom,scn,
            BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    pid=tl.program_id(0)
    mb=tl.load(mb_ptr+pid); nb=tl.load(nb_ptr+pid)
    om=mb*BM+tl.arange(0,BM); on=nb*BN+tl.arange(0,BN)
    acc=tl.zeros((BM,BN),dtype=tl.float32)
    for kk in range(0,K,BK):
        ok=kk+tl.arange(0,BK)
        a=tl.load(a_ptr+om[:,None]*sam+ok[None,:],mask=(om[:,None]<M)&(ok[None,:]<K),other=0.0)
        b=tl.load(b_ptr+ok[:,None]*sbk+on[None,:]*sbn,mask=(ok[:,None]<K)&(on[None,:]<N),other=0.0)
        acc+=tl.dot(a,b)
    tl.store(o_ptr+om[:,None]*scom+on[None,:]*scn, acc, mask=(om[:,None]<M)&(on[None,:]<N))
@triton.jit
def dense_gemm(a_ptr,b_ptr,o_ptr,M,N,K,sam,sbk,sbn,scom,scn,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    pmb=tl.program_id(0); pnb=tl.program_id(1)
    om=pmb*BM+tl.arange(0,BM); on=pnb*BN+tl.arange(0,BN)
    acc=tl.zeros((BM,BN),dtype=tl.float32)
    for kk in range(0,K,BK):
        ok=kk+tl.arange(0,BK)
        a=tl.load(a_ptr+om[:,None]*sam+ok[None,:],mask=(om[:,None]<M)&(ok[None,:]<K),other=0.0)
        b=tl.load(b_ptr+ok[:,None]*sbk+on[None,:]*sbn,mask=(ok[:,None]<K)&(on[None,:]<N),other=0.0)
        acc+=tl.dot(a,b)
    tl.store(o_ptr+om[:,None]*scom+on[None,:]*scn, acc, mask=(om[:,None]<M)&(on[None,:]<N))
def tri_sparse(a,b,mask,BM=128,BN=128,BK=64,nw=8,ns=3):
    M,K=a.shape; _,N=b.shape
    o=torch.zeros((M,N),device=a.device,dtype=torch.float32)
    nz_mb=torch.tensor([m for m in range(mask.shape[0]) for n in range(mask.shape[1]) if mask[m,n]],device=a.device,dtype=torch.int32)
    nz_nb=torch.tensor([n for m in range(mask.shape[0]) for n in range(mask.shape[1]) if mask[m,n]],device=a.device,dtype=torch.int32)
    nz=nz_mb.numel()
    if nz>0:
        bs_gemm[(nz,)](a,b,o,nz_mb,nz_nb,M,N,K,a.stride(0),b.stride(0),b.stride(1),o.stride(0),o.stride(1),BM=BM,BN=BN,BK=BK,num_warps=nw,num_stages=ns)
    return o
def tri_dense(a,b,BM=128,BN=128,BK=64,nw=8,ns=3):
    M,K=a.shape; _,N=b.shape; o=torch.zeros((M,N),device=a.device,dtype=torch.float32)
    dense_gemm[(cdiv(M,BM),cdiv(N,BN))](a,b,o,M,N,K,a.stride(0),b.stride(0),b.stride(1),o.stride(0),o.stride(1),BM=BM,BN=BN,BK=BK,num_warps=nw,num_stages=ns)
    return o
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
    BM=BN=128
    for M,N,K,density in [(4096,4096,4096,0.25),(4096,4096,4096,0.10),(2048,2048,2048,0.25),(8192,8192,4096,0.25)]:
        torch.manual_seed(0)
        a=torch.randn(M,K,device="cuda",dtype=torch.bfloat16)*0.3; b=torch.randn(K,N,device="cuda",dtype=torch.bfloat16)*0.3
        nmb,nnb=cdiv(M,BM),cdiv(N,BN)
        mask=(torch.rand(nmb,nnb,device="cuda")<density)
        # dense ref (all tiles) restricted to nonzero mask positions should equal sparse
        dens=tri_dense(a,b); sp=tri_sparse(a,b,mask)
        # zero out masked tiles in dense for comparison
        for m in range(nmb):
            for n in range(nnb):
                if not mask[m,n]:
                    dens[m*BM:(m+1)*BM, n*BN:(n+1)*BN]=0
        err=(sp-dens).abs().max().item()
        nz=mask.sum().item(); tot=nmb*nnb
        ts=lambda: tri_sparse(a,b,mask); td=lambda: tri_dense(a,b)
        tsm,_=time_fn(ts); tdm,_=time_fn(td)
        rows.append(dict(M=M,N=N,K=K,density=density,nz=int(nz),tot=int(tot),err=round(err,3),sparse_ms=round(tsm,4),dense_ms=round(tdm,4),dense_over_sparse=round(tdm/tsm,3)))
        print(f"M={M} N={N} K={K} dens={density} tiles={nz}/{tot} err={err:.3f} sparse={tsm:.4f}ms dense={tdm:.4f}ms dense/sparse={tdm/tsm:.2f}x")
    print("RESULT_JSON", json.dumps(rows))
if __name__=="__main__": main()
