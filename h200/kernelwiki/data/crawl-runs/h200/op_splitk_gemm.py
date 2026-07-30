#!/usr/bin/env python3
"""H200 Split-K GEMM (Triton) validation.
Split the K reduction across SPLIT blocks; each block computes a partial
[BM,BN] for its K-chunk and atomicAdds to a fp32 output. Increases parallelism
for tall-K / small-MN GEMMs (where the M*N tile grid under-fills the GPU).
PURPOSE = speedup (better occupancy for tall-K). vs single GEMM (full K).
H200, Triton 3.6.
"""
import json, statistics
import torch, triton, triton.language as tl
def cdiv(a,b): return (a+b-1)//b

@triton.jit
def splitk_gemm(a_ptr,b_ptr,c_ptr,M,N,K,sam,sak,sbk,sbn,scm,scn,
                BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,SPLIT:tl.constexpr):
    pm=tl.program_id(0); pn=tl.program_id(1); ps=tl.program_id(2)
    om=pm*BM+tl.arange(0,BM); on=pn*BN+tl.arange(0,BN)
    Kc=tl.cdiv(K,SPLIT)                        # K per split (tl.cdiv for runtime K)
    k0=ps*Kc
    acc=tl.zeros((BM,BN),dtype=tl.float32)
    for kk in range(k0,k0+Kc,BK):
        ok=kk+tl.arange(0,BK)
        a=tl.load(a_ptr+om[:,None]*sam+ok[None,:]*sak,mask=(om[:,None]<M)&(ok[None,:]<K),other=0.0)
        b=tl.load(b_ptr+ok[:,None]*sbk+on[None,:]*sbn,mask=(ok[:,None]<K)&(on[None,:]<N),other=0.0)
        acc+=tl.dot(a,b)
    cptr=c_ptr+om[:,None]*scm+on[None,:]*scn
    tl.atomic_add(cptr, acc, mask=(om[:,None]<M)&(on[None,:]<N))

@triton.jit
def gemm_single(a_ptr,b_ptr,c_ptr,M,N,K,sam,sak,sbk,sbn,scm,scn,
                BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    pm=tl.program_id(0); pn=tl.program_id(1)
    om=pm*BM+tl.arange(0,BM); on=pn*BN+tl.arange(0,BN)
    acc=tl.zeros((BM,BN),dtype=tl.float32)
    for kk in range(0,K,BK):
        ok=kk+tl.arange(0,BK)
        a=tl.load(a_ptr+om[:,None]*sam+ok[None,:]*sak,mask=(om[:,None]<M)&(ok[None,:]<K),other=0.0)
        b=tl.load(b_ptr+ok[:,None]*sbk+on[None,:]*sbn,mask=(ok[:,None]<K)&(on[None,:]<N),other=0.0)
        acc+=tl.dot(a,b)
    tl.store(c_ptr+om[:,None]*scm+on[None,:]*scn,acc,mask=(om[:,None]<M)&(on[None,:]<N))

def tri_splitk(a,b,SPLIT,BM=64,BN=64,BK=32,nw=4,ns=3):
    M,K=a.shape; _,N=b.shape
    c=torch.zeros((M,N),device=a.device,dtype=torch.float32)
    splitk_gemm[(cdiv(M,BM),cdiv(N,BN),SPLIT)](a,b,c,M,N,K,a.stride(0),1,b.stride(0),b.stride(1),c.stride(0),c.stride(1),BM=BM,BN=BN,BK=BK,SPLIT=SPLIT,num_warps=nw,num_stages=ns); return c
def tri_single(a,b,BM=64,BN=64,BK=32,nw=4,ns=3):
    M,K=a.shape; _,N=b.shape
    c=torch.empty((M,N),device=a.device,dtype=torch.float32)
    gemm_single[(cdiv(M,BM),cdiv(N,BN))](a,b,c,M,N,K,a.stride(0),1,b.stride(0),b.stride(1),c.stride(0),c.stride(1),BM=BM,BN=BN,BK=BK,num_warps=nw,num_stages=ns); return c

def time_fn(fn,trials=30):
    for _ in range(8): fn(); torch.cuda.synchronize()
    ts=[]
    for _ in range(trials):
        s,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    return min(ts), statistics.median(ts)

def main():
    torch.backends.cuda.matmul.allow_tf32=False
    print(f"triton={triton.__version__} dev={torch.cuda.get_device_name(0)} SMs={torch.cuda.get_device_properties(0).multi_processor_count}")
    rows=[]
    # tall-K / small-MN shapes where split-K should help
    for M,N,K,SPLIT in [(128,128,8192,8),(256,256,8192,8),(128,128,16384,16),(256,128,8192,8),(512,512,4096,4)]:
        a=torch.randn(M,K,device="cuda",dtype=torch.bfloat16)*0.3; b=torch.randn(K,N,device="cuda",dtype=torch.bfloat16)*0.3
        ref=(a.float()@b.float())
        sk=tri_splitk(a,b,SPLIT); sg=tri_single(a,b)
        err_sk=(sk-ref).abs().max().item(); err_sg=(sg-ref).abs().max().item()
        f_sk=lambda: tri_splitk(a,b,SPLIT); f_sg=lambda: tri_single(a,b)
        tsk,_=time_fn(f_sk); tsg,_=time_fn(f_sg)
        rows.append(dict(M=M,N=N,K=K,SPLIT=SPLIT,err_splitk=round(err_sk,3),err_single=round(err_sg,3),
                         splitk_ms=round(tsk,4),single_ms=round(tsg,4),single_over_splitk=round(tsg/tsk,3)))
        print(f"M={M} N={N} K={K} SPLIT={SPLIT} err_sk={err_sk:.3f} err_sg={err_sg:.3f} splitk={tsk:.4f}ms single={tsg:.4f}ms single/splitk={tsg/tsk:.2f}x")
    print("RESULT_JSON", json.dumps(rows))
if __name__=="__main__": main()
