#!/usr/bin/env python3
"""H200 Batched Matmul (bmm) (Triton) vs torch.bmm.
[B,M,K] @ [B,K,N] -> [B,M,N]. Grid (B, M-tiles, N-tiles); each program one batch's
tile. PURPOSE = characterization. vs torch.bmm (cuBLAS strided batched). H200.
"""
import json, statistics
import torch, triton, triton.language as tl
def cdiv(a,b): return (a+b-1)//b
@triton.jit
def bmm(a_ptr,b_ptr,o_ptr,B,M,N,K,sab,sam,sak,sbb,sbk,sbn,sob,som,son,
        BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    pbb=tl.program_id(0); pm=tl.program_id(1); pn=tl.program_id(2)
    om=pm*BM+tl.arange(0,BM); on=pn*BN+tl.arange(0,BN)
    acc=tl.zeros((BM,BN),dtype=tl.float32)
    a_base=a_ptr+pbb*sab; b_base=b_ptr+pbb*sbb
    for k0 in range(0,K,BK):
        ok=k0+tl.arange(0,BK)
        a=tl.load(a_base+om[:,None]*sam+ok[None,:]*sak,mask=(om[:,None]<M)&(ok[None,:]<K),other=0.0)
        b=tl.load(b_base+ok[:,None]*sbk+on[None,:]*sbn,mask=(ok[:,None]<K)&(on[None,:]<N),other=0.0)
        acc+=tl.dot(a,b)
    o_base=o_ptr+pbb*sob
    tl.store(o_base+om[:,None]*som+on[None,:]*son, acc, mask=(om[:,None]<M)&(on[None,:]<N))
def tri_bmm(A,B,BM=128,BN=128,BK=64,nw=4,ns=3):
    Bb,M,K=A.shape; _,_,N=B.shape; O=torch.empty((Bb,M,N),device=A.device,dtype=torch.float32)
    bmm[(Bb,cdiv(M,BM),cdiv(N,BN))](A,B,O,Bb,M,N,K,A.stride(0),A.stride(1),1,B.stride(0),B.stride(1),1,O.stride(0),O.stride(1),1,BM=BM,BN=BN,BK=BK,num_warps=nw,num_stages=ns); return O
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
    for Bb,M,N,K in [(16,512,512,512),(32,1024,1024,1024),(16,2048,2048,2048),(8,2048,2048,4096),(32,1024,1024,2048)]:
        A=torch.randn(Bb,M,K,device="cuda",dtype=torch.bfloat16)*0.3; Bb2=torch.randn(Bb,K,N,device="cuda",dtype=torch.bfloat16)*0.3
        ref=(A.float()@Bb2.float())
        out=tri_bmm(A,Bb2)
        err=(out-ref).abs().max().item()
        tri=lambda: tri_bmm(A,Bb2); tor=lambda: torch.bmm(A,Bb2)
        tm,_=time_fn(tri); tm_t,_=time_fn(tor)
        fl=2*Bb*M*N*K; tt=fl/(tm*1e-3)/1e12; ct=fl/(tm_t*1e-3)/1e12
        rows.append(dict(B=Bb,M=M,N=N,K=K,err=round(err,3),triton_ms=round(tm,4),torch_ms=round(tm_t,4),triton_tf=round(tt,0),torch_tf=round(ct,0),torch_over_triton=round(tm_t/tm,3)))
        print(f"B={Bb} M={M} N={N} K={K} err={err:.3f} triton={tm:.4f}ms({tt:.0f}TF) torch={tm_t:.4f}ms({ct:.0f}TF) torch/triton={tm_t/tm:.2f}x")
    print("RESULT_JSON", json.dumps(rows))
if __name__=="__main__": main()
