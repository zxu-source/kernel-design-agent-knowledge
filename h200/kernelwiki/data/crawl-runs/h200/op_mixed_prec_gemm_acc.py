#!/usr/bin/env python3
"""H200 Mixed-Precision GEMM Accumulator Robustness validation.
bf16 GEMM C=A@B: compares fp32 accumulator (correct) vs bf16 accumulator (lossy).
For large K, bf16 accumulation drops sub-ULP partial sums -> output drift / inf.
fp32 accumulator is the robustness fix. PURPOSE = robustness. H200, Triton 3.6.
"""
import json, torch, triton, triton.language as tl
def cdiv(a,b): return (a+b-1)//b
@triton.jit
def gemm_fp32acc(a_ptr,b_ptr,c_ptr,M,N,K,sam,sbk,sbn,scm,scn,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    pm=tl.program_id(0); pn=tl.program_id(1)
    om=pm*BM+tl.arange(0,BM); on=pn*BN+tl.arange(0,BN)
    acc=tl.zeros((BM,BN),dtype=tl.float32)             # fp32 accumulator
    for k0 in range(0,K,BK):
        ok=k0+tl.arange(0,BK)
        a=tl.load(a_ptr+om[:,None]*sam+ok[None,:],mask=(om[:,None]<M)&(ok[None,:]<K),other=0.0)
        b=tl.load(b_ptr+ok[:,None]*sbk+on[None,:]*sbn,mask=(ok[:,None]<K)&(on[None,:]<N),other=0.0)
        acc+=tl.dot(a,b)
    tl.store(c_ptr+om[:,None]*scm+on[None,:]*scn,acc,mask=(om[:,None]<M)&(on[None,:]<N))
@triton.jit
def gemm_bf16acc(a_ptr,b_ptr,c_ptr,M,N,K,sam,sbk,sbn,scm,scn,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    pm=tl.program_id(0); pn=tl.program_id(1)
    om=pm*BM+tl.arange(0,BM); on=pn*BN+tl.arange(0,BN)
    acc=tl.zeros((BM,BN),dtype=tl.bfloat16)            # bf16 accumulator (lossy)
    for k0 in range(0,K,BK):
        ok=k0+tl.arange(0,BK)
        a=tl.load(a_ptr+om[:,None]*sam+ok[None,:],mask=(om[:,None]<M)&(ok[None,:]<K),other=0.0)
        b=tl.load(b_ptr+ok[:,None]*sbk+on[None,:]*sbn,mask=(ok[:,None]<K)&(on[None,:]<N),other=0.0)
        acc+=tl.dot(a,b).to(tl.bfloat16)               # truncate to bf16 each step
    tl.store(c_ptr+om[:,None]*scm+on[None,:]*scn,acc.to(tl.float32),mask=(om[:,None]<M)&(on[None,:]<N))
def run(kern,a,b,BM=128,BN=128,BK=32,nw=4,ns=3):
    M,K=a.shape;_,N=b.shape;c=torch.empty((M,N),device=a.device,dtype=torch.float32)
    kern[(cdiv(M,BM),cdiv(N,BN))](a,b,c,M,N,K,a.stride(0),b.stride(0),b.stride(1),c.stride(0),c.stride(1),BM=BM,BN=BN,BK=BK,num_warps=nw,num_stages=ns);return c
def main():
    print(f"triton={triton.__version__} dev={torch.cuda.get_device_name(0)} SMs={torch.cuda.get_device_properties(0).multi_processor_count}")
    rows=[]
    for M,N,K in [(2048,2048,2048),(4096,4096,4096),(4096,4096,8192),(2048,2048,16384),(2048,2048,32768)]:
        torch.manual_seed(0)
        a=torch.randn(M,K,device="cuda",dtype=torch.bfloat16)*1.0; b=torch.randn(K,N,device="cuda",dtype=torch.bfloat16)*1.0
        ref=a.float()@b.float()
        cf=run(gemm_fp32acc,a,b); cb=run(gemm_bf16acc,a,b)
        err32=(cf-ref).abs().max().item(); errbf=(cb-ref).abs().max().item()
        rel32=err32/(ref.abs().max().item()+1e-9); relbf=errbf/(ref.abs().max().item()+1e-9)
        bf_inf=torch.isinf(cb).any().item()
        rows.append(dict(M=M,N=N,K=K,rel_fp32=round(rel32,5),rel_bf16=round(relbf,5),bf16_inf=bool(bf_inf),err_ratio=round(relbf/max(rel32,1e-9),1)))
        print(f"M={M} N={N} K={K} fp32_rel={rel32:.2e} bf16_rel={relbf:.2e} ratio={relbf/max(rel32,1e-9):.1f}x bf16_inf={bf_inf}")
    print("RESULT_JSON", json.dumps(rows))
if __name__=="__main__": main()
