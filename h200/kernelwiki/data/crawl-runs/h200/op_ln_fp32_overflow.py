#!/usr/bin/env python3
"""H200 LayerNorm fp32-reduction robustness validation.
Compares LayerNorm variance reduction in fp32 (correct) vs bf16 (lossy). For large
hidden N, bf16 accumulation of sum_x/sum_x2 loses precision (bf16 ~3 decimals,
the sum grows to thousands -> sub-ULP contributions dropped) -> large output error
and potential overflow. fp32 reduction is the robustness fix. PURPOSE = robustness.
H200, Triton 3.6.
"""
import json, statistics
import torch, triton, triton.language as tl

@triton.jit
def ln_fp32(x_ptr,w_ptr,b_ptr,o_ptr,eps,M,N,BLOCK_N:tl.constexpr):
    row=tl.program_id(0); offs=tl.arange(0,BLOCK_N); mask=offs<N
    x=tl.load(x_ptr+row*N+offs, mask=mask, other=0.0).to(tl.float32)   # fp32 reduction
    sx=tl.sum(x,axis=0); sx2=tl.sum(x*x,axis=0); mean=sx/N; var=sx2/N-mean*mean
    rrms=tl.rsqrt(var+eps)
    w=tl.load(w_ptr+offs,mask=mask).to(tl.float32); bb=tl.load(b_ptr+offs,mask=mask).to(tl.float32)
    tl.store(o_ptr+row*N+offs, ((x-mean)*rrms*w+bb).to(o_ptr.dtype.element_ty), mask=mask)

@triton.jit
def ln_bf16(x_ptr,w_ptr,b_ptr,o_ptr,eps,M,N,BLOCK_N:tl.constexpr):
    row=tl.program_id(0); offs=tl.arange(0,BLOCK_N); mask=offs<N
    x=tl.load(x_ptr+row*N+offs, mask=mask, other=0.0)                   # bf16 reduction (lossy)
    sx=tl.sum(x,axis=0); sx2=tl.sum(x*x,axis=0); mean=sx/N; var=sx2/N-mean*mean
    rrms=tl.rsqrt(var.to(tl.float32)+eps)
    w=tl.load(w_ptr+offs,mask=mask); bb=tl.load(b_ptr+offs,mask=mask)
    tl.store(o_ptr+row*N+offs, ((x.to(tl.float32)-mean.to(tl.float32))*rrms*w.to(tl.float32)+bb.to(tl.float32)).to(o_ptr.dtype.element_ty), mask=mask)

def run(kern, x, w, b, eps, BLOCK_N):
    M,N=x.shape; o=torch.empty_like(x)
    kern[(M,)](x,w,b,o,eps,M,N,BLOCK_N=BLOCK_N); return o

def main():
    print(f"triton={triton.__version__} dev={torch.cuda.get_device_name(0)} SMs={torch.cuda.get_device_properties(0).multi_processor_count}")
    eps=1e-5
    rows=[]
    for M,N in [(4096,4096),(8192,8192),(8192,14336),(8192,28672),(8192,57344)]:
        torch.manual_seed(0)
        x=torch.randn(M,N,device="cuda",dtype=torch.bfloat16)*1.0
        w=torch.randn(N,device="cuda",dtype=torch.bfloat16)*0.1+1.0
        b=torch.zeros(N,device="cuda",dtype=torch.bfloat16)
        ref=torch.nn.functional.layer_norm(x,(N,),w,b,eps)
        BN=triton.next_power_of_2(N)
        of32=run(ln_fp32, x, w, b, eps, BN)
        obf=run(ln_bf16, x, w, b, eps, BN)
        err32=(of32.float()-ref.float()).abs().max().item()
        errbf=(obf.float()-ref.float()).abs().max().item()
        rel32=err32/(ref.float().abs().max().item()+1e-9)
        relbf=errbf/(ref.float().abs().max().item()+1e-9)
        has_nan_bf=torch.isnan(obf).any().item()
        rows.append(dict(M=M,N=N,err_fp32=round(err32,4),err_bf16=round(errbf,4),
                         rel_fp32=round(rel32,5),rel_bf16=round(relbf,5),bf16_nan=bool(has_nan_bf)))
        print(f"M={M} N={N:6d} fp32_err={err32:.3e}(rel {rel32:.2e}) bf16_err={errbf:.3e}(rel {relbf:.2e}) bf16_nan={has_nan_bf}")
    print("RESULT_JSON", json.dumps(rows))
if __name__=="__main__": main()
