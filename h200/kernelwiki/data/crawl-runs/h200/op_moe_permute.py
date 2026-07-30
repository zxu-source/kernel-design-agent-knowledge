#!/usr/bin/env python3
"""H200 MoE Permute/Unpermute (dispatch gather+scatter) (Triton) vs torch.
MoE dispatch: permute tokens into expert-contiguous order (gather by sort index),
run expert GEMMs, then unpermute back (scatter to original positions). PURPOSE =
characterization (dispatch building block). vs torch fancy indexing on H200.
"""
import json, statistics
import torch, triton, triton.language as tl
def cdiv(a,b): return (a+b-1)//b
# permute: permuted[i, :] = tokens[order[i], :]   (gather)
@triton.jit
def permute_gather(tok_ptr, ord_ptr, out_ptr, M, D, BLOCK_D: tl.constexpr):
    i=tl.program_id(0)
    src=tl.load(ord_ptr+i)
    d=tl.arange(0, BLOCK_D); mask=d<D
    v=tl.load(tok_ptr + src*D + d, mask=mask, other=0.0)
    tl.store(out_ptr + i*D + d, v, mask=mask)
# unpermute: out[order[i], :] = permuted[i, :]   (scatter)
@triton.jit
def unpermute_scatter(perm_ptr, ord_ptr, out_ptr, M, D, BLOCK_D: tl.constexpr):
    i=tl.program_id(0)
    dst=tl.load(ord_ptr+i)
    d=tl.arange(0, BLOCK_D); mask=d<D
    v=tl.load(perm_ptr + i*D + d, mask=mask, other=0.0)
    tl.store(out_ptr + dst*D + d, v, mask=mask)
def tri_permute(tokens, order, D):
    M=tokens.shape[0]; o=torch.empty((M,D),device=tokens.device,dtype=tokens.dtype)
    permute_gather[(M,)](tokens, order, o, M, D, BLOCK_D=triton.next_power_of_2(D)); return o
def tri_unpermute(permuted, order, D):
    M=permuted.shape[0]; o=torch.empty((M,D),device=permuted.device,dtype=permuted.dtype)
    unpermute_scatter[(M,)](permuted, order, o, M, D, BLOCK_D=triton.next_power_of_2(D)); return o
def time_fn(fn,trials=50):
    for _ in range(10): fn(); torch.cuda.synchronize()
    ts=[]
    for _ in range(trials):
        s,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    return min(ts), statistics.median(ts)
def main():
    print(f"triton={triton.__version__} dev={torch.cuda.get_device_name(0)} SMs={torch.cuda.get_device_properties(0).multi_processor_count}")
    rows=[]
    for M,D in [(4096,4096),(8192,8192),(8192,11008),(8192,14336),(16384,14336)]:
        for dt in (torch.float32, torch.bfloat16):
            torch.manual_seed(0); tokens=torch.randn(M,D,device="cuda",dtype=dt)
            order=torch.randperm(M, device="cuda")  # expert-contiguous sort order
            tp=tri_permute(tokens, order, D); rp=tokens[order]
            err_p=(tp-rp).abs().max().item()
            back=tri_unpermute(tp, order, D); rback=torch.zeros_like(rp); rback[order]=rp
            err_u=(back-rback).abs().max().item()
            tri=lambda: (tri_permute(tokens,order,D), tri_unpermute(tp,order,D))
            tor=lambda: (tokens[order], (lambda r:(r.__setitem__(slice(None),None),r)[1])(torch.zeros_like(tokens)))  # placeholder; timed below
            # simpler: time permute only (gather) both
            tp_fn=lambda: tri_permute(tokens,order,D); rp_fn=lambda: tokens[order]
            tm_p,_=time_fn(tp_fn); tm_t,_=time_fn(rp_fn)
            rows.append(dict(M=M,D=D,dtype=str(dt).split(".")[-1],err_p=round(err_p,5),err_u=round(err_u,5),triton_ms=round(tm_p,4),torch_ms=round(tm_t,4),torch_over_triton=round(tm_t/tm_p,3)))
            print(f"M={M} D={D} {str(dt).split('.')[-1]:8s} err_p={err_p:.2e} err_u={err_u:.2e} triton_gather={tm_p:.4f}ms torch_gather={tm_t:.4f}ms torch/triton={tm_t/tm_p:.2f}x")
    print("RESULT_JSON", json.dumps(rows))
if __name__=="__main__": main()
