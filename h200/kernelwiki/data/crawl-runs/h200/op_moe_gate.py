#!/usr/bin/env python3
"""H200 MoE Top-K Gating (fused top-k + softmax) (Triton) vs torch.
MoE router: per token, take top-k expert logits (k=2), softmax over them ->
routing weights. Fuses top-k selection + softmax over the k winners.
PURPOSE = speedup (fused gating). vs torch (logits.topk(k) + softmax) on H200.
"""
import json, statistics
import torch, triton, triton.language as tl
@triton.jit
def moe_gate(logits_ptr, w_ptr, idx_ptr, M, E, K: tl.constexpr, BLOCK_E: tl.constexpr):
    row=tl.program_id(0)
    offs=tl.arange(0, BLOCK_E); mask=offs<E
    x=tl.load(logits_ptr+row*E+offs, mask=mask, other=-1e30).to(tl.float32)
    # k-pass top-k (k small)
    for k in tl.static_range(K):
        m=tl.max(x, axis=0); sel = x==m
        idx=tl.argmax(tl.where(sel, offs.to(tl.float32), -1e30), axis=0)
        # store the picked logit at slot k; mask out
        tl.store(idx_ptr + row*K + k, idx.to(tl.int64))
        tl.store(w_ptr + row*K + k, m)        # temp store raw logit; softmax later
        x = tl.where(sel, -1e30, x)
    # softmax over the K stored raw logits (w_ptr[row*K : row*K+K])
    kw=tl.load(w_ptr + row*K + tl.arange(0, K))      # [K]
    mx=tl.max(kw, axis=0)
    e=tl.exp(kw - mx); s=tl.sum(e, axis=0)
    tl.store(w_ptr + row*K + tl.arange(0, K), e/s)
def tri_gate(logits, K):
    M,E=logits.shape
    w=torch.empty((M,K),device=logits.device,dtype=torch.float32); idx=torch.empty((M,K),device=logits.device,dtype=torch.int64)
    moe_gate[(M,)](logits, w, idx, M, E, K=K, BLOCK_E=triton.next_power_of_2(E))
    return w, idx
def torch_gate(logits, K):
    val, idx = logits.topk(K, dim=-1)
    return torch.softmax(val, dim=-1), idx
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
    K=2
    for M,E in [(4096,64),(8192,128),(8192,256),(8192,64),(16384,128)]:  # (tokens, experts)
        torch.manual_seed(0); logits=torch.randn(M,E,device="cuda",dtype=torch.float32)
        tw,ti=tri_gate(logits,K); rw,ri=torch_gate(logits,K)
        # compare: weights at the sorted expert indices, and index sets
        werr=(tw-rw).abs().max().item()  # weights may be in different order (per topk tie); sort rows
        sw = torch.sort(tw, -1).values - torch.sort(rw, -1).values
        werr=sw.abs().max().item()
        iset=sum(len(set(ti[m].tolist())&set(ri[m].tolist())) for m in range(M))/(M*K)
        tri=lambda: tri_gate(logits,K); tor=lambda: torch_gate(logits,K)
        tm,_=time_fn(tri); tm_t,_=time_fn(tor)
        rows.append(dict(M=M,E=E,K=K,werr=round(werr,5),set_match=round(iset,4),triton_ms=round(tm,4),torch_ms=round(tm_t,4),torch_over_triton=round(tm_t/tm,3)))
        print(f"M={M} E={E} K={K} werr={werr:.3e} set_match={iset:.4f} triton={tm:.4f}ms torch={tm_t:.4f}ms torch/triton={tm_t/tm:.2f}x")
    print("RESULT_JSON", json.dumps(rows))
if __name__=="__main__": main()
