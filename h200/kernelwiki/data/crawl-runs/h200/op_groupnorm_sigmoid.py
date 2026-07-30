#!/usr/bin/env python3
"""H200 GroupNorm + Sigmoid Forward (Triton) vs torch.
GroupNorm [N,C] with G groups: normalize within each group per batch element.
Sigmoid: 1/(1+exp(-x)). Both fused elementwise/reduction.
PURPOSE = speedup. H200, Triton 3.6.
"""
import json, statistics
import torch, triton, triton.language as tl

# ---- GroupNorm [N,C], G groups, per-channel weight+bias ----
@triton.jit
def group_norm(x_ptr, w_ptr, b_ptr, o_ptr, N, C, G, cg, eps,
               BLOCK_CG: tl.constexpr):
    # grid: (N*G,). cg = C//G (channels per group).
    ng = tl.program_id(0)
    n = ng // G; g = ng % G
    offs = tl.arange(0, BLOCK_CG); mask = offs < cg
    base = n*C + g*cg + offs  # contiguous group channels in [N,C] row-major
    x = tl.load(x_ptr + base, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / cg
    var = tl.sum((x-mean)*(x-mean), axis=0) / cg
    rrms = tl.rsqrt(var + eps)
    w = tl.load(w_ptr + g*cg + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + g*cg + offs, mask=mask, other=0.0).to(tl.float32)
    y = (x - mean) * rrms * w + b
    tl.store(o_ptr + base, y.to(o_ptr.dtype.element_ty), mask=mask)

def tri_gn(x, w, b, G, eps=1e-5):
    N, C = x.shape; cg = C // G
    o = torch.empty_like(x)
    group_norm[(N*G,)](x, w, b, o, N, C, G, cg, eps, BLOCK_CG=triton.next_power_of_2(cg))
    return o

# ---- Sigmoid forward ----
@triton.jit
def sigmoid_fwd(x_ptr, o_ptr, total, BLOCK: tl.constexpr):
    pid = tl.program_id(0); offs = pid*BLOCK + tl.arange(0, BLOCK); mask = offs < total
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(o_ptr + offs, (1.0 / (1.0 + tl.exp(-x))).to(o_ptr.dtype.element_ty), mask=mask)

def tri_sig(x, BLOCK=4096):
    o = torch.empty_like(x); sigmoid_fwd[(triton.cdiv(x.numel(),BLOCK),)](x, o, x.numel(), BLOCK=BLOCK); return o

def time_fn(fn, trials=50):
    for _ in range(10): fn(); torch.cuda.synchronize()
    ts = []
    for _ in range(trials):
        s,e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    return min(ts), statistics.median(ts)

def main():
    print(f"triton={triton.__version__} dev={torch.cuda.get_device_name(0)} SMs={torch.cuda.get_device_properties(0).multi_processor_count}")
    # GroupNorm
    print("--- GroupNorm ---")
    for N, C, G in [(4096, 4096, 32), (8192, 8192, 32), (4096, 8192, 32), (8192, 4096, 16), (4096, 4096, 8)]:
        dt = torch.float32
        torch.manual_seed(0)
        x = torch.randn(N, C, device="cuda", dtype=dt) * 0.3
        w = torch.randn(C, device="cuda", dtype=dt) * 0.1 + 1.0
        b = torch.zeros(C, device="cuda", dtype=dt)
        ref = torch.nn.functional.group_norm(x, G, w, b, eps=1e-5)
        out = tri_gn(x, w, b, G)
        err = (out - ref).abs().max().item()
        tri = lambda: tri_gn(x, w, b, G)
        tor = lambda: torch.nn.functional.group_norm(x, G, w, b, eps=1e-5)
        tm, _ = time_fn(tri); tm_t, _ = time_fn(tor)
        print(f"  N={N} C={C} G={G} err={err:.3e} triton={tm:.4f}ms torch={tm_t:.4f}ms torch/triton={tm_t/tm:.2f}x")
    # Sigmoid
    print("--- Sigmoid ---")
    for M, N in [(4096, 4096), (8192, 8192), (8192, 14336)]:
        for dt in (torch.float32, torch.bfloat16):
            torch.manual_seed(0); x = torch.randn(M, N, device="cuda", dtype=dt) * 0.5
            ref = torch.sigmoid(x); out = tri_sig(x)
            err = (out.float() - ref.float()).abs().max().item()
            tri = lambda: tri_sig(x); tor = lambda: torch.sigmoid(x)
            tm, _ = time_fn(tri); tm_t, _ = time_fn(tor)
            print(f"  M={M} N={N} {str(dt).split('.')[-1]:8s} err={err:.3e} triton={tm:.4f}ms torch={tm_t:.4f}ms torch/triton={tm_t/tm:.2f}x")
if __name__ == "__main__": main()
