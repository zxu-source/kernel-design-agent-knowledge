#!/usr/bin/env python3
"""H200 Depthwise Convolution (direct, Triton) vs torch conv2d(groups=C).
Each output channel is independent: out[c,oh,ow] = sum_{kh,kw} in[c,oh+kh-pad,ow+kw-pad]*w[c,kh,kw].
Direct kernel (one program per (channel, spatial tile)) gathers 9 taps (kernel=3,
stride=1, pad=1, same). PURPOSE = characterization. vs torch conv2d(groups=C).
H200, Triton 3.6.
"""
import json, statistics
import torch, triton, triton.language as tl
@triton.jit
def dwconv2d(in_ptr, w_ptr, o_ptr, C, H, W, K: tl.constexpr, PAD: tl.constexpr,
             BLOCK: tl.constexpr):
    c=tl.program_id(0); pid=tl.program_id(1)
    p = pid*BLOCK + tl.arange(0, BLOCK)
    oh = p // W; ow = p % W
    mask = p < H*W
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for kh in tl.static_range(K):
        for kw in tl.static_range(K):
            ih = oh + kh - PAD; iw = ow + kw - PAD
            valid = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W) & mask
            wv = tl.load(w_ptr + c*K*K + kh*K + kw)            # scalar weight for this tap
            iv = tl.load(in_ptr + c*H*W + ih*W + iw, mask=valid, other=0.0)
            acc += iv * wv
    tl.store(o_ptr + c*H*W + p, acc, mask=mask)
def tri_dwconv(x, w, K=3, PAD=1, BLOCK=4096):
    C,H,Ww = x.shape; o=torch.empty_like(x)
    dwconv2d[(C, triton.cdiv(H*Ww, BLOCK))](x, w, o, C, H, Ww, K=K, PAD=PAD, BLOCK=BLOCK)
    return o
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
    for C,H,W in [(64,128,128),(128,128,128),(256,64,64),(128,256,256),(64,512,512)]:
        torch.manual_seed(0)
        x=torch.randn(1,C,H,W,device="cuda",dtype=torch.float32)
        w=torch.randn(C,1,3,3,device="cuda",dtype=torch.float32)
        ref=torch.nn.functional.conv2d(x, w, stride=1, padding=1, groups=C)  # [1,C,H,W]
        out=tri_dwconv(x[0], w[:,0])  # [C,H,W]
        err=(out-ref[0]).abs().max().item()
        rel=err/(ref.abs().max().item()+1e-9)
        tri=lambda: tri_dwconv(x[0], w[:,0])
        tor=lambda: torch.nn.functional.conv2d(x, w, stride=1, padding=1, groups=C)
        tm,_=time_fn(tri); tm_t,_=time_fn(tor)
        rows.append(dict(C=C,H=H,W=W,err=round(err,4),rel=round(rel,5),triton_ms=round(tm,4),torch_ms=round(tm_t,4),torch_over_triton=round(tm_t/tm,3)))
        print(f"C={C} H={H} W={W} err={err:.3e} rel={rel:.2e} triton={tm:.4f}ms torch={tm_t:.4f}ms torch/triton={tm_t/tm:.2f}x")
    print("RESULT_JSON", json.dumps(rows))
if __name__=="__main__": main()
