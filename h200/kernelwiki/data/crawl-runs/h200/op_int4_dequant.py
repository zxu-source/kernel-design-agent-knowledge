#!/usr/bin/env python3
"""H200 INT4 -> BF16 Dequantization (W4A16) (Triton) vs torch.
INT4 weights packed 2-per-byte (uint8). Dequant: unpack nibbles (two's-complement
int4), multiply by per-row scale, output bf16. W4A16: 4-bit weights, bf16 act,
matmul in bf16 after on-the-fly dequant -> 4x weight-memory reduction.
PURPOSE = speedup (memory bandwidth). vs torch reference on H200, Triton 3.6.
"""
import json, statistics
import torch, triton, triton.language as tl
# nibble unpack + two's-complement int4 (0..7 -> 0..7; 8..15 -> -8..-1)
@triton.jit
def int4_dequant(pack_ptr, scale_ptr, o_ptr, M, N, BLOCK_N: tl.constexpr):
    row=tl.program_id(0)
    on=tl.arange(0, BLOCK_N); mask=on<N
    byte_idx=on//2
    hi=on%2
    raw=tl.load(pack_ptr + row*(N//2) + byte_idx, mask=mask, other=0).to(tl.uint8)
    nib=(raw >> (hi*4)) & 0xF
    # two's-complement int4: subtract 16 if >=8
    val=(nib.to(tl.int32) ^ 0x8) - 0x8
    scale=tl.load(scale_ptr+row).to(tl.float32)
    out=val.to(tl.float32)*scale
    tl.store(o_ptr+row*N+on, out.to(tl.bfloat16), mask=mask)
def tri_dequant(pack, scale):
    M=pack.shape[0]; N=pack.shape[1]*2
    o=torch.empty((M,N),device=pack.device,dtype=torch.bfloat16)
    int4_dequant[(M,)](pack, scale, o, M, N, BLOCK_N=triton.next_power_of_2(N))
    return o
def torch_dequant(pack, scale):
    M=pack.shape[0]; N=pack.shape[1]*2
    low=(pack & 0xF).to(torch.int32); hi=((pack>>4)&0xF).to(torch.int32)
    low=torch.where(low>=8, low-16, low); hi=torch.where(hi>=8, hi-16, hi)
    intw=torch.stack([low,hi],dim=-1).reshape(M,N)
    return (intw.to(torch.float32)*scale[:,None].float()).to(torch.bfloat16)
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
    for M,N in [(4096,4096),(8192,8192),(8192,11008),(8192,14336),(16384,14336)]:
        torch.manual_seed(0)
        pack=torch.randint(0,256,(M,N//2),device="cuda",dtype=torch.uint8)
        scale=torch.rand(M,device="cuda",dtype=torch.float32)*0.05+0.01
        ref=torch_dequant(pack,scale); out=tri_dequant(pack,scale)
        err=(out.float()-ref.float()).abs().max().item()
        match=(out==ref).float().mean().item()
        tri=lambda: tri_dequant(pack,scale); tor=lambda: torch_dequant(pack,scale)
        tm,_=time_fn(tri); tm_t,_=time_fn(tor)
        rows.append(dict(M=M,N=N,err=round(err,4),match=round(match,4),triton_ms=round(tm,4),torch_ms=round(tm_t,4),torch_over_triton=round(tm_t/tm,3)))
        print(f"M={M} N={N} err={err:.3e} match={match:.4f} triton={tm:.4f}ms torch={tm_t:.4f}ms torch/triton={tm_t/tm:.2f}x")
    print("RESULT_JSON", json.dumps(rows))
if __name__=="__main__": main()
