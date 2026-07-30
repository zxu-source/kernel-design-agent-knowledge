#!/usr/bin/env python3
"""H200 FP8 (e4m3) matmul via Triton on Hopper (SM90).

The H200 has FP8 (e4m3/e5m2) tensor cores (wgmma fp8). This harness runs a
Triton FP8 e4m3 matmul (fp32 accumulation), validates correctness against a
dequantized fp32 reference, and measures attained TFLOPS vs a BF16 matmul on the
same shapes. FP8 is the H200 efficiency headline (~2x the TFLOPS of BF16).

Hardware: NVIDIA H200, 132 SMs, cc 9.0. Triton 3.6.0, PyTorch 2.11.0+cu130.
"""
import json, statistics
import torch, triton, triton.language as tl

def cdiv(a, b): return (a + b - 1) // b

@triton.jit
def matmul_fp8(a_ptr, b_ptr, c_ptr, M, N, K,
               stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0); pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        offs_k = k0 + tl.arange(0, BK)
        a = tl.load(a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
                    mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                    mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc=acc, out_dtype=tl.float32)   # fp8 x fp8 -> fp32 (Hopper wgmma fp8)
    tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, acc,
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

@triton.jit
def matmul_bf16(a_ptr, b_ptr, c_ptr, M, N, K,
                stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0); pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        offs_k = k0 + tl.arange(0, BK)
        a = tl.load(a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
                    mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                    mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc=acc)
    tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, acc,
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

def run(kernel, a, b, BM, BN, BK, nw, ns):
    M, K = a.shape; _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    grid = (cdiv(M, BM), cdiv(N, BN))
    kernel[grid](a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        BM=BM, BN=BN, BK=BK, num_warps=nw, num_stages=ns)
    return c

def time_fn(fn, trials=20):
    for _ in range(5): fn(); torch.cuda.synchronize()
    ts=[]
    for _ in range(trials):
        s,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    return min(ts), statistics.median(ts)

def main():
    print(f"triton={triton.__version__} dev={torch.cuda.get_device_name(0)} SMs={torch.cuda.get_device_properties(0).multi_processor_count}")
    f8 = torch.float8_e4m3fn
    shapes = [(2048,2048,2048),(4096,4096,1024),(4096,4096,4096),(8192,8192,1024)]
    BM=BN=128; BK=128; nw=4; ns=3
    rows=[]
    print(f"config BM={BM} BN={BN} BK={BK} nw={nw} ns={ns}")
    for M,N,K in shapes:
        torch.manual_seed(0)
        af = (torch.randn(M,K,device="cuda",dtype=torch.float32)*0.5).to(f8)
        bf = (torch.randn(K,N,device="cuda",dtype=torch.float32)*0.5).to(f8)
        ab = af.to(torch.bfloat16); bb = bf.to(torch.bfloat16)
        # references
        ref_f8 = af.to(torch.float32) @ bf.to(torch.float32)        # fp32 matmul of dequantized fp8
        ref_bf = ab.to(torch.float32) @ bb.to(torch.float32)
        try:
            cf = run(matmul_fp8, af, bf, BM, BN, BK, nw, ns)
        except Exception as ex:
            print(f"M={M} N={N} K={K} fp8 run FAILED: {repr(ex)[:200]}"); continue
        cb = run(matmul_bf16, ab, bb, BM, BN, BK, nw, ns)
        # correctness: fp8 matmul vs dequantized fp32 reference (fp8 quant error expected)
        err_f8_abs = (cf - ref_f8).abs().max().item()
        rel_f8 = err_f8_abs / (ref_f8.abs().mean().item() + 1e-9)
        err_bf_abs = (cb - ref_bf).abs().max().item()
        tf = lambda: run(matmul_fp8, af, bf, BM, BN, BK, nw, ns)
        tb = lambda: run(matmul_bf16, ab, bb, BM, BN, BK, nw, ns)
        tfmin,_ = time_fn(tf); tbmin,_ = time_fn(tb)
        flops = 2*M*N*K
        tflops_f8 = flops/(tfmin*1e-3)/1e12
        tflops_bf = flops/(tbmin*1e-3)/1e12
        rows.append(dict(M=M,N=N,K=K, fp8_ms=round(tfmin,4), bf16_ms=round(tbmin,4),
                         speedup_bf_over_f8_time=round(tbmin/tfmin,3),
                         fp8_tflops=round(tflops_f8,0), bf16_tflops=round(tflops_bf,0),
                         fp8_rel_err=round(rel_f8,5), bf16_abs_err=round(err_bf_abs,4)))
        print(f"M={M} N={N} K={K}  fp8={tfmin:.4f}ms({tflops_f8:.0f}TF) bf16={tbmin:.4f}ms({tflops_bf:.0f}TF)  bf16/fp8_time={tbmin/tfmin:.2f}x  fp8_rel_err={rel_f8:.2e} bf16_abs_err={err_bf_abs:.2e}")
    print("RESULT_JSON", json.dumps(rows))

if __name__=="__main__": main()
