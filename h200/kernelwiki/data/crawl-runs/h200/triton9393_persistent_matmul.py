#!/usr/bin/env python3
"""H200 validation for Triton PR #9393 (persistent matmul for FP32 inputs).

Source-informed microbenchmark. The PR extends Triton's persistent matmul
scheduling to FP32/TF32 inputs and adds resource heuristics (32 KB SMEM
reservation, stage-3 cap, disable for m*n*k < 131072). The PR ships no
benchmarks, so this harness tests the underlying claim directly: that a
*persistent* (fixed-size grid looping over tiles) FP32/TF32 matmul matches the
correctness of a standard static-grid matmul and changes wall-clock latency.

Both variants below share the SAME tile sizes, num_warps, and num_stages; they
differ ONLY in grid/loop structure (persistent vs one-program-per-tile). That
isolates the scheduling effect from everything else.

Hardware: NVIDIA H200, 132 SMs, cc 9.0. Triton 3.6.0. TF32 matmul.
Timing: CUDA events, min of TRIALS trials after warmup.
"""
import json, statistics
import torch, triton, triton.language as tl

NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count

def cdiv(a, b): return (a + b - 1) // b

@triton.jit
def matmul_standard(a_ptr, b_ptr, c_ptr, M, N, K,
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
        acc += tl.dot(a, b)            # TF32 on Hopper for fp32 inputs
    tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, acc,
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

@triton.jit
def matmul_persistent(a_ptr, b_ptr, c_ptr, M, N, K,
                      stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                      BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                      GRID_M: tl.constexpr, NUM_PID: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_m = GRID_M
    num_pid_n = tl.cdiv(N, BN)
    num_tiles = num_pid_m * num_pid_n
    for tile_idx in range(pid, num_tiles, NUM_PID):   # round-robin over persistent programs
        pid_m = tile_idx // num_pid_n
        pid_n = tile_idx % num_pid_n
        offs_m = pid_m * BM + tl.arange(0, BM)
        offs_n = pid_n * BN + tl.arange(0, BN)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, K, BK):
            offs_k = k0 + tl.arange(0, BK)
            a = tl.load(a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
                        mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
            b = tl.load(b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                        mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
            acc += tl.dot(a, b)
        tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, acc,
                 mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

def run_standard(a, b, BM, BN, BK, nw, ns):
    M, K = a.shape; K2, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    grid = (cdiv(M, BM), cdiv(N, BN))
    matmul_standard[grid](a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        BM=BM, BN=BN, BK=BK, num_warps=nw, num_stages=ns)
    return c

def run_persistent(a, b, BM, BN, BK, nw, ns, grid_mul=1):
    M, K = a.shape; K2, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    grid_m = cdiv(M, BM); grid_n = cdiv(N, BN)
    num_pid = NUM_SMS * grid_mul
    num_pid = min(num_pid, grid_m * grid_n)   # don't launch more programs than tiles
    matmul_persistent[(num_pid,)](a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        BM=BM, BN=BN, BK=BK, GRID_M=grid_m, NUM_PID=num_pid, num_warps=nw, num_stages=ns)
    return c

def time_fn(fn, trials=15):
    for _ in range(5): fn(); torch.cuda.synchronize()
    ts = []
    for _ in range(trials):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return min(ts), statistics.median(ts)

def main():
    torch.backends.cuda.matmul.allow_tf32 = True   # match Triton TF32 dot
    print(f"triton={triton.__version__} dev={torch.cuda.get_device_name(0)} SMs={NUM_SMS} allow_tf32=True")
    # tile config + shapes. BM=BN=128, BK=32; num_stages=3 (PR's persistent FP32 cap).
    BM = BN = 128; BK = 32; nw = 4; ns = 3
    shapes = [
        (256, 256, 256),     # small-ish, ~4 tiles -> persistent has few tiles
        (1024, 1024, 1024),  # 8x8=64 tiles, <1 wave on 132 SMs
        (2048, 2048, 2048),  # 16x16=256 tiles, ~2 waves
        (4096, 4096, 1024),  # 32x32=1024 tiles, ~8 waves
        (512, 512, 4096),    # 4x4=16 tiles (tail-effect case), long K
    ]
    rows = []
    for M, N, K in shapes:
        a = torch.randn(M, K, device="cuda", dtype=torch.float32)
        b = torch.randn(K, N, device="cuda", dtype=torch.float32)
        ref = a @ b
        cs = run_standard(a, b, BM, BN, BK, nw, ns)
        cp = run_persistent(a, b, BM, BN, BK, nw, ns)
        err_s = (cs - ref).abs().max().item(); err_p = (cp - ref).abs().max().item()
        agree = (cs - cp).abs().max().item()
        ts_min, ts_med = time_fn(lambda: run_standard(a, b, BM, BN, BK, nw, ns))
        tp_min, tp_med = time_fn(lambda: run_persistent(a, b, BM, BN, BK, nw, ns))
        speedup = ts_min / tp_min if tp_min > 0 else float("nan")
        mkk = M * N * K
        rows.append(dict(M=M, N=N, K=K, tiles=cdiv(M,BM)*cdiv(N,BN), mkk=mkk,
                         std_ms=round(ts_min,4), pers_ms=round(tp_min,4),
                         speedup=round(speedup,3), err_std=err_s, err_pers=err_p,
                         agree=agree))
        print(f"M={M:5d} N={N:5d} K={K:5d} tiles={cdiv(M,BM)*cdiv(N,BN):5d}  "
              f"std={ts_min:8.4f}ms pers={tp_min:8.4f}ms speedup={speedup:5.3f}x  "
              f"err_std={err_s:.3e} err_pers={err_p:.3e} |std-pers|={agree:.3e}")
    print("RESULT_JSON", json.dumps(rows))
    print("max_err_std=", max(r["err_std"] for r in rows), "max_err_pers=", max(r["err_pers"] for r in rows),
          "max_|std-pers|=", max(r["agree"] for r in rows))

if __name__ == "__main__":
    main()
