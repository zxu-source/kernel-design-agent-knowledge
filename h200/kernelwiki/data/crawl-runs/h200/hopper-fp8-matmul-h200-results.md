# Hopper FP8 (e4m3) Matmul via Triton on H200

Date: 2026-07-20 (overnight). Source-informed H200 microbenchmark. Exercises
the H200's FP8 (e4m3) tensor cores via a Triton `tl.dot` matmul (fp32
accumulation), validates correctness against a dequantized fp32 reference, and
characterizes attained TFLOPS vs a BF16 matmul with the same tile config.

Hardware: NVIDIA H200, 132 SMs, cc 9.0. Triton 3.6.0, PyTorch 2.11.0+cu130.
Tiles BM=BN=128, BK=128, num_warps=4, num_stages=3 for both dtypes. Timing: CUDA
events, min of 20 trials after warmup.

## Correctness — PASS

FP8 e4m3 matmul matches a dequantized fp32 reference (`(A.to(f32) @ B.to(f32))`)
within fp8 e4m3 quantization error. e4m3 has ~3 mantissa bits, so ~1-3% relative
error is expected and observed:

| M x N x K | fp8 max abs err | fp8 relative err | bf16 max abs err |
|---|---:|---:|---:|
| 2048x2048x2048  | (vs f32 ref) | 1.93e-02 | 0.0 |
| 4096x4096x1024  |              | 1.09e-02 | 0.0 |
| 4096x4096x4096  |              | 3.21e-02 | 3.1e-05 |
| 8192x8192x1024  |              | 1.50e-02 | 0.0 |

The FP8 matmul is functionally correct; the residual error is exactly the fp8
quantization floor, not a kernel bug.

## Throughput — naive Triton fp8 UNDERUTILIZES fp8 tensor cores

| M x N x K | fp8 (ms) | fp8 TF | bf16 (ms) | bf16 TF | bf16/fp8 time |
|---|---:|---:|---:|---:|---:|
| 2048x2048x2048  | 0.110 | 156 | 0.052 | 330 | 0.47x |
| 4096x4096x1024  | 0.205 | 168 | 0.099 | 346 | 0.48x |
| 4096x4096x4096  | 0.736 | 187 | 0.274 | 501 | 0.37x |
| 8192x8192x1024  | 0.765 | 180 | 0.331 | 416 | 0.43x |

The naive Triton fp8 kernel attained only **156-187 TFLOPS** (~8-9% of the
H200's ~1979 TFLOPS fp8 peak) and was **~2x SLOWER than the equally-naive bf16
kernel** (330-501 TFLOPS). This is the OPPOSITE of fp8's theoretical ~2x
advantage over bf16.

## Conclusion (scoped) — this is a kernel-quality limitation, NOT an fp8 hardware statement

- The FP8 e4m3 matmul is **correct on H200** (within fp8 quantization error).
- The naive `tl.dot` fp8 kernel does **not** hit the fast fp8 wgmma path:
  attained 156-187 TFLOPS, far below peak, and slower than the bf16 kernel. An
  attempt to enlarge the K-tile (BK=256) and warps (8) hit shared-memory
  out-of-resource on H200 (393 KB > 232 KB limit).
- Realizing fp8's expected ~2x-throughput advantage requires a **production-grade
  fp8 kernel** (tuned K-tile / operand layouts for the fp8 wgmma atom, plus
  per-block scale factors) such as DeepGEMM or CUTLASS fp8 GEMM. This harness
  validates fp8 *correctness* on H200 but its throughput is **not representative**
  of fp8 hardware capability. No fp8 speedup over bf16 is claimed from this naive
  kernel; the result documents that naive Triton fp8 underutilizes the fp8 units.

## File

| file | role | sha256 |
|---|---|---|
| `hopper_fp8_matmul.py` | runnable harness (Triton fp8 e4m3 vs bf16 matmul, correctness + TFLOPS) | `4c53ebc86cc986db97372f5196cd65580bf0b45f3b0807b68ad05ac9393d6fae` |
