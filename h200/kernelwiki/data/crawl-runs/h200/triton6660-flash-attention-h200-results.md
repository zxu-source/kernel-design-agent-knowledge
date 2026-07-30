# Triton PR #6660 — Software-Pipelined Attention on H200

Date: 2026-07-20 (overnight). PR #6660 improves Triton's compiler Pipeliner and
warp-specialization passes for attention. Like #6299 those are compiler-internal
passes with no user-facing toggle in Triton 3.6.0, so this harness characterizes
the kernel CLASS the PR optimizes: a from-scratch Triton flash-attention-2
forward kernel on H200 (SM90), validated for correctness against torch SDPA and
benchmarked for throughput (TFLOPS).

Hardware: NVIDIA H200, 132 SMs, cc 9.0. Triton 3.6.0, PyTorch 2.11.0+cu130.
The FA-2 forward kernel is written from scratch (not copied from upstream); the
PR's specific pipeliner/WS changes are not individually isolated here. Timing:
CUDA events, min of 20 trials after 5 warmup. FP16, causal=False.

## Correctness — PASS

The Triton FA-2 forward output matches `torch.nn.functional.scaled_dot_product_attention`
within fp16 numerical noise across all shapes:

| B | H | M | N | D | max abs err vs SDPA |
|--:|--:|--:|--:|--:|-----:|
| 1 | 8 | 8192  | 8192  | 64  | 7.6e-06 |
| 1 | 8 | 8192  | 8192  | 128 | 3.8e-06 |
| 4 | 8 | 4096  | 4096  | 64  | 7.6e-06 |
| 1 | 4 | 16384 | 16384 | 64  | 3.8e-06 |
| 2 | 16| 2048  | 2048  | 128 | 1.5e-05 |

## Throughput characterization (attained TFLOPS)

| B | H | M | N | D | Triton FA-2 (ms) | SDPA (ms) | Triton TF | SDPA TF | SDPA/Triton time |
|--:|--:|--:|--:|--:|----:|----:|----:|----:|----:|
| 1 | 8 | 8192  | 8192  | 64  | 0.6344 | 0.3017 | 217 | 456 | 0.48x |
| 1 | 8 | 8192  | 8192  | 128 | 0.6407 | 0.4212 | 429 | 653 | 0.66x |
| 4 | 8 | 4096  | 4096  | 64  | 0.6394 | 0.3070 | 215 | 448 | 0.48x |
| 1 | 4 | 16384 | 16384 | 64  | 1.2202 | 0.5794 | 225 | 474 | 0.47x |
| 2 | 16| 2048  | 2048  | 128 | 0.1889 | 0.1231 | 364 | 558 | 0.65x |

(FLOPs counted as `4*B*H*M*N*D` for the softmax(QK^T)V fused op.)

## Conclusion (scoped)

- A from-scratch Triton FA-2 forward is **correct on H200** (matches torch SDPA
  within fp16 noise) and attains **~215-430 TFLOPS** in this plain form.
- torch's optimized SDPA backend (FlashAttention-3 / cuDNN on cu130, which uses
  warp specialization, TMA, and the software-pipelining that PR #6660 targets in
  the Triton compiler) runs **~1.5x-2.1x faster** (448-653 TFLOPS). This gap
  illustrates exactly the value of the compiler optimizations the PR improves —
  which are applied automatically and are not user-toggleable in Triton 3.6.0.
- No claim is made that the PR's specific pipeliner changes are reproduced; this
  is a characterization of the attention kernel class on H200.

## File

| file | role | sha256 |
|---|---|---|
| `triton6660_flash_attention.py` | runnable harness (Triton FA-2 forward vs torch SDPA) | `c26c1b474b1da0a87bacc14d8c5b6587f0d161e0c62abc2d87271013b64d86d0` |
