# H200 non-causal FA2 benchmark replay

Date: 2026-07-21  
Target: `h200_ncu`  
GPU: NVIDIA H200  
Toolchain: Triton 3.6.0

Raw H200 stdout from `artifacts/kernels/triton-fa2-hopper/variants/flash_attention_fwd_h200.py`:

```text
[h200_ncu] status=ok rc=0 seconds=4.24
B=1 H=8 M=8192 N=8192 D=64 err=7.629e-06 triton=0.6236ms(220TF) sdpa=0.2995ms(459TF) sdpa/triton=0.48x
B=1 H=8 M=8192 N=8192 D=128 err=3.815e-06 triton=0.6383ms(431TF) sdpa=0.4204ms(654TF) sdpa/triton=0.66x
B=4 H=8 M=4096 N=4096 D=64 err=7.629e-06 triton=0.6260ms(220TF) sdpa=0.3026ms(454TF) sdpa/triton=0.48x
B=1 H=4 M=16384 N=16384 D=64 err=3.815e-06 triton=1.2130ms(227TF) sdpa=0.5753ms(478TF) sdpa/triton=0.47x
B=2 H=16 M=2048 N=2048 D=128 err=1.526e-05 triton=0.1889ms(364TF) sdpa=0.1229ms(559TF) sdpa/triton=0.65x
```

Correctness passed within fp16 tolerance. This local FA2 implementation remains
slower than the optimized torch SDPA backend for every tested shape.
