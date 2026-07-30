# H200 causal-mask benchmark replay (corrected correctness check)

Date: 2026-07-21  
Target: `h200_ncu`  
GPU: NVIDIA H200  
Toolchain: Triton 3.6.0

The original error metric subtracted two `-inf` sentinels and therefore produced
NaN. The harness now checks equality of the negative-infinity mask and zero
region separately. Raw H200 stdout:

```text
[h200_ncu] status=ok rc=0 seconds=4.10
triton=3.6.0 dev=NVIDIA H200 SMs=132
N= 1024 correct=True err=0.000e+00 triton=0.0157ms torch=0.0204ms torch/triton=1.30x
N= 2048 correct=True err=0.000e+00 triton=0.0229ms torch=0.0343ms torch/triton=1.50x
N= 4096 correct=True err=0.000e+00 triton=0.0319ms torch=0.1007ms torch/triton=3.16x
N= 8192 correct=True err=0.000e+00 triton=0.0711ms torch=0.3619ms torch/triton=5.09x
```
