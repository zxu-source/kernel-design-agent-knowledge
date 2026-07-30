# H200 upstream source run: DMMA FP64 Tensor Core GEMM

- Source: `mirrors/cuda-samples@b7c5481c556c3fe98db060207ecaa41a4b9a9abc`
- File: `cpp/3_CUDA_Features/dmmaTensorCoreGemm/dmmaTensorCoreGemm.cu`
- SHA256: `4e61bd5afa4e36cf68df808f5a5d80a231d2cdefd6f5c6e0fc7856174118c509`
- Build: `nvcc -O3 -arch=sm_90 -I Common dmmaTensorCoreGemm.cu`

The unmodified upstream source compiled and ran on NVIDIA H200 (SM90):

```text
M: 8192 (8 x 1024)
N: 8192 (8 x 1024)
K: 4096 (4 x 1024)
Required shared memory size: 68 Kb
Computing using high performance kernel = 0 - compute_dgemm_async_copy
Time: 21.293535 ms
FP64 TFLOPS: 25.82
```

The source reports a single CUDA-event interval and this run has no external
reference comparison. The values above are upstream sample output only; this
record is `runnable`, not a validated benchmark or speedup claim.
