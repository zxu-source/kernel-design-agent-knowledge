# H200 upstream source run: IMMA INT8 Tensor Core GEMM

- Source: `mirrors/cuda-samples@b7c5481c556c3fe98db060207ecaa41a4b9a9abc`
- File: `cpp/3_CUDA_Features/immaTensorCoreGemm/immaTensorCoreGemm.cu`
- SHA256: `4f549bf3e3a5bbee03295785283c721ac31914717a547c8e28fac49c4fa0e9ad`
- Build: `nvcc -O3 -arch=sm_90 -I Common immaTensorCoreGemm.cu`

The unmodified upstream source compiled and ran on NVIDIA H200 (SM90):

```text
M: 4096 (16 x 256)
N: 4096 (16 x 256)
K: 4096 (16 x 256)
Required shared memory size: 64 Kb
Computing... using high performance kernel compute_gemm_imma
Time: 0.701728 ms
TOPS: 195.86
```

The source measures one kernel launch and does not include a host reference
comparison in this run. The timing is retained as upstream sample output only;
this evidence establishes `runnable`, not `validated`, `benchmarked`, or a
speedup versus PyTorch/cuBLAS.
