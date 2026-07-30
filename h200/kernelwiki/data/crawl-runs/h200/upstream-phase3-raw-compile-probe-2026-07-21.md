# H200 upstream source raw compile probe

All probes used `nvcc -O3 -arch=sm_90` on NVIDIA H200 with the verbatim `.cu`
file only. The four samples below require NVIDIA CUDA Samples support headers
that were deliberately not fabricated in their clean one-file bundles.

```text
cudaTensorCoreGemm.cu: fatal error: helper_cuda.h: No such file or directory
simpleAtomicIntrinsics.cu: fatal error: helper_functions.h: No such file or directory
simpleVoteIntrinsics.cu: fatal error: helper_cuda.h: No such file or directory
tf32TensorCoreGemm.cu: fatal error: helper_cuda.h: No such file or directory
```

This is an environment/dependency block, not a source-code correctness failure.
