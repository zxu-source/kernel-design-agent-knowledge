# H200 upstream CUDA Samples batch 2

All samples were built from verbatim Gitee CUDA Samples sources at
`b7c5481c556c3fe98db060207ecaa41a4b9a9abc` with official Common headers and
`nvcc -O3 -arch=sm_90` on NVIDIA H200.

- `LargeKernelParameter`: `Test passed!`; 4 KB and 32,764-byte parameter cases ran.
- `alignedTypes`: all misaligned and aligned data-type checks reported `TEST OK`; 0 failures.
- `bf16TensorCoreGemm`: completed the async-copy BF16 Tensor Core kernel; reported 6.744768 ms / 163.02 TFLOPS.
- `globalToShmemAsyncCopy`: completed `AsyncCopyMultiStageLargeChunk`; result check `PASS`; reported 7605.51 GFLOP/s.
- `streamOrderedAllocation`: both vector-add result checks passed.

The reported timings are CUDA Sample outputs only. No torch, cuBLAS, or exact
KDA baseline was measured, so all five entries are `runnable`, not benchmarked.
