# H200 aligned validation: globalToShmemAsyncCopy

- Upstream source: `mirrors/cuda-samples@b7c5481c556c3fe98db060207ecaa41a4b9a9abc`
- File: `cpp/3_CUDA_Features/globalToShmemAsyncCopy/globalToShmemAsyncCopy.cu`
- SHA256: `1186a716aae34dde931c3b4ea955c6b5012faa72d1a54f0a425dad6282b7bf40`
- H200 build/run: `nvcc -O3 -arch=sm_90` with the matching CUDA Samples Common headers.

## Matched microbenchmark policy

The source performs one warmup, executes `nIter = 100` launches, and reports
`msecTotal / nIter` (`globalToShmemAsyncCopy.cu:807-890`).  Its selected
`AsyncCopyMultiStageLargeChunk` run uses 1280x1280 FP32 matrices, reports
`Time=0.551 msec`, and ends with `Result = PASS`.

On the same H200, the task-native reference `torch.matmul(a, b)` for 1280x1280
FP32 matrices was run with one warmup and 100 CUDA-event iterations.  Its
measured mean latency was `0.1217440033 ms`.

## Result and scope

The upstream source is correct under its own result check, but its latency is
`4.53x` the PyTorch reference (equivalently, torch/upstream is `0.221x`) for
this aligned microbenchmark.  This is a negative performance result: it is
recorded as `validated`, not `benchmarked` or `kda-ready`, and must not be
selected as a speedup candidate without a new task-specific result.
