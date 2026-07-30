# H200 upstream source run: separable shared-memory convolution

- Source: `mirrors/cuda-samples@b7c5481c556c3fe98db060207ecaa41a4b9a9abc`
- Files: `convolutionSeparable.cu`, `convolutionSeparable_common.h`,
  `convolutionSeparable_gold.cpp`, `main.cpp`
- Build: `nvcc -O3 -arch=sm_90 -I Common main.cpp convolutionSeparable.cu convolutionSeparable_gold.cpp`

The unmodified four-file upstream sample compiled and passed on NVIDIA H200
(SM90). It performs one warmup, then 16 GPU convolution iterations over a
3072x3072 FP32 image. Its CPU row/column gold reference comparison printed:

```text
Throughput = 108395.5085 MPixels/sec, Time = 0.00009 s
Relative L2 norm: 0.000000E+00
Test passed
```

The throughput is an upstream sample measurement, not a torch or task-native
baseline. This establishes runnable source and source-defined correctness
evidence, not a benchmarked KDA speedup.
