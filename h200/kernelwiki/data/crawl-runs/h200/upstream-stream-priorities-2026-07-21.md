# H200 upstream source run: StreamPriorities

- Source: `mirrors/cuda-samples@b7c5481c556c3fe98db060207ecaa41a4b9a9abc`
- File: `cpp/3_CUDA_Features/StreamPriorities/StreamPriorities.cu`
- SHA256: `0b1eff9fde84fef6f4e261edde75d2a07729468fe59437259786adf0d574b00b`
- Build: `nvcc -O3 -arch=sm_90 -I Common StreamPriorities.cu`

The unmodified upstream sample compiled and exited successfully on NVIDIA H200
(SM90). Its source executes `memcmp` checks for both copied outputs before
`EXIT_SUCCESS`; therefore the successful run is a source-defined correctness
smoke test.

```text
CUDA stream priority range: LOW: 0 to HIGH: -5
elapsed time of kernels launched to LOW priority stream: 0.447 ms
elapsed time of kernels launched to HI  priority stream: 0.365 ms
```

The two event durations are from one upstream scheduling scenario, with no
torch baseline, repeated-trial protocol, or controlled contention study. They
must not be interpreted as a general stream-priority speedup claim.
