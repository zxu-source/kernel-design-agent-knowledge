# H200 upstream source run: simpleAtomicIntrinsics

The prior one-file probe lacked the upstream companion
`simpleAtomicIntrinsics_cpu.cpp`. Both source files and Common headers were
captured from the same Gitee commit, then built with `nvcc -O3 -arch=sm_90`.

```text
Processing time: 1.111000 (ms)
simpleAtomicIntrinsics completed, returned OK
```

This is an upstream correctness smoke run, not a comparative benchmark.
