# H200 upstream source probe: warpAggregatedAtomicsCG

- Source: `mirrors/cuda-samples@b7c5481c556c3fe98db060207ecaa41a4b9a9abc`
- File SHA256: `9a8e7416efefd6fb6c29cc2d2114e62e3b405eb4fbe69deb91cccc6e716d4114`
- Command: `nvcc -O3 -arch=sm_90 -I Common warpAggregatedAtomicsCG.cu && ./warpAggregatedAtomicsCG`
- Result: H200 completed the upstream CPU/GPU check and printed `CPU max matches GPU max` and `Warp Aggregated Atomics PASSED`.

This is an upstream correctness smoke test; it does not compare latency against
an unfused atomic baseline.
