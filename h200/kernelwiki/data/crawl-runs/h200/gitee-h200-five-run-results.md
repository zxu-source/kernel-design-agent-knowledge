# Gitee CUDA Samples — Five H200 Property Runs

Date: 2026-07-20

All five source snapshots were collected from `mirrors/cuda-samples` at
`b7c5481c556c3fe98db060207ecaa41a4b9a9abc`. Each H200 result below comes
from a source-informed, standalone CUDA microbenchmark compiled with
`nvcc -O3 -std=c++17 -arch=sm_90`. Each measurement is the mean of 50 launches
after 10 warmups. These are not claims that the complete CUDA Sample was built
or that its upstream benchmark numbers were reproduced.

| Iteration | Gitee source | Property tested on H200 | Correctness | Baseline / optimized latency | Speedup | Conclusion |
|---|---|---|---|---|---:|---|
| 1 | `cpp/0_Introduction/vectorAdd/vectorAdd.cu` | Elementwise copy with aligned `float4` access versus scalar copy | PASS, 0 mismatches | 0.0538 / 0.0348 ms | 1.545x | Aligned vector access improved this copy microbenchmark. The upstream vectorAdd sample itself is a baseline example, not a vectorization claim. |
| 2 | `cpp/6_Performance/transpose/transpose.cu` | Shared-memory tiled transpose with padded tile versus naive transpose | PASS, 0 mismatches | 0.0710 / 0.0137 ms | 5.182x | The sample's coalescing/bank-conflict-avoidance property is supported. |
| 3 | `cpp/3_CUDA_Features/warpAggregatedAtomicsCG/warpAggregatedAtomicsCG.cu` | One atomic per warp versus one atomic per active thread, same final count | PASS | 0.0990 / 0.0991 ms | 0.999x | The reduction of atomic operations is functionally correct, but did not improve this single-address H200 microbenchmark. |
| 4 | `cpp/3_CUDA_Features/globalToShmemAsyncCopy/globalToShmemAsyncCopy.cu` | `cuda::memcpy_async` global-to-shared-to-global copy versus direct copy | PASS, 0 mismatches | 0.0538 / 0.0562 ms | 0.957x | The async-copy path is correct but slower without useful computation to overlap; no speedup claim. |
| 5 | `cpp/3_CUDA_Features/bf16TensorCoreGemm/bf16TensorCoreGemm.cu` | BF16 WMMA Tensor Core GEMM versus SIMT BF16 GEMM, 512x512 | PASS, 0 mismatches | 0.0709 / 0.0129 ms | 5.509x | The Tensor Core property is supported and materially faster in this microbenchmark. |

GitCode was not used for these five H200 runs: the authenticated
`AI4Science/dgl-ascend` crawl returned only Ascend/NPU PRs, which are not
H200-runnable and were excluded rather than force-fit into this experiment.
