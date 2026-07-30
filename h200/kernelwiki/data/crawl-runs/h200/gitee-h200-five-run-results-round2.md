# Gitee CUDA Samples — Five More H200 Runs

Date: 2026-07-20. Gitee API tree/contents calls were rate-limited with HTTP
403 during this round, so the five files were fetched from the same public
Gitee mirror's `raw/master` endpoint. Each raw file is kept in the matching
`data/crawl-runs/gitee/round2-*` directory with its SHA256. The H200 harness
was compiled with `nvcc -O3 -std=c++17 -arch=sm_90`; timings are 30 launches
after 5 warmups.

| Round | Gitee source | H200 property check | Correctness | Measurement | Relative result |
|---|---|---|---|---|---:|
| 1 | `cpp/0_Introduction/simpleAtomicIntrinsics/simpleAtomicIntrinsics.cu` | `atomicAdd` versus CAS increment under the same counter workload | PASS | add 0.0034 ms; CAS 14.9052 ms | atomicAdd is ~4,384x faster (CAS/add = 0.00023x) |
| 2 | `cpp/0_Introduction/simpleCooperativeGroups/simpleCooperativeGroups.cu` | cooperative-groups tiled warp reduction versus shared reduction | PASS | cooperative 0.3082 ms; shared 0.0113 ms | 0.037x; no improvement in this workload |
| 3 | `cpp/2_Concepts_and_Techniques/shfl_scan/shfl_scan.cu` | warp shuffle scan versus shared scan | PASS | shuffle 0.0037 ms; shared 0.0039 ms | 1.069x |
| 4 | `cpp/3_CUDA_Features/cudaTensorCoreGemm/cudaTensorCoreGemm.cu` | FP16 WMMA Tensor Core GEMM versus SIMT FP16 GEMM, 512x512 | PASS, 0 mismatches | SIMT 0.3242 ms; WMMA 0.0128 ms | 25.328x |
| 5 | `cpp/3_CUDA_Features/tf32TensorCoreGemm/tf32TensorCoreGemm.cu` | TF32 WMMA Tensor Core GEMM versus FP32 SIMT GEMM, 512x512 | PASS, 0 mismatches | SIMT 0.0582 ms; WMMA 0.0190 ms | 3.060x |

These are source-informed H200 property microbenchmarks, not claims that the
complete samples or their upstream benchmark tables were reproduced. The
cooperative-groups result is a useful negative control: functional use of a
cooperative tile did not imply a speedup for this reduction shape.
