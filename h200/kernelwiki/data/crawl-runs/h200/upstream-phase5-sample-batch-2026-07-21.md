# H200 upstream source batch: execution and scheduling primitives

All files below are verbatim captures from
`mirrors/cuda-samples@b7c5481c556c3fe98db060207ecaa41a4b9a9abc`, built with
`nvcc -O3 -arch=sm_90 -I Common` on NVIDIA H200 (SM90).

| Source | H200 outcome | Evidence scope |
|---|---|---|
| `simpleAWBarrier.cu` | `Result = PASSED`, returned OK | source-defined correctness smoke test |
| `simpleHyperQ.cu` | completed; measured 0.060 s for the sample | source scheduling observation only |
| `simpleOccupancy.cu` | `Test PASSED`; manual 0.099232 ms, occupancy-selected 0.01264 ms | source-internal configuration comparison only |
| `simplePrintf.cu` | completed device-side printf output | runnability only |
| `simpleStreams.cu` | completed 100-repetition stream demo | source stream-overlap observation only |
| `simpleZeroCopy.cu` | vectorAdd result check completed | source-defined correctness smoke test |
| `reductionMultiBlockCG.cu` | GPU 1.992401719093 vs CPU 1.992401361465; completed | source CPU comparison smoke test |
| `asyncAPI.cu` | compilation blocked: `cuda_profiler_api.h` absent in remote toolkit include path | captured only; no runnable promotion |

The timing values are upstream sample output. No entry in this batch has a
torch baseline or an exact KDA task contract, so successful entries are
`runnable`, never benchmarked or KDA-ready.
