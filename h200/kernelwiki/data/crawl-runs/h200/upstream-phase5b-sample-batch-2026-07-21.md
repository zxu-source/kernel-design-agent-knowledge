# H200 upstream source batch: shared memory and reduction primitives

All files below are verbatim captures from
`mirrors/cuda-samples@b7c5481c556c3fe98db060207ecaa41a4b9a9abc`, built with
`nvcc -O3 -arch=sm_90 -I Common` on NVIDIA H200 (SM90).

| Source | H200 outcome | Evidence scope |
|---|---|---|
| `simpleTemplates.cu` | float/32 `Compare OK`; int/64 `Compare OK`; 0 failures | source-defined correctness smoke test |
| `threadFenceReduction.cu` | GPU 0.062298238277 vs CPU 0.062298242003 | source CPU comparison smoke test |
| `shfl_scan.cu` | simple-sum GPU/CPU diff = 0; integral-image checksum 2073600 expected 1920x1080 | source CPU comparison smoke test |

The timings printed by these samples are upstream sample output. They have no
torch baseline or exact KDA contract, so all three are `runnable`, not
benchmarked or KDA-ready.
