# H200 upstream source run: CUDA Graph memory nodes

- Source: `mirrors/cuda-samples@b7c5481c556c3fe98db060207ecaa41a4b9a9abc`
- File: `cpp/3_CUDA_Features/graphMemoryNodes/graphMemoryNodes.cu`
- SHA256: `9382875a4a36016b3586af08375961daff8e05925fe836e9bd5fe0854a3caf42`
- Build: `nvcc -O3 -arch=sm_90 -I Common graphMemoryNodes.cu`

The unmodified upstream source compiled and completed on NVIDIA H200 (SM90).
Its built-in validation passed for all six paths: plain stream, stream-captured
graph, explicitly constructed graph, and three graph/stream free variants.
The explicit-graph and graph-external-free paths also printed that
`d_negSquare` and `d_input` share a virtual address.

```text
Validation PASSED!  # stream
Validation PASSED!  # stream-captured graph
Validation PASSED!  # explicitly constructed graph
Validation PASSED!  # freed outside stream
Validation PASSED!  # freed outside graph
Validation PASSED!  # freed in a different graph
```

This is functional H200 evidence only. The sample provides no controlled
latency baseline, so it is `runnable`, not a benchmark or KDA adoption claim.
