# H200 upstream source probe: vectorAdd

- Source: `mirrors/cuda-samples@b7c5481c556c3fe98db060207ecaa41a4b9a9abc`
- File SHA256: `38c186374058ad214541aa06bf0595df61c94b2579a3ac35d21ce32016f9f369`
- Command: `nvcc -O3 -arch=sm_90 -I Common vectorAdd.cu && ./vectorAdd`
- Result: ran on NVIDIA H200. The upstream test processed 50,000 elements and printed `Test PASSED`.

This is a correctness smoke test only; no latency baseline was captured.
