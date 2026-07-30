# H200 upstream source build and run: CUDA Samples phase 3

All files and dependencies were captured from `mirrors/cuda-samples` at
`b7c5481c556c3fe98db060207ecaa41a4b9a9abc`, then compiled with
`nvcc -O3 -arch=sm_90` on NVIDIA H200.

- `simpleVoteIntrinsics`: ran all three Vote.Any / Vote.All tests and printed `OK`.
- `cudaTensorCoreGemm`: ran the upstream high-performance kernel at
  M=N=K=4096 and reported 1.172192 ms / 117.25 TFLOPS. No comparison baseline
  was run, so this is a smoke run, not a speedup claim.
- `tf32TensorCoreGemm`: ran at M=N=8192, K=4096 and reported 21.112703 ms /
  26.04 TFLOPS. No comparison baseline was run, so this is a smoke run.
- `simpleAtomicIntrinsics`: source compiled but failed to link because its
  upstream `computeGold` implementation was not included in the one-file
  capture. It remains blocked rather than being marked runnable.
