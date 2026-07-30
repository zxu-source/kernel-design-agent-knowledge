# H200 upstream source run: binaryPartitionCG

- Source: `mirrors/cuda-samples@b7c5481c556c3fe98db060207ecaa41a4b9a9abc`
- File: `cpp/3_CUDA_Features/binaryPartitionCG/binaryPartitionCG.cu`
- SHA256: `5a3be526cdd535a8c2cf2cb740105e4b4f46d7aa93ef966f9543723a460a1457`
- Build: `nvcc -O3 -arch=sm_90 -I Common binaryPartitionCG.cu`

The unmodified upstream source compiled and completed on NVIDIA H200 (SM90).
It launched 264 blocks of 1024 threads and printed the odd/even partition
counts and sums below. The source has no host-side expected-value comparison
or latency timing, so this is runnability evidence only, not a correctness or
performance promotion.

```text
GPU Device 0: "Hopper" with compute capability 9.0

Launching 264 blocks with 1024 threads...

Array size = 102400 Num of Odds = 50945 Sum of Odds = 1272565 Sum of Evens 1233938

...Done.
```
