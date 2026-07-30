# FP16 Rowwise Softmax Warp-Shuffle Reduction on H200

Two hand-written CUDA C++ kernels for row-wise FP16 softmax:

- **candidate_00_baseline**: Shared-memory tree reduction (one block per 4 rows)
- **candidate_01_warp_shuffle**: Hierarchical warp-shuffle reduction via
  `__shfl_xor_sync` butterfly + shared-memory cross-warp reduction

## Build

```bash
/usr/local/cuda/bin/nvcc -std=c++17 -arch=sm_90 -Iinclude \
  src/rowwise_softmax.cu tests/rowwise_softmax_test.cu \
  -o rowwise_softmax_test
```

## Validate (correctness only)

```bash
CUDA_VISIBLE_DEVICES=0 ./rowwise_softmax_test
```

## Benchmark (correctness + CUDA-event timing)

```bash
CUDA_VISIBLE_DEVICES=0 ./rowwise_softmax_test --bench
```

Results in `benchmark.csv`.

## Remote H200 execution

From workspace root:
```bash
bash scripts/h200-raw.sh "cd kda-workspace/rowwise_softmax_warp_reduce_h200_upload/rowwise_softmax_warp_reduce_h200 && ..."
```

## Torch benchmark

```bash
python3 benchmark.py
```
