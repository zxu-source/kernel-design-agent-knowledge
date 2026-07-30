# BF16 Ragged Grouped GEMM H200 experiment

Build and validate on the H200 execution-side directory:

```bash
cd /inspire/hdd/project/qianghuaxuexi/wangtongyu-25057/kda-workspace/ragged_grouped_gemm_h200_upload/ragged_grouped_gemm_h200
/usr/local/cuda/bin/nvcc -std=c++17 -arch=sm_90 -Iinclude src/ragged_grouped_gemm.cu tests/ragged_grouped_gemm_test.cu -o ragged_grouped_gemm_test
CUDA_VISIBLE_DEVICES=0 ./ragged_grouped_gemm_test
CUDA_VISIBLE_DEVICES=0 ./ragged_grouped_gemm_test --bench
```

The project is intentionally self-contained because the current remote shell
does not expose the formerly documented conda/PyTorch environment. `--bench`
performs 20 warmups, 100 CUDA-event iterations, and three trials per candidate.
