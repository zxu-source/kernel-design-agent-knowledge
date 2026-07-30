# H200 upstream source run: CUDA Samples transpose

- Source: `mirrors/cuda-samples@b7c5481c556c3fe98db060207ecaa41a4b9a9abc`
- File SHA256: `ba300b6b5f4dbc17ec72bfd3d1ee082508632ec431fe46ba54e6e3ce05292a95`
- Command: `nvcc -O3 -arch=sm_90 -I Common transpose.cu && ./transpose`
- Result: H200 run completed and printed `Test passed` for a 1024×1024 FP32 matrix.

Selected upstream-internal measurements:

- naive: 435.9810 GB/s, 0.01792 ms
- coalesced: 1097.1626 GB/s, 0.00712 ms
- optimized: 2396.5901 GB/s, 0.00326 ms
- fine-grained: 2437.2629 GB/s, 0.00321 ms

These compare variants inside the upstream CUDA Sample only; they are not a
PyTorch, cuBLAS, or KDA task baseline.

## Formal-baseline preflight

On the same H200, `torch.transpose(x).contiguous()` for one 1024×1024 FP32
tensor measured `0.0091536 ms` using 20 warmups and 100 CUDA-event iterations.
The upstream sample's fine-grained variant reported `0.00321 ms`, but its
timing policy is not yet aligned with the torch preflight. This is therefore a
baseline probe only, not a validated speedup or a level promotion.

## Aligned validation

The upstream source defines `NUM_REPS=100` and performs one warmup launch.
Torch was rerun with the identical one-warmup/100-event-iteration policy:
`0.00974496 ms`. The complete upstream `optimized` transpose variant measured
`0.00326 ms` for the same 1024×1024 FP32 contract and passed its correctness
check. This is a 2.99x torch-over-upstream ratio for this exact microbenchmark;
it remains `validated`, not `benchmarked`, because it is not yet a target KDA
task-contract replay.
