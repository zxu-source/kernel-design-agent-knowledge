# PR-431 910B2C compatibility validation, microbenchmark, and profile

## Scope and status

This record concerns `sgl-project/sgl-kernel-npu` PR-431 at merge commit
`a50742f5626dac25815460d96894c20064113a0e`.  It upgrades the PR to the
schema-supported `npu_runnable: runnable` and `verification_level: validated`
**only for the recorded compatibility build** below.  It is not a claim that the pristine
checkout builds, nor a general performance claim.

The pristine-build result remains separately preserved in
`VALIDATION_RESULT_20260728.md`: with `SOC_VERSION=Ascend910B2C`, its build
failed because GCC received AscendC-only flags.  This later experiment used a
fresh detached checkout at the exact commit and a recorded CMake-only
compatibility patch; no operator Python or kernel source was changed.

## Environment and checkout

- Remote compatibility checkout:
  `/inspire/sj-ssd/project/qianghuaxuexi/s26043/npu-kernelwiki-validation/PR-431/compat/repo/sgl-kernel-npu`
- Detached commit verified as: `a50742f5626dac25815460d96894c20064113a0e`
- Device: Ascend 910B2C; `torch.npu.is_available() == True`; device count `1`
- CANN: 9.0.0; torch: `2.11.0+cpu`; torch_npu: `2.11.0.rc4`; GCC: 13.3.0
- Each command sourced `/usr/local/Ascend/ascend-toolkit/set_env.sh` and unset
  `ASCEND_VISIBLE_DEVICES` and `ASCEND_RT_VISIBLE_DEVICES`.

The compatibility patch is
`evidence/compat/runs/pr431-gcc-acl-compat-v2.patch` (SHA-256
`855145689326077ab0f4b3d44a451a6489dbfde3664655d05c324c0bc82adfc7`).
It removes two AscendC-only flags from the host compilation options and adds
the torch_npu ACL include directory.  It changes `CMakeLists.txt` and
`csrc/CMakeLists.txt` only.

## Build and correctness

All listed exit-code files contain `0`:

- Configure: `evidence/compat/logs/cmake-config-pr431-compat-v2-20260728.log`
- Build target `sgl_kernel_npu` with `-j2`:
  `evidence/compat/logs/build-pr431-compat-v2-20260728.log`
- Repository-local import and upstream test:
  `PYTHONPATH=python/sgl_kernel_npu python tests/python/sgl_kernel_npu/test_rmsnorm_without_weight.py`
  - Result: `Passed!`
  - Test: `tests/python/sgl_kernel_npu/test_rmsnorm_without_weight.py`
  - Log: `evidence/compat/logs/upstream-correctness-pr431-20260728.log`

The test exercises the repository-local extension and compares the fused
RMSNorm-without-weight path to its PyTorch reference.

## Microbenchmark

`evidence/scripts/benchmark_rmsnorm.py` ran 20 warm-up iterations and 100
NPU-event-timed iterations per implementation and shape, after a correctness
assertion.  Values are medians in microseconds; `baseline/fused` below is less
than one when the fused candidate takes longer.

| Case | Shape / dtype | PyTorch reference | PR-431 fused | baseline / fused |
| --- | --- | ---: | ---: | ---: |
| upstream | 1x130x2048, float32 | 45.57 | 66.28 | 0.688 |
| seq512 | 1x512x2048, float32 | 53.83 | 64.75 | 0.831 |
| hidden4096 | 1x128x4096, float32 | 53.16 | 65.71 | 0.809 |

Raw CSV and JSON are in `evidence/compat/outputs/`.  On these three measured
910B2C/CANN-9.0.0 float32 cases, the fused candidate did not show a speedup;
this limited microbenchmark must not be generalized to other shapes, dtypes,
or end-to-end workloads.

## Profiling

The successful scheduled trace used three active NPU steps for shape
`[1, 130, 2048]`, float32.  Its archive is
`evidence/compat/profile/torch_npu_v3-20260728.tar.gz` (SHA-256
`3f06aa3fbcbfa635d3ddea2d91f0e11b0ac3b034beeda6feafbc15e47d3ee267`).
It contains CANN-parsed `trace_view.json`, `kernel_details.csv`,
`operator_details.csv`, `step_trace_time.csv`, and profiler databases.

`kernel_details.csv` records three fused-kernel device durations of 6.560 us,
3.361 us, and 3.480 us.  These instrumented trace values are not substituted
for the benchmark timings.  The profiler warned that zero warmup can skew
profile results; that warning and the complete stdout/stderr are retained in
`evidence/compat/logs/profile-pr431-v3-retry-20260728.log`.

The first v3 invocation exited `1` before profiling because its required
`PR431_PROFILE_DIR` environment variable was omitted.  Its log and exit code
are retained alongside the successful retry; it generated no trace output.

## Evidence boundaries

No benchmark/profiler result is promoted to an upstream or end-to-end
performance conclusion.  PR-431's original upstream description contains an
external performance statement, but it is not used as validation evidence in
this record.
