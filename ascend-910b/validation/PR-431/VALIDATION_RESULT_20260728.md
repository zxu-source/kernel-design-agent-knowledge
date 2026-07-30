# PR-431 910B validation result

## Verdict

`captured/unverified` remains unchanged. PR-431 is **not runnable** and is **not validated** on this recorded server environment.

The exact repository checkout and 910B2C environment gates succeeded, but the required repository-local extension library did not build. Consequently, the repository-local import and upstream correctness test were not run. No benchmark or profiling command was run.

## Exact target and environment

- Repository: `sgl-project/sgl-kernel-npu`
- PR: `431`
- Commit: `a50742f5626dac25815460d96894c20064113a0e`
- Remote clone: `/inspire/sj-ssd/project/qianghuaxuexi/s26043/npu-kernelwiki-validation/PR-431/repo/sgl-kernel-npu`
- Device: Ascend 910B2C; `torch.npu.is_available() == True`; device count `1`
- CANN: `/usr/local/Ascend/cann-9.0.0`
- torch / torch_npu / Triton: `2.11.0+cpu` / `2.11.0.rc4` / `3.2.0`
- Each remote command sourced `set_env.sh` and unset `ASCEND_VISIBLE_DEVICES` and `ASCEND_RT_VISIBLE_DEVICES`.

## Commands and results

1. Native configure without a SoC argument: exit `1`.
   - Command: `cmake -S . -B build -DBUILD_DEEPEP_MODULE=OFF`
   - Failure: CANN's `host_config.cmake` rejected empty `SOC_VERSION`.

2. Native configure with the server-derived 910B2C value: exit `0`.
   - Command: `cmake -S . -B build -DBUILD_DEEPEP_MODULE=OFF -DSOC_VERSION=Ascend910B2C -DASCEND_INCLUDE_DIR=/usr/local/Ascend/cann-9.0.0/include`

3. Native build: exit `2`.
   - Command: `cmake --build build --target sgl_kernel_npu -j2`
   - Failure: the generated host-stub compilation reaches GNU C++ and fails with `c++: error: unrecognized command-line option ‘-h’`.
   - Result: `python/sgl_kernel_npu/sgl_kernel_npu/lib/libsgl_kernel_npu.so` is absent.

## Correctness status

The upstream test remains unexecuted:

`tests/python/sgl_kernel_npu/test_rmsnorm_without_weight.py`

It imports `sgl_kernel_npu`, whose package initializer calls `torch.ops.load_library` on the absent repository-local `libsgl_kernel_npu.so`. Bypassing that initializer would not satisfy the required repository-local import gate, so no synthetic direct-kernel test was substituted.

## Evidence

- `evidence/logs/cmake-config-pr431-20260728.log`
- `evidence/logs/cmake-config-pr431-soc-20260728.log`
- `evidence/logs/build-pr431-native-20260728.log`
- `evidence/logs/build-pr431-native-rerun-20260728.log`
- `evidence/runs/*-exit-code.txt`

The first long build invocation was interrupted at the client WebSocket layer before it could write its exit-code file. Its complete server-side log is retained. The subsequent native incremental build was run only to capture a definitive exit code; it also failed with exit `2`.

## Stop condition

No source patch or compatibility experiment was applied. A later compatibility experiment must be separately authorized and must retain this native-build failure as a distinct record; it cannot upgrade this PR without a fresh exact-checkout build, repository-local import, and passing upstream reference correctness test.
