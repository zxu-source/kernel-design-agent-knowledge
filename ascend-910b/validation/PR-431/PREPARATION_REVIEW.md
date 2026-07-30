# PR-431 phase-1 preparation review

## Scope and status

- Repository: `sgl-project/sgl-kernel-npu`
- PR: `431`
- Merge SHA: `a50742f5626dac25815460d96894c20064113a0e`
- Current status: `captured/unverified`
- This record contains only connectivity, environment, source synchronization, and candidate review evidence. No candidate build, import, JIT compilation, correctness execution, benchmark, or profiling was run.

## Candidate basis

PR-431 adds a Triton fused RMSNorm-without-weight implementation:

- key file: `python/sgl_kernel_npu/sgl_kernel_npu/norm/rmsnorm_without_weight.py`
- upstream correctness test: `tests/python/sgl_kernel_npu/test_rmsnorm_without_weight.py`
- reference in that test: `torch.nn.functional.rms_norm`

The candidate changes one independent Python/Triton operator and includes its own direct correctness test. It avoids PR-592's multi-file AscendC/CMake extension build chain. Its source metadata records `target_architecture: unknown` and `cann_version: unknown`; no explicit 910B, SoC, or CANN compatibility claim was found in the archived PR evidence.

## Gate and synchronization

The 910B gate passed after CANN setup and unsetting both Ascend visible-device variables. The remote NPU reported `Ascend910B2C`, and `torch.npu.is_available()` was `True` with one logical device. See `evidence/logs/phase1-environment-gate-20260728.log`.

The isolated remote root is `/inspire/sj-ssd/project/qianghuaxuexi/s26043/npu-kernelwiki-validation/PR-431`. A fresh clone there is detached at the exact merge SHA and clean. The source page, `diff.patch`, `PROVENANCE.yaml`, and key file were copied to its `metadata/` subtree and matched local SHA-256 values. See `evidence/logs/phase1-sync-integrity-20260728.log`.

## Phase-2 plan and stop conditions

Only after explicit authorization: load CANN, unset both visible-device variables, reconfirm the clone SHA and clean status, use the repository-local Python package, then run the upstream correctness test against its `F.rms_norm` reference. Capture complete stdout/stderr, commands, exit code, environment, and resulting status here.

Stop without status promotion if the environment gate changes, the clone SHA or source/provenance SHA differs, repository-local import fails, Triton-on-Ascend compilation fails, or the reference comparison fails. A passing test would support only runnable/validated correctness evidence for this exact server environment; it would not support any performance claim. No benchmark or profiling is authorized.
