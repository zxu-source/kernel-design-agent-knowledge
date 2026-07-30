# PR-592 next validation plan

## Identity and scope

- Repository: `sgl-project/sgl-kernel-npu`
- PR / merge SHA: PR-592 / `8f13e502e5bbbd027f5e677e3a99ba1ab1095fce`
- Operator: custom causal conv1d / GDN path
- Current status: `captured`, `unverified`
- Target hardware: Ascend 910B2C only if the source/build preflight succeeds

PR-592 changes a different custom-op implementation from PR-572 and includes
`op_kernel/arch35`. Its upstream page does not explicitly establish 910B
support. Passing PR-572 does not validate this PR.

## Required stop gates

1. Check out the exact merge SHA in a fresh or explicitly cleaned separate
   server clone; do not reuse PR-572's dirty working tree.
2. Inspect the PR-592 `CMakeLists.txt`, build script, and `arch35` dispatch for
   the requested SoC target before changing source files.
3. Configure for `Ascend910B2C` with the recorded CANN 9.0.0 environment. If
   configuration or compilation fails, save stdout/stderr, the exact command,
   and the failure reason; leave the source page `captured/unverified`.
4. If it builds, prove the imported package and `.so` come from that clone,
   then run a minimal NPU import/registration smoke test.
5. Run only the upstream or PR-specific correctness test against its reference.
   Record output and exit code. Do not benchmark or profile in this round.

## 2026-07-27 execution result

The exact merge SHA was checked out in a separate server worktree on Ascend
910B2C / CANN 9.0.0. No clean or compatibility build produced the extension
library. The evidence archive records three build failures:

1. clean build: GNU C++ rejects global AscendC-only `-h...` flags;
2. compatibility v1: ACL header include-order conflict involving
   `aclmdlRITask`;
3. compatibility v2: CANN linker reports `unknown file type` while building
   the unrelated `mega_chunk_gdn_kernel_preprocess` target.

The compatibility attempts changed only CMake include/flag configuration and
disabled DeepEP; their patches and SHA-256 values are in `evidence/runs/`.
No shared library was produced, so import, correctness, benchmark, and
profiling did not run. The source page remains `captured` / `unverified`.

## Promotion rule

Promote PR-592 only after all five gates pass and evidence is downloaded into
this directory. The maximum permissible result in this round is `validated` /
`runnable`; no performance status may be assigned.
