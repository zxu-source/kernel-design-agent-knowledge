# PR-592 910B2C validation result

## Verdict

**Not validated.** PR-592 remains `captured` / `unverified`.

The exact merge commit `8f13e502e5bbbd027f5e677e3a99ba1ab1095fce` was
checked out on Ascend 910B2C under CANN 9.0.0. No build produced
`libsgl_kernel_npu.so`; therefore the repository-local import, correctness,
benchmark, and profiling gates were not reached.

## Attempt record

| Attempt | Outcome | Recorded blocker |
| --- | --- | --- |
| Clean build | exit 2 | GNU C++ rejects global AscendC-only `-h...` flags. |
| Compatibility v1 | exit 2 | ACL header include-order conflict (`aclmdlRITask`). |
| Compatibility v2 | exit 2 | CANN linker reports `unknown file type` in unrelated `mega_chunk_gdn_kernel_preprocess`. |

Compatibility patches were confined to the isolated server checkout: removing
the two host-incompatible global flags, adding the torch_npu bundled ACL include
path, and configuring with `BUILD_DEEPEP_MODULE=OFF`. They are provenance for
the failed attempts, not a portability fix or a statement about upstream code.

## Evidence

- Downloaded archive SHA-256:
  `03a0c5f6d6081cb8a2dc7b6157d0b1c3ea27c9238451e5f05be28250cc9604c3`
- Archive: `evidence/pr592-910b-build-failures-20260727.tar.gz`
- Exact logs, exit codes, diffs, patch hashes, and library search result:
  `evidence/runs/` and `evidence/logs/`

This result is limited to the recorded server environment and build route. It
does not show that 910B is unsupported in general, that the PR's operator is
incorrect, or that any upstream performance claim holds locally.
