# First-pass failure taxonomy

The correctness-candidate scheduler classified 93 bundles as 24 `FULL_PASS`,
66 failed/limited, and 3 pre-bootstrap skips. The counts below are execution
gate outcomes, not final claims about upstream kernel correctness.

| Gate status | Count | Interpretation | Next action |
|---|---:|---|---|
| `build` | 13 | Configure succeeded but the extension target did not build. Most entries have only `result.json`, so a specific compiler/linker cause is not yet locally evidenced. | Recover/reproduce build logs before assigning a root cause. |
| `correctness_mismatch` | 23 | The generic test invocation returned nonzero after build/import. This mixes real numeric mismatches with invalid invocation modes. | Inspect each test's intended runner before treating it as an operator mismatch. |
| `correctness_failed` | 4 | Test-stage nonzero outcome. | Preserve logs and distinguish collection/fixture problems from numerical failure. |
| `multi_device_required` | 26 | DeepEP/intranode/low-latency path requires more than one NPU. | Record as single-card environment limitation, not a kernel failure. |
| `prebootstrap_no_cmake` | 3 | Historical target SHA predates the current CMake project layout. | Use revision-appropriate build route or keep as pre-bootstrap. |

## Verified invocation problems

- PR-31 and PR-41 invoked `test_cache_assign.py` through pytest and exited 5
  with no collected tests. This is a test-entrypoint problem, not a numerical
  correctness mismatch.
- PR-38 invoked the DeepEP intranode script through pytest and failed because
  its CLI parameters (`args`, `local_rank`) are not pytest fixtures. It needs
  its documented launcher and likely multiple devices; the generic pytest
  command is not a valid correctness invocation.
- PR-46 has a recorded test-stage failure but no local correctness log in its
  latest evidence attempt. Do not infer a numerical root cause until the log is
  recovered or the exact attempt is reproduced.

## Evidence boundary

No generic failure bucket may be reported as an upstream operator defect until
the exact test entrypoint, target-SHA checkout, and complete stdout/stderr are
available locally.
