# Validation status definitions

| Status | Meaning |
|---|---|
| `captured` | Source page or code evidence was archived; no runtime claim. |
| `artifact_complete` | `diff.patch`, key files, and provenance are available; no runtime claim. |
| `runnable` | A target-hardware run completed, but the strict evidence set is not yet complete. |
| `validated` | Exact SHA, build, repository-local import, correct correctness/reference invocation, and complete local evidence are available. |
| `FULL_PASS` | Runner stages returned success. This must not be promoted automatically to `validated`. |
| `build_failed` | Build did not complete in the recorded environment. |
| `test_entrypoint_invalid` | The selected invocation is not the target revision's valid test entrypoint. |
| `correctness_failed` | A valid correctness/reference invocation failed. |
| `multi_gpu_required` | The path requires more NPU devices or a specific distributed topology. |
| `reference_required` | No direct correctness reference is available; build/import smoke or a new reference is required. |
| `not_applicable` | The change is not an independently verifiable operator, for example CI, license, or configuration. |
| `evidence_incomplete` | A run result exists but raw logs, profiles, or other required evidence have not been recovered locally. |
