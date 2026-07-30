# Ascend NPU pilot operator manifest

This is a classification of changed upstream paths for archive navigation. It
is not an implementation, compatibility, or performance ranking.

| Category | Core kernel PRs with complete bundle | Captured/unbundled or context |
| --- | --- | --- |
| MoE | PR-600, PR-604 | PR-603 (unbundled) |
| Communication / DeepEP | PR-573, PR-591, PR-608 | none |
| Quantization | PR-566 | PR-605 (context, unbundled) |
| Other kernel paths | PR-548, PR-549, PR-572, PR-592, PR-598, PR-599 | PR-556 (unbundled) |

PR-572 is validated and runnable on the recorded Ascend 910B2C / CANN 9.0.0
environment. The evidence includes a repository-local library import and a
passing upstream correctness suite, but no benchmark or profiling result. Its
build used a recorded compatibility patch, so this is not a claim that a
pristine checkout builds in every CANN environment.

The other 14 source pages remain `captured` and `unverified`. PR-572 remains
the sole record with an upstream-explicit `target_architecture: ascend-910b`.
PR-592 has a recorded 910B2C build failure and also remains `captured` /
`unverified`; no extension library was produced.
