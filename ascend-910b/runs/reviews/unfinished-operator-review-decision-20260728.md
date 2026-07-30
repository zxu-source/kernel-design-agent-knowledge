# Source-only PR review decision

## Scope

- Reviewed source-only merged PRs: 103
- Evidence inspected: cached GitHub PR title/body, changed paths, and exact
  local-Git merge diff for every record.
- Promotion rule: a PR may receive a complete kernel artifact only when its
  own diff changes a kernel/operator implementation or an eligible direct NPU
  Python implementation path.  Test, CI, documentation, dependency, release,
  and build wiring changes are not promoted.

## Result

No source-only PR in this set changes a direct operator implementation.  No
additional artifact bundle is warranted, and the 103 records remain correctly
classified as source-only/context evidence.

## Breakdown

- 55 documentation, CI, or release-only changes.
- 10 test-only changes.
- 18 build/dependency/configuration-only changes.
- 20 mixed test/CI/build or nonstandard-path changes, manually checked as
  non-implementation changes.

## Highest-risk manual checks

- PR-519 changes CANN path handling, DeepEP documentation, and removes only
  commented-out A5 conditional code from `op_host/CMakeLists.txt`; it does not
  change an operator implementation.
- PR-614 adds an external custom-ops A2 build patch that changes the host
  compiler `-march` option from `native` to `armv8.2-a`; it does not alter an
  operator source file.
- PR-28, PR-53, PR-59, and PR-318 change shell/CMake build selection or
  compilation caching only.

The machine-readable scan is recorded in
`unfinished-operator-review-2026-07-28.json`; its conservative candidate list
was resolved by the manual checks above.
