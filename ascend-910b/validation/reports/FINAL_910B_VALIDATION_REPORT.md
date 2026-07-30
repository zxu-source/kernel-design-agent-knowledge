# 910B stage validation report

Generated: `2026-07-29T01:46:24Z`

## Scope

- Complete implementation bundles: **281**
- Correctness-candidate phase: **93 classified**
- Heuristic probes not started: **160**
- Reference-required not started: **28**

## First-pass results

| Status | Count |
|---|---:|
| `FULL_PASS` | 24 |
| `build` | 13 |
| `correctness_failed` | 4 |
| `correctness_mismatch` | 23 |
| `multi_device_required` | 26 |
| `prebootstrap_no_cmake` | 3 |
| `queued_heuristic_probe` | 160 |
| `reference_required` | 28 |

## Evidence audit

- Runner `FULL_PASS`: **24**
- FULL_PASS with all core logs locally: **3**
- FULL_PASS with local benchmark data: **3**
- FULL_PASS with local raw profile archive: **0**

`FULL_PASS` is not automatically promoted to `validated`: local raw profile archives are absent, and most pass attempts contain only `result.json`.

## Next work

1. Preserve and classify current failures; do not relax correctness thresholds.
2. Recover complete evidence where the remote attempt data still exists.
3. Verify coverage before running heuristic probes.
4. Implement references only for reference-required semantic operators.
