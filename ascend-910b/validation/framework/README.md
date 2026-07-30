# 910B PR validation framework

`manifest.json` is the only per-PR input.  Each entry declares the exact merge
SHA, an upstream correctness test, and any recorded compatibility layer.

Run locally:

```bash
python3 run_ascend_validation.py
python3 run_ascend_validation.py --execute --pr 557
```

To create the full 281-bundle review inventory (this never starts a remote
job), run:

```bash
python3 generate_inventory_manifest.py
```

The generated queue maps the 93 bundles whose archived source page names a
test. The remaining 188 have no recorded test path. During a full execution,
those 188 produce a local `queue-results/PR-<N>.json` with
`needs_test_mapping` and do not run remotely. This is deliberate: a generic
test cannot prove arbitrary kernel correctness.

Run the full queue with:

```bash
python3 generate_inventory_manifest.py
python3 run_ascend_validation.py --manifest inventory-manifest.json --execute
```

Before executing the 188 unmapped items, generate the mapping and reference
plans:

```bash
python3 build_test_mapping.py
```

This writes `test-mapping-proposals.json` (existing-test discovery hints) and
`reference-test-plan.json` (the required fields for a new reference test).
The materialized all-PR queue keeps name-based proposals visibly separate from
source-page test mappings; it executes them only as coverage probes.

For a single full-queue submission, use:

```bash
bash run_all_281.sh
```

It runs 93 source-mapped correctness candidates and 160 heuristic coverage
probes through the same build/test/benchmark/profile gates.  Their
`result.json` files retain `test_mapping` and `validation_scope`, so a passing
heuristic probe is never misreported as correctness validation. The final 28
entries have no test candidate and are recorded as `needs_test_mapping` until
a repository-local reference test is implemented.

The runner creates an isolated remote directory per PR and records checkout,
build, correctness, benchmark, and profile exit codes in `result.json`.
Benchmark is only upstream-test wall time; it is not an operator-performance
claim.  After all three repeated test runs pass, `profile_mode: pytest` runs
the same upstream test under `torch_npu.profiler` and saves its trace in
`profile/`. A PR that fails build, correctness, or the repeated test gate never
continues to profiling, and the framework never changes source-page validation
status automatically.

Set `profile_mode: none` on a manifest entry to suppress profiling.  A profile
failure is recorded separately and is not silently treated as a correctness
failure or a successful trace.
