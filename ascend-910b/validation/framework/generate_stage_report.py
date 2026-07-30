#!/usr/bin/env python3
"""Build a local, auditable 281-PR validation matrix and stage report."""
from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
VALIDATION = HERE.parent
QUEUE = HERE / "full-execution-queue.json"
STATE = HERE / "batch-state-v5.json"
OUT = VALIDATION / "reports"


def latest_attempt(pr_dir: Path) -> Path | None:
    attempts = sorted((pr_dir / "evidence").glob("attempt-*")) if (pr_dir / "evidence").exists() else []
    return attempts[-1] if attempts else None


def main() -> int:
    queue = json.loads(QUEUE.read_text())["candidates"]
    state = json.loads(STATE.read_text())
    state_by_pr = {}
    for bucket in ("completed", "failed", "skipped"):
        for item in state.get(bucket, []):
            state_by_pr[item["pr"]] = {**item, "state_bucket": bucket}

    OUT.mkdir(exist_ok=True)
    rows, audits = [], []
    for item in sorted(queue, key=lambda x: x["pr"]):
        pr = item["pr"]
        result = state_by_pr.get(pr, {})
        pr_dir = VALIDATION / f"PR-{pr}"
        attempt = latest_attempt(pr_dir)
        attempt_files = [p.relative_to(attempt).as_posix() for p in attempt.rglob("*") if p.is_file()] if attempt else []
        required_logs = {"logs/configure.log", "logs/build.log", "logs/import.log", "logs/correctness.log", "logs/profile.log"}
        profile_archive = any(x.endswith("profile.tar.gz") for x in attempt_files)
        audit = {
            "pr": pr,
            "attempt": attempt.name if attempt else None,
            "attempt_file_count": len(attempt_files),
            "has_result": "result.json" in attempt_files,
            "has_all_core_logs": required_logs.issubset(set(attempt_files)),
            "has_benchmark_data": "outputs/benchmark-walltime.json" in attempt_files,
            "has_profile_marker": any("trace_found" in x for x in attempt_files),
            "has_profile_archive": profile_archive,
        }
        if result.get("status") == "FULL_PASS":
            audits.append(audit)
        status = result.get("status")
        if not status:
            status = item.get("state", "not_started")
        rows.append({
            "pr": pr,
            "merge_sha": item["merge_sha"],
            "queue_category": item.get("validation_scope", item.get("state", "unknown")),
            "test": item.get("test") or "",
            "test_mapping": item.get("test_mapping") or "",
            "status": status,
            "failure_gate": result.get("failure_gate", ""),
            "state_bucket": result.get("state_bucket", "not_run"),
            "attempt": result.get("attempt", ""),
            "started_at": result.get("started_at", ""),
            "finished_at": result.get("finished_at", ""),
            "local_attempt": audit["attempt"] or "",
        })

    status_counts = Counter(row["status"] for row in rows)
    full_audit = {
        "full_pass_count": len(audits),
        "core_logs_complete": sum(x["has_all_core_logs"] for x in audits),
        "benchmark_data_present": sum(x["has_benchmark_data"] for x in audits),
        "profile_archive_present": sum(x["has_profile_archive"] for x in audits),
        "profile_marker_present": sum(x["has_profile_marker"] for x in audits),
        "items": audits,
    }
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    report = {
        "generated_at": now,
        "scope": "sgl-project/sgl-kernel-npu complete implementation bundles",
        "total_bundles": len(rows),
        "status_counts": dict(sorted(status_counts.items())),
        "batch_state_metadata": {k: state.get(k) for k in ("phase", "started_at", "updated_at", "finished_at", "total_processed")},
        "full_pass_evidence_audit": full_audit,
        "limitations": [
            "FULL_PASS is a runner status; it is not promoted to validated unless the required local evidence chain is complete.",
            "No FULL_PASS attempt currently has a local raw profile archive.",
            "160 heuristic probes and 28 reference-required bundles have not started this phase.",
        ],
    }
    (OUT / "final-validation-matrix.json").write_text(json.dumps(rows, indent=2) + "\n")
    with (OUT / "final-validation-matrix.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (OUT / "full-pass-evidence-audit.json").write_text(json.dumps(full_audit, indent=2) + "\n")
    (OUT / "stage-report.json").write_text(json.dumps(report, indent=2) + "\n")
    md = f'''# 910B stage validation report\n\nGenerated: `{now}`\n\n## Scope\n\n- Complete implementation bundles: **{len(rows)}**\n- Correctness-candidate phase: **93 classified**\n- Heuristic probes not started: **{status_counts.get("queued_heuristic_probe", 0)}**\n- Reference-required not started: **{status_counts.get("reference_required", 0)}**\n\n## First-pass results\n\n| Status | Count |\n|---|---:|\n'''
    for key, value in sorted(status_counts.items()):
        md += f"| `{key}` | {value} |\n"
    md += f'''\n## Evidence audit\n\n- Runner `FULL_PASS`: **{len(audits)}**\n- FULL_PASS with all core logs locally: **{full_audit["core_logs_complete"]}**\n- FULL_PASS with local benchmark data: **{full_audit["benchmark_data_present"]}**\n- FULL_PASS with local raw profile archive: **{full_audit["profile_archive_present"]}**\n\n`FULL_PASS` is not automatically promoted to `validated`: local raw profile archives are absent, and most pass attempts contain only `result.json`.\n\n## Next work\n\n1. Preserve and classify current failures; do not relax correctness thresholds.\n2. Recover complete evidence where the remote attempt data still exists.\n3. Verify coverage before running heuristic probes.\n4. Implement references only for reference-required semantic operators.\n'''
    (OUT / "FINAL_910B_VALIDATION_REPORT.md").write_text(md)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
