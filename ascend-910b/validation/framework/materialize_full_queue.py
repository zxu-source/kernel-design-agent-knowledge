#!/usr/bin/env python3
"""Turn reviewed source tests and heuristic candidates into one runnable queue.

Heuristic entries are explicitly labelled coverage probes in their remote
result; a passing probe never promotes the PR to validated/runnable.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
INVENTORY = HERE / "inventory-manifest.json"
PROPOSALS = HERE / "test-mapping-proposals.json"
OUT = HERE / "full-execution-queue.json"


def main() -> int:
    inventory = json.loads(INVENTORY.read_text())
    proposals = {x["pr"]: x for x in json.loads(PROPOSALS.read_text())["proposals"]}
    source_mapped = heuristic = reference_required = 0
    for item in inventory["candidates"]:
        if item.get("test"):
            item["test_mapping"] = "source_page_test_path"
            item["validation_scope"] = "correctness_candidate"
            source_mapped += 1
            continue
        proposal = proposals[item["pr"]]
        if proposal["candidate_tests"]:
            top = proposal["candidate_tests"][0]
            item.update({
                "test": top["path"], "test_mapping": "heuristic_name_overlap",
                "mapping_evidence": top, "validation_scope": "coverage_probe",
                "state": "queued_heuristic_probe",
            })
            heuristic += 1
        else:
            item["state"] = "reference_required"
            reference_required += 1
    inventory["purpose"] = "full queue; heuristic passes are coverage probes, not correctness claims"
    OUT.write_text(json.dumps(inventory, indent=2) + "\n")
    print(f"wrote {len(inventory['candidates'])} entries: {source_mapped} source-mapped, {heuristic} heuristic probes, {reference_required} reference-required")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
