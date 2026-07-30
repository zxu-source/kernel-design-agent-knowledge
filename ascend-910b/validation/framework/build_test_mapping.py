#!/usr/bin/env python3
"""Build auditable test-mapping proposals for bundles without changed tests.

This is deliberately a proposal generator, not a test oracle.  A matching test
name can locate a useful starting point but cannot prove that it covers every
changed kernel branch.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
CORPUS = HERE.parents[3] / "npu-kernelwiki"
PAGES = CORPUS / "sources/prs/sgl-kernel-npu"
INVENTORY = HERE / "inventory-manifest.json"
OUT = HERE / "test-mapping-proposals.json"
REFERENCE_OUT = HERE / "reference-test-plan.json"

STOP = {
    "python", "sgl", "kernel", "npu", "test", "tests", "utils", "util",
    "src", "csrc", "include", "ascend", "torch", "triton", "module", "ops",
    "op", "common", "base", "impl", "extension", "extensions", "bindings",
}


def tokens(value: str) -> set[str]:
    return {x.lower() for x in re.findall(r"[A-Za-z][A-Za-z0-9]+", value)
            if len(x) >= 3 and x.lower() not in STOP}


def changed_paths(page: str) -> list[str]:
    block = re.search(r"^changed_paths:\n(?P<body>(?:^  - .+\n)+)", page, re.M)
    return re.findall(r'"([^"]+)"', block.group("body")) if block else []


def main() -> int:
    inventory = json.loads(INVENTORY.read_text())
    known_tests = sorted({x["test"] for x in inventory["candidates"] if x.get("test")})
    test_tokens = {test: tokens(Path(test).stem) for test in known_tests}
    proposals, reference_plan = [], []
    for item in inventory["candidates"]:
        if item.get("test"):
            continue
        page = (PAGES / f"PR-{item['pr']}.md").read_text()
        paths = changed_paths(page)
        signal = set().union(*(tokens(path) for path in paths)) if paths else set()
        scored = []
        for test, candidate_tokens in test_tokens.items():
            overlap = sorted(signal & candidate_tokens)
            if overlap:
                scored.append({"path": test, "score": len(overlap), "shared_tokens": overlap})
        scored.sort(key=lambda x: (-x["score"], x["path"]))
        proposal = {
            "pr": item["pr"], "merge_sha": item["merge_sha"],
            "changed_paths": paths, "signal_tokens": sorted(signal),
            "candidate_tests": scored[:5],
            "decision": "review_required",
            "reason": "name overlap is a discovery hint, not coverage proof",
        }
        proposals.append(proposal)
        reference_plan.append({
            "pr": item["pr"], "merge_sha": item["merge_sha"],
            "changed_paths": paths,
            "reference_required_if_no_candidate_is_coverage_valid": True,
            "required_fields": ["repository_local_import", "input_shapes", "dtype", "seed", "reference_implementation", "tolerance"],
        })
    OUT.write_text(json.dumps({"method": "changed-path basename token overlap", "proposals": proposals}, indent=2) + "\n")
    REFERENCE_OUT.write_text(json.dumps({"purpose": "per-PR reference-test requirements", "items": reference_plan}, indent=2) + "\n")
    with_candidates = sum(bool(x["candidate_tests"]) for x in proposals)
    print(f"wrote {len(proposals)} review-required mappings; {with_candidates} have one or more name-overlap hints")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
