#!/usr/bin/env python3
"""Generate a full 281-bundle queue without scheduling remote work."""
from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
CORPUS = HERE.parents[3] / "npu-kernelwiki"
PAGES = CORPUS / "sources/prs/sgl-kernel-npu"
BUNDLES = CORPUS / "artifacts/prs/sgl-kernel-npu"
OUT = HERE / "inventory-manifest.json"


def main() -> int:
    candidates = []
    for bundle in sorted(BUNDLES.glob("PR-*"), key=lambda p: int(p.name[3:])):
        if not (bundle / "diff.patch").is_file() or not (bundle / "PROVENANCE.yaml").is_file():
            continue
        if not any(p.is_file() for p in (bundle / "key-files").rglob("*")):
            continue
        pr = int(bundle.name[3:])
        page = (PAGES / f"PR-{pr}.md").read_text()
        match = re.search(r'^merge_sha: "([0-9a-f]{40})"$', page, re.M)
        if not match:
            raise ValueError(f"PR-{pr}: no merge SHA in source page")
        tests = re.findall(r'"(tests/(?:python/)?[^" ]*test_[A-Za-z0-9_]+\.py)"', page)
        if not tests:
            tests = re.findall(r'(?<![\w/])(tests/(?:python/)?[^\s`:)]+?test_[A-Za-z0-9_]+\.py)', page)
        tests = list(dict.fromkeys(tests))
        candidates.append({
            "pr": pr,
            "merge_sha": match.group(1),
            "test": tests[0] if tests else None,
            "test_mapping": "source_page_test_path" if tests else None,
            "compatibility": [],
            "state": "queued" if tests else "needs_reviewed_test_mapping",
            "artifact": str(bundle.relative_to(CORPUS)),
        })
    result = {
        "schema_version": 1,
        "repo": "sgl-project/sgl-kernel-npu",
        "remote_root": "/inspire/sj-ssd/project/qianghuaxuexi/s26043/npu-kernelwiki-validation/framework-runs-all",
        "defaults": {"benchmark_repeats": 3, "profile_mode": "pytest"},
        "purpose": "full queue; entries without a test map to a recorded preflight result only",
        "candidates": candidates,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {len(candidates)} complete bundles to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
