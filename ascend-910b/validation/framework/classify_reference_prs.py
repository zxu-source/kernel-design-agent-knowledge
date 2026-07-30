#!/usr/bin/env python3
"""Classify the 28 reference_required PRs and generate per-PR reference-test
design that must be implemented before any build/correctness/benchmark run.

Categories:
- version_config_only: no kernel code changed — build-smoke only
- formatting_only: clang-format / style — build-smoke only
- license_only: license header changes — build + import smoke
- pure_python_fla: Python-only FLA changes — needs full build (sgl_kernel_npu import depends on .so), then reference test
- mla_ascendc: MLA preprocess AscendC changes — needs build + tensor reference
- zbccl_skeleton: ZBCCL contrib skeleton — separate contrib path, not part of sgl_kernel_npu build
- torch_memory_saver: contrib mem saver — needs separate contrib build
"""
from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
CORPUS = HERE.parents[3] / "npu-kernelwiki"
PAGES = CORPUS / "sources/prs/sgl-kernel-npu"

QUEUE = HERE / "full-execution-queue.json"
OUT = HERE / "reference-test-classification-v2.json"


def classify(paths: list[str], title: str) -> str:
    """Return the classification category for a PR based on changed files and title."""
    path_str = " ".join(paths)

    # Version bumps: only config.ini
    non_config = [p for p in paths if p not in ("config.ini", "scripts/npu_ci_install_dependency.sh")]
    if not non_config:
        return "version_config_only"

    # Clang-format
    if all(".clang-format" in p for p in paths):
        return "formatting_only"
    if all(p.endswith(".clang-format") or "helloworld" in p or "version.h" in p for p in paths):
        return "formatting_only"

    # License only
    if all("license" in p.lower() or "License" in p for p in [" ".join(paths)]):
        return "license_only"
    if all("utils/" in p and any(x in p for x in ("common.h", "defines.h", "torch_helper.h", "version.h")) for p in paths):
        return "license_only"

    # ZBCCL
    if any("zbccl" in p for p in paths):
        return "zbccl_skeleton"

    # Torch memory saver
    if any("torch_memory_saver" in p for p in paths):
        return "torch_memory_saver_contrib"

    # MLA preprocess (AscendC)
    if any("mla_preprocess" in p for p in paths):
        return "mla_ascendc"

    # Pure Python FLA / spec / other
    if all(p.endswith(".py") or "README" in p or ".md" in p for p in paths):
        return "pure_python_requires_build"

    # C++ extensions
    if any(p.endswith((".cpp", ".h", ".hpp", ".cmake")) or "CMakeLists.txt" in p for p in paths):
        return "cpp_extension"

    return "unclassified"


def design_reference_test(pr: int, category: str, key_files: list[str], paths: list[str]) -> dict:
    """Return the reference-test design for this PR."""
    base = {
        "pr": pr,
        "category": category,
        "changed_paths": paths,
        "key_files": key_files,
    }

    if category == "version_config_only":
        base.update({
            "verification_strategy": "build_smoke_only",
            "description": (
                "Version/config-only change. Verify that the build at this SHA "
                "succeeds and repository-local import works. No operator-level "
                "correctness test needed."
            ),
            "gates": ["checkout", "configure", "build", "import"],
            "can_skip_correctness": True,
        })

    elif category == "formatting_only":
        base.update({
            "verification_strategy": "build_smoke_only",
            "description": (
                "Formatting-only change. Verify the build still succeeds, "
                "then mark as build-verified without correctness gate."
            ),
            "gates": ["checkout", "configure", "build", "import"],
            "can_skip_correctness": True,
        })

    elif category == "license_only":
        base.update({
            "verification_strategy": "build_smoke_only",
            "description": (
                "License-header-only change. Build + import smoke test."
            ),
            "gates": ["checkout", "configure", "build", "import"],
            "can_skip_correctness": True,
        })

    elif category == "pure_python_requires_build":
        base.update({
            "verification_strategy": "reference_test_required",
            "description": (
                "Pure Python change, but sgl_kernel_npu import requires the "
                "C++ extension library. Full build + then repository-local "
                "import correctness against a PyTorch/NumPy reference."
            ),
            "reference_design": {
                "repository_local_import": "sgl_kernel_npu (via PYTHONPATH=python/sgl_kernel_npu)",
                "input_shapes": "to be derived from key file's function signatures",
                "dtype": "to be derived from key file",
                "seed": 42,
                "reference_implementation": "PyTorch native ops or NumPy",
                "tolerance": {"float32": "1e-5", "float16": "1e-3", "bfloat16": "1e-2"},
            },
            "gates": ["checkout", "configure", "build", "import", "reference_correctness"],
            "can_skip_correctness": False,
        })

    elif category == "mla_ascendc":
        base.update({
            "verification_strategy": "reference_test_required",
            "description": (
                "MLA preprocess AscendC kernel change. Requires full build "
                "and a reference test comparing operator output against "
                "a CPU reference implementation. The operator likely has "
                "an existing test somewhere in the repo from later PRs."
            ),
            "reference_design": {
                "repository_local_import": "sgl_kernel_npu (via PYTHONPATH=python/sgl_kernel_npu)",
                "input_shapes": "MLA preprocess shapes: (bsz, seq_len, hidden_dim) variant",
                "dtype": "float16 (primary), bfloat16 (if NZ format supported)",
                "seed": 42,
                "reference_implementation": "PyTorch manual MLA preprocessing on CPU",
                "tolerance": {"float16": "1e-3", "bfloat16": "1e-2"},
            },
            "gates": ["checkout", "configure", "build", "import", "reference_correctness"],
            "can_skip_correctness": False,
        })

    elif category == "zbccl_skeleton":
        base.update({
            "verification_strategy": "contrib_build_only",
            "description": (
                "ZBCCL is a contrib/ subproject with its own build system "
                "(not part of sgl_kernel_npu target). Verify the ZBCCL "
                "CMake configures independently. Full integration test "
                "requires multi-device environment — out of scope."
            ),
            "gates": ["zbccl_contrib_configure"],
            "can_skip_correctness": True,
            "note": "ZBCCL contrib is not built by sgl_kernel_npu target; separate verification needed",
        })

    elif category == "torch_memory_saver_contrib":
        base.update({
            "verification_strategy": "contrib_build_smoke",
            "description": (
                "torch_memory_saver is a contrib/ subproject. Verify its "
                "C++ extension builds and imports. Operator-level correctness "
                "requires a full workload."
            ),
            "gates": ["checkout", "contrib_build", "contrib_import"],
            "can_skip_correctness": True,
        })

    elif category == "cpp_extension":
        base.update({
            "verification_strategy": "reference_test_required",
            "description": (
                "C++ extension change requiring full build. Design reference "
                "test against known-good operator output."
            ),
            "reference_design": {
                "repository_local_import": "sgl_kernel_npu (via PYTHONPATH=python/sgl_kernel_npu)",
                "input_shapes": "to be derived",
                "dtype": "float16",
                "seed": 42,
                "reference_implementation": "PyTorch equivalent",
                "tolerance": {"float16": "1e-3"},
            },
            "gates": ["checkout", "configure", "build", "import", "reference_correctness"],
            "can_skip_correctness": False,
        })

    else:
        base.update({
            "verification_strategy": "review_required",
            "description": "Unclassified — needs manual review before any test design.",
            "gates": ["manual_review"],
            "can_skip_correctness": True,
        })

    return base


def main() -> int:
    queue = json.loads(QUEUE.read_text())
    refs = [x for x in queue["candidates"] if x.get("state") == "reference_required"]

    classified = []
    category_counts: dict[str, int] = {}

    for item in refs:
        pr = int(item["pr"])
        sha = str(item["merge_sha"])

        page_file = PAGES / f"PR-{pr}.md"
        page = page_file.read_text()

        # Get title
        title_m = re.search(r'^title: "(.+)"', page, re.M)
        title = title_m.group(1) if title_m else "unknown"

        # Get changed paths
        block = re.search(r"^changed_paths:\n(?P<body>(?:^  - .+\n)+)", page, re.M)
        paths = re.findall(r'"([^"]+)"', block.group("body")) if block else []

        # Get key files in bundle
        bundle = CORPUS / str(item["artifact"]) / "key-files"
        key_files = [str(p.relative_to(bundle))
                     for p in sorted(bundle.rglob("*")) if p.is_file()] if bundle.exists() else []

        category = classify(paths, title)
        design = design_reference_test(pr, category, key_files, paths)
        design["title"] = title
        design["merge_sha"] = sha
        classified.append(design)
        category_counts[category] = category_counts.get(category, 0) + 1

    result = {
        "schema_version": 2,
        "total_reference_required": len(refs),
        "category_counts": category_counts,
        "categories_explained": {
            "version_config_only": "config.ini version bumps — build smoke only",
            "formatting_only": "clang-format changes — build smoke only",
            "license_only": "license header changes — build + import smoke",
            "pure_python_requires_build": "pure Python but .so required for import — full build + reference test",
            "mla_ascendc": "MLA preprocess AscendC changes — full build + tensor reference",
            "zbccl_skeleton": "ZBCCL contrib skeleton — separate contrib build",
            "torch_memory_saver_contrib": "torch_memory_saver contrib — contrib build + import",
            "cpp_extension": "C++ extension — full build + reference test",
        },
        "items": classified,
    }

    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Classified {len(refs)} reference_required PRs: {category_counts}")
    print(f"Output: {OUT}")

    # Print summary table
    print("\n=== Category breakdown ===")
    can_skip = [x for x in classified if x.get("can_skip_correctness")]
    need_ref = [x for x in classified if not x.get("can_skip_correctness")]
    print(f"Can skip correctness (build/import smoke): {len(can_skip)}")
    for c in can_skip:
        print(f"  PR-{c['pr']}: {c['category']} — {c['title'][:60]}")
    print(f"\nNeed reference test: {len(need_ref)}")
    for c in need_ref:
        print(f"  PR-{c['pr']}: {c['category']} — {c['title'][:60]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
