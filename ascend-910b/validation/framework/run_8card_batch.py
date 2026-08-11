#!/usr/bin/env python3
"""8-card batch coverage runner for the sgl-kernel-npu execution inventory.

Local orchestrator for the 281-bundle queue. Each bundle runs as an isolated,
detached, persistent remote job on `ascend_910b_8card` with:
  - exact-SHA checkout, compatibility migration patch (recorded, never claimed
    as upstream-original), configure/build, repository-local import
  - correctness/reference/probe run driven by the manifest's `execution_kind`
    (pytest | python_spawn | torchrun | shell | reference | build_import_only)
  - per-stage state.json on the remote, stage logs streamed to files, a unique
    MASTER_PORT per attempt, and a compact local result.json + 281-row ledger.

Resume-safe: a bundle whose latest local result.json holds a terminal status is
skipped; an interrupted one is re-polled (if still running) or re-dispatched.

Never uses the single-card `910bctl`/`ascend_910b`. Never runs a CLI/spawn
script through pytest.
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
VALIDATION = HERE.parent  # .../validation/ascend-910b
CORPUS = VALIDATION.parents[2]  # .../npu-kernelwiki
CTL = "/home/kirin_14379/projects/ai4qz/scripts/910b8ctl"
AI4QZ = "/home/kirin_14379/projects/ai4qz/.venv/bin/ai4qz"
CONFIG = "/home/kirin_14379/projects/ai4qz/configs/notebooks-910b-8card.yaml"
TARGET = "ascend_910b_8card"
REMOTE_ROOT = "/inspire/sj-ssd/project/qianghuaxuexi/s26043/npu-kernelwiki-validation-8card/batch"
GATE_EVIDENCE = VALIDATION / "multicard-gate" / "20260803T081037Z" / "hccl-smoke.log"
MANIFEST = HERE / "execution-manifest-8card.json"
LEDGER = VALIDATION / "ledger-8card.json"
BATCH_STATE = HERE / "batch-state-8card.json"
REF_CLASSIFICATION = HERE / "reference-test-classification-v2.json"

# Statuses that mean "this bundle is done; skip on resume".
TERMINAL = {
    "validated", "correctness_failed", "runnable", "probe_passed",
    "reference_required", "test_entrypoint_invalid", "build_failed",
    "requires_resolution", "evidence_incomplete", "not_applicable",
}

REMOTE_WORKER = r'''
import json, os, re, signal, socket, subprocess, sys, time, traceback
from pathlib import Path

cfg = json.loads(Path(sys.argv[1]).read_text())
base = Path(cfg["run_dir"])
repo = base / "repo"
logs = base / "logs"
runs = base / "runs"
for d in (logs, runs):
    d.mkdir(parents=True, exist_ok=True)
state = base / "state.json"

started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def _fatal(kind, value, tb):
    import traceback as _tb
    doc = {"pr": cfg.get("pr"), "merge_sha": cfg.get("sha"),
           "status": "runner_exception", "exception": kind.__name__,
           "message": str(value), "updated_at": now()}
    try:
        (base / "runner-exception.log").write_text("".join(_tb.format_exception(kind, value, tb)))
        (base / "result.json").write_text(json.dumps(doc, indent=2) + "\n")
        checkpoint("finished", status="runner_exception")
    except BaseException:
        pass
    sys.__excepthook__(kind, value, tb)

sys.excepthook = _fatal

def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def checkpoint(stage, **extra):
    doc = {
        "pr": cfg.get("pr"), "merge_sha": cfg.get("sha"),
        "target": "ascend_910b_8card", "stage": stage,
        "updated_at": now(), **extra,
    }
    state.write_text(json.dumps(doc, indent=2) + "\n")

def persist(res):
    res.setdefault("updated_at", now())
    (base / "result.json").write_text(json.dumps(res, indent=2) + "\n")
    return res

res = {
    "pr": cfg.get("pr"), "merge_sha": cfg.get("sha"),
    "execution_kind": cfg.get("execution_kind"),
    "target": "ascend_910b_8card", "started_at": started_at,
    "master_port": cfg.get("master_port"),
    "world_size": cfg.get("world_size"),
    "compatibility": cfg.get("compatibility", []),
    "compatibility_migration": cfg.get("compatibility_migration", False),
    "validation_scope": cfg.get("validation_scope"),
    "status": "running",
}

def run(cmd, name, cwd=None, env=None, heartbeat_every=15, start_new_session=False):
    checkpoint(f"{name}_started")
    logfile = logs / name
    p_env = dict(os.environ)
    if env:
        p_env.update(env)
    with logfile.open("w") as f:
        f.write("$ " + (" ".join(cmd) if isinstance(cmd, list) else str(cmd)) + "\n")
        f.flush()
        p = subprocess.Popen(cmd, cwd=cwd or repo, env=p_env, text=True,
                             stdout=f, stderr=subprocess.STDOUT,
                             shell=isinstance(cmd, str),
                             executable="/bin/bash" if isinstance(cmd, str) else None,
                             start_new_session=start_new_session)
        last = time.monotonic()
        while p.poll() is None:
            if time.monotonic() - last >= heartbeat_every:
                checkpoint(f"{name}_running", pid=p.pid)
                last = time.monotonic()
            time.sleep(1)
        f.write(f"\n[exit_code={p.returncode}]\n")
    res[f"{name}_rc"] = p.returncode
    checkpoint(f"{name}_finished", exit_code=p.returncode)
    return p.returncode

def _classify_correctness_failure(text, rc=None):
    if rc == 5:
        return "test_entrypoint_mismatch"
    # DeepEP's aclnn APIs are supplied by the PR-local custom OPP package, not
    # by CANN's stock libopapi.so.  Keep this distinct from topology errors so
    # a missing PR-local build is never misreported as a world-size limitation.
    if re.search(r"aclnn\w+.*not in libopapi|not in libopapi\.so|libcust_opapi", text, re.I):
        return "missing_custom_opapi"
    if re.search(r"do tiling failed|TilingAndUpdateBinInfo|NnopbaseExecutorDoTiling", text, re.I):
        return "custom_op_tiling_failure"
    # A generic launcher string (for example a test's --num-processes help or
    # command echo) must not mask an actual device-side failure emitted later
    # in the same log.  These signatures are observed after PR-local custom
    # OPPs successfully build and import, so they are a separate runtime gate.
    if re.search(r"aicore|AI_CORE|AclrtSynchronizeStream|507014|ERR00100|execute kernel param invalid|fftsplus|device.*timeout|NPU function error|SIGSEGV|ProcessExitedException|Segmentation", text, re.I):
        return "device_or_runtime_error"
    if re.search(r"world size|world_size.*less or equal|new group's world size|nproc|num-processes|num_processes.*(8|16)", text):
        return "launcher_worldsize_mismatch"
    if re.search(r"EADDRINUSE|DistNetworkError|address already in use|listen on any local", text):
        return "dist_init_or_port"
    if re.search(r"HCCL|hcclCommInit|Communication_Error|P2P communication|not on the same plane|ASCEND_VISIBLE", text):
        return "transport_or_env"
    if re.search(r"No module named|ModuleNotFoundError|ImportError|has no attribute|AttributeError", text):
        return "no_module_or_import"
    # undefined name in test harness (e.g. NameError: name 'F' is not defined) —
    # a test-source defect, not a kernel assertion failure.
    if re.search(r"NameError: name '\w+' is not defined", text):
        return "test_call_mismatch"
    # test-vs-API call mismatch (TypeError, unexpected/multiple keyword args, wrong
    # signature) — a test/entrypoint defect, not a kernel assertion failure.
    if re.search(r"TypeError:|multiple values for argument|got an unexpected keyword|takes \d+ positional|missing \d+ required", text):
        return "test_call_mismatch"
    # tensor shape/broadcast mismatch in the test harness — test/API-shape defect.
    if re.search(r"shape mismatch|cannot be broadcast|size mismatch|Sizes of tensors must match|expected.*same size", text):
        return "test_call_mismatch"
    # pytest exit code 5: collection produced no runnable tests (arg-taking test
    # functions are not pytest fixtures) — a test-entrypoint mismatch.
    if re.search(r"no tests ran|no tests collected|No tests ran|collection failed|ERROR: no tests", text):
        return "test_entrypoint_mismatch"
    if re.search(r"AssertionError|assert\b|Mismatch|tensor_equal|not close|allclose", text):
        return "assertion_mismatch"
    return "unknown_nonzero"

# --------------------------------------------------------------------- env snapshot
env_doc = {
    "target": "ascend_910b_8card",
    "preamble": "source /usr/local/Ascend/ascend-toolkit/set_env.sh; unset ASCEND_VISIBLE_DEVICES ASCEND_RT_VISIBLE_DEVICES; HCCL_INTRA_ROCE_ENABLE=1",
    "master_addr": "127.0.0.1",
    "master_port": cfg.get("master_port"),
    "world_size": cfg.get("world_size"),
    "recorded_at": now(),
}
try:
    import torch, torch_npu
    env_doc["torch"] = torch.__version__
    env_doc["torch_npu"] = torch_npu.__version__
    env_doc["npu_available"] = bool(torch.npu.is_available())
    env_doc["npu_count"] = int(torch.npu.device_count())
except Exception as e:
    env_doc["import_error"] = str(e)
import sys as _sys
env_doc["python"] = _sys.version.split()[0]
(base / "environment.log").write_text(json.dumps(env_doc, indent=2) + "\n")

# --------------------------------------------------------------------- checkout
rc = run(["git", "clone", "--no-checkout",
          "https://github.com/sgl-project/sgl-kernel-npu.git", str(repo)],
         "git-clone.log", cwd=base)
if rc == 0:
    rc = run(["git", "fetch", "--depth=1", "origin", cfg["sha"]], "git-fetch.log")
if rc == 0:
    rc = run(["git", "checkout", "--detach", cfg["sha"]], "git-checkout.log")
    if rc == 0:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                              capture_output=True, text=True).stdout.strip()
        res["head"] = head
        if head != cfg["sha"]:
            res["status"] = "evidence_incomplete"
            res["stop_reason"] = "checkout SHA mismatch"
            persist(res); checkpoint("finished", status=res["status"]); sys.exit(0)

# --------------------------------------------------------------------- compat patch (gcc_acl)
# The queue's `compatibility` lists are empty, but the modern GCC toolchain on
# this target rejects the upstream `-hno-unused-parameter -lno-unused-function`
# flags, so the gcc_acl migration is applied to every bundle and recorded as a
# compatibility migration, never claimed as an upstream-original reproduction.
compat = cfg.get("compatibility", [])
if "gcc_acl" in compat or (cfg.get("compatibility_migration") and res.get("head")):
    try:
        cm = repo / "CMakeLists.txt"
        cs = repo / "csrc" / "CMakeLists.txt"
        before_cm = cm.read_text()
        before_cs = cs.read_text() if cs.exists() else ""
        cm.write_text(before_cm.replace("-hno-unused-parameter -lno-unused-function ", ""))
        acl_path = "${TORCH_NPU_DIR}/include/third_party/acl/inc"
        if cs.exists() and acl_path not in before_cs:
            cs.write_text(before_cs.replace(
                "${TORCH_NPU_DIR}/include",
                "${TORCH_NPU_DIR}/include\n        " + acl_path, 1))
        # Plan A: cstdint compatibility shim.  SHAs predating upstream fix
        # 9b3d25bd (PR #158) use uint32_t/uint64_t in csrc/utils/common.h without
        # #include <cstdint>, breaking the build.  Add the include only when
        # absent; it is a compile-only migration, never an upstream reproduction.
        ch = repo / "csrc" / "utils" / "common.h"
        if ch.exists():
            before_ch = ch.read_text()
            if "#include <cstdint>" not in before_ch and "UTILS_COMMON_H" in before_ch:
                ch.write_text(before_ch.replace(
                    "#ifndef UTILS_COMMON_H",
                    "#ifndef UTILS_COMMON_H\n#include <cstdint>", 1))
        # Triton-Ascend 3.2.1 ships these CANN helpers under the extension
        # namespace, while the paired stock Triton root module does not export
        # them.  Export only the installed implementations; record this as a
        # compatibility migration rather than an upstream reproduction.
        sc = repo / "python" / "sgl_kernel_npu" / "sitecustomize.py"
        sc.write_text("""# Generated by the 8-card compatibility runner.\ntry:\n    import triton.language as _tl\n    import triton.language.extra.cann.extension as _ext\n    for _name in ('extract_slice', 'insert_slice', 'get_element'):\n        if hasattr(_ext, _name) and not hasattr(_tl, _name):\n            setattr(_tl, _name, getattr(_ext, _name))\nexcept Exception:\n    pass\n""")
        patch = subprocess.run(
            ["git", "diff", "--", "CMakeLists.txt", "csrc/CMakeLists.txt",
             "csrc/utils/common.h"],
            cwd=repo, capture_output=True, text=True).stdout
        # sitecustomize.py is intentionally untracked, so include its complete
        # creation diff in the per-PR compatibility evidence as well.
        patch += subprocess.run(
            ["git", "diff", "--no-index", "/dev/null", str(sc)],
            cwd=repo, capture_output=True, text=True).stdout
        (runs / "compat.patch").write_text(patch)
        res["compat_patch_applied"] = True
        res["compatibility_migration"] = True
        res["cstdint_compat_applied"] = "#include <cstdint>" in ch.read_text() if ch.exists() else False
        res["triton_root_compat_applied"] = sc.exists()
    except Exception as e:
        res["compat_error"] = str(e)

# --------------------------------------------------------------------- configure + build
build_dir = repo / "build"
cfg_cmd = ["cmake", "-S", ".", "-B", str(build_dir),
           "-DSOC_VERSION=Ascend910B2C",
           "-DASCEND_INCLUDE_DIR=/usr/local/Ascend/cann-9.0.0/include"]
if res.get("head"):
    rc = run(cfg_cmd, "configure.log")
    if rc != 0:
        res["status"] = "build_failed"; res["stop_reason"] = "configure failed"
    else:
        rc = run(["cmake", "--build", str(build_dir), "--target", "sgl_kernel_npu", "-j2"],
                 "build.log")
        if rc != 0:
            res["status"] = "build_failed"; res["stop_reason"] = "build failed"
        else:
            so = repo / "python" / "sgl_kernel_npu" / "sgl_kernel_npu" / "lib" / "libsgl_kernel_npu.so"
            if not so.is_file():
                res["status"] = "build_failed"
                res["stop_reason"] = "library not produced"
                res["library_absent"] = True

# DeepEP tests exercise a PR-provided pybind extension and a custom OPP
# package.  Building only the sgl_kernel_npu CMake target leaves Python to load
# an unrelated global deep_ep package, whose stock libopapi cannot contain the
# PR's aclnn symbols.  The upstream build script installs into this isolated
# checkout's python/deep_ep/deep_ep directory; it does not touch shared CANN.
if res.get("status") == "running" and cfg.get("requires_deepep_runtime"):
    # This historical build.sh derives set_env.sh from dirname(ASCEND_HOME_PATH).
    # The sourced CANN 9 environment exports the resolved cann-9.0.0 directory,
    # whose parent has no set_env.sh.  Give this child the stable toolkit/latest
    # entrypoint instead; its own source command then resolves CANN normally.
    deepep_env = {
        "ASCEND_CUSTOM_OPP_PATH": "",
        "ASCEND_HOME_PATH": "/usr/local/Ascend/ascend-toolkit/latest",
    }
    build_script = repo / "build.sh"
    # Early revisions take SoC as their first positional argument and do not
    # implement the newer `-a deepep` selector.  Passing `-a` there makes
    # CMake literally receive SOC_VERSION=-a.
    script_text = build_script.read_text(errors="replace") if build_script.exists() else ""
    ops2 = repo / "csrc" / "deepep" / "ops2" / "build.sh"
    if "getopts" in script_text and ops2.is_file():
        # This branch is the upstream-provided 910B-oriented OPP layout.  The
        # normal `deepep` selector generates ascend910_93-only package data,
        # which CANN rejects on this target as socVersion=ascend910b.
        deepep_cmd = ["bash", "build.sh", "-a", "deepep2", "Ascend910B2C"]
        res["deepep_build_mode"] = "selector_ops2_910b"
    elif "getopts" in script_text:
        deepep_cmd = ["bash", "build.sh", "-a", "deepep", "Ascend910B2C"]
        res["deepep_build_mode"] = "selector"
    else:
        deepep_cmd = ["bash", "build.sh", "Ascend910B2C"]
        res["deepep_build_mode"] = "legacy_positional_soc"
    rc = run(deepep_cmd,
             "deepep-build.log", cwd=repo, env=deepep_env, heartbeat_every=30)
    if rc != 0:
        res["status"] = "build_failed"
        res["stop_reason"] = "PR-local DeepEP/custom OPP build failed"
    else:
        deepep_pkg = repo / "python" / "deep_ep" / "deep_ep"
        custom_opapi = list(deepep_pkg.glob("vendors/*/op_api/lib/libcust_opapi.so"))
        deepep_cpp = list(deepep_pkg.glob("deep_ep_cpp*.so"))
        res["deepep_package_dir"] = str(deepep_pkg)
        res["deepep_custom_opapi"] = [str(p) for p in custom_opapi]
        res["deepep_cpp"] = [str(p) for p in deepep_cpp]
        if not custom_opapi or not deepep_cpp:
            res["status"] = "build_failed"
            res["stop_reason"] = "PR-local DeepEP build produced incomplete package"

# --------------------------------------------------------------------- repository-local import
if res.get("status") == "running" and res.get("head"):
    import_paths = [str(repo / "python" / "sgl_kernel_npu")]
    if cfg.get("requires_deepep_runtime"):
        # This historical package uses absolute ``import deep_ep_cpp`` even
        # though the extension is packaged inside deep_ep/.  Add both its
        # parent and the package directory, otherwise Python can silently
        # resolve an ABI-incompatible globally-installed deep_ep_cpp first.
        import_paths.extend([str(repo / "python" / "deep_ep" / "deep_ep"),
                             str(repo / "python" / "deep_ep")])
    import_cmd = [sys.executable, "-c", "import sgl_kernel_npu as m; print('import_ok', m.__file__); "
                  "import deep_ep as d, deep_ep_cpp as c; print('deep_ep_ok', d.__file__); print('deep_ep_cpp_ok', c.__file__)" if cfg.get("requires_deepep_runtime")
                  else "import sgl_kernel_npu as m; print('import_ok', m.__file__)"]
    env_import = {"PYTHONPATH": ":".join(import_paths)}
    rc = run(import_cmd, "import.log", env=env_import)
    if rc != 0:
        res["status"] = "requires_resolution"
        res["stop_reason"] = "repository-local import failed"
    else:
        # confirm the imported module is the repo build, not a system install
        out = (logs / "import.log").read_text(errors="replace")
        m = re.search(r"import_ok (\S+)", out)
        res["sgl_kernel_npu_file"] = m.group(1) if m else None
        repo_pkg = str(repo / "python" / "sgl_kernel_npu")
        res["repo_local_import"] = bool(m and repo_pkg in m.group(1))
        if cfg.get("requires_deepep_runtime"):
            d = re.search(r"deep_ep_ok (\S+)", out)
            c = re.search(r"deep_ep_cpp_ok (\S+)", out)
            res["deep_ep_file"] = d.group(1) if d else None
            res["deep_ep_cpp_file"] = c.group(1) if c else None
            res["repo_local_deep_ep"] = bool(d and str(repo / "python" / "deep_ep") in d.group(1))
            res["repo_local_deep_ep_cpp"] = bool(c and str(repo / "python" / "deep_ep" / "deep_ep") in c.group(1))
            if not res["repo_local_deep_ep"] or not res["repo_local_deep_ep_cpp"]:
                res["status"] = "requires_resolution"
                res["stop_reason"] = "repository-local DeepEP/deep_ep_cpp import failed"

# --------------------------------------------------------------------- correctness / probe / reference
kind = cfg.get("execution_kind")
if res.get("status") == "running":
    if kind in ("pytest", "python_spawn", "torchrun", "shell"):
        launcher = cfg.get("launcher")
        if not launcher:
            res["status"] = "evidence_incomplete"
            res["stop_reason"] = "no launcher provided"
        else:
            p_env = {}
            if cfg.get("master_port"):
                p_env["MASTER_ADDR"] = "127.0.0.1"
                p_env["MASTER_PORT"] = str(cfg["master_port"])
            # Repo-local packages plus the test file's own directory (the
            # upstream deepep scripts import sibling `utils` modules).
            paths = [str(repo / "python" / "sgl_kernel_npu")]
            if cfg.get("requires_deepep_runtime"):
                # Importing this package sets ASCEND_CUSTOM_OPP_PATH and
                # LD_LIBRARY_PATH to its PR-local vendor package.
                paths.extend([str(repo / "python" / "deep_ep" / "deep_ep"),
                              str(repo / "python" / "deep_ep")])
            if cfg.get("test"):
                test_file = repo / cfg["test"]
                paths.append(str(test_file.parent))
            # Preserve CANN's Python packages (notably tbe) inherited from
            # set_env.sh; the earlier overwrite caused GEInitialize failures.
            inherited_pythonpath = os.environ.get("PYTHONPATH", "")
            p_env["PYTHONPATH"] = ":".join(paths + ([inherited_pythonpath] if inherited_pythonpath else []))
            if kind in ("python_spawn", "torchrun"):
                # The low-latency path may be invoked by fused test entrypoints
                # whose filename does not contain "low_latency" (e.g. PR-214).
                # PR-166's tiler calculates 1913MB as the minimum; 2GiB is the
                # smallest practical test value, not the old 200MB default.
                p_env["HCCL_BUFFSIZE"] = "2048"
            res["launcher"] = launcher
            test_cwd = (repo / Path(cfg["test"]).parent) if cfg.get("test") else repo
            rc = run(launcher, "correctness.log", cwd=test_cwd, env=p_env,
                     heartbeat_every=cfg.get("heartbeat_every", 30),
                     start_new_session=True)
            if rc == 0:
                res["status"] = "validated"
                res["resolution_status"] = "closed"
            else:
                logtext = (logs / "correctness.log").read_text(errors="replace")
                cls = _classify_correctness_failure(logtext, rc=rc)
                res["correctness_log_classification"] = cls
                if cls == "test_entrypoint_mismatch":
                    res["status"] = "test_entrypoint_invalid"
                    res["stop_reason"] = "mapped test not runnable under this launcher (pytest collected nothing)"
                elif cls in ("dist_init_or_port", "device_or_runtime_error",
                             "transport_or_env", "no_module_or_import",
                             "launcher_worldsize_mismatch", "test_call_mismatch",
                             "missing_custom_opapi", "custom_op_tiling_failure"):
                    res["status"] = "requires_resolution"
                    res["stop_reason"] = "correctness run stopped in environment/entrypoint layer"
                else:
                    res["status"] = "correctness_failed"
                    res["stop_reason"] = "correctness assertion failed"
    elif kind == "reference":
        # build/import smoke only; record what a reference test would need.
        res["status"] = "reference_required"
        res["stop_reason"] = "no reference entrypoint implemented; precondition/missing-reference record"
        res["reference_strategy"] = cfg.get("reference_strategy")
        res["reference_category"] = cfg.get("reference_category")
    elif kind == "build_import_only":
        smoke_ok = res.get("build.log_rc") == 0 and res.get("import.log_rc") == 0
        res["smoke_build_ok"] = smoke_ok
        if smoke_ok:
            res["status"] = cfg.get("smoke_status", "probe_passed")
        else:
            res["status"] = res.get("status") or "build_failed"
        res["stop_reason"] = cfg.get("smoke_reason")

if res.get("status") == "running":
    res["status"] = "evidence_incomplete"
    res["stop_reason"] = "no valid execution branch reached"

persist(res)
checkpoint("finished", status=res["status"])
sys.exit(0)
'''

def classify_correctness_failure(text: str, rc: int | None = None) -> str:
    if rc == 5:
        return "test_entrypoint_mismatch"
    if re.search(r"aclnn\w+.*not in libopapi|not in libopapi\.so|libcust_opapi", text, re.I):
        return "missing_custom_opapi"
    if re.search(r"do tiling failed|TilingAndUpdateBinInfo|NnopbaseExecutorDoTiling", text, re.I):
        return "custom_op_tiling_failure"
    if re.search(r"aicore|AI_CORE|AclrtSynchronizeStream|507014|ERR00100|execute kernel param invalid|fftsplus|device.*timeout|NPU function error|SIGSEGV|ProcessExitedException|Segmentation", text, re.I):
        return "device_or_runtime_error"
    if re.search(r"world size|world_size.*less or equal|new group's world size|nproc|num-processes|num_processes.*(8|16)", text):
        return "launcher_worldsize_mismatch"
    if re.search(r"EADDRINUSE|DistNetworkError|address already in use|listen on any local", text):
        return "dist_init_or_port"
    if re.search(r"HCCL|hcclCommInit|Communication_Error|P2P communication|not on the same plane|ASCEND_VISIBLE", text):
        return "transport_or_env"
    if re.search(r"No module named|ModuleNotFoundError|ImportError|has no attribute|AttributeError", text):
        return "no_module_or_import"
    # undefined name in test harness (e.g. NameError: name 'F' is not defined) —
    # a test-source defect, not a kernel assertion failure.
    if re.search(r"NameError: name '\w+' is not defined", text):
        return "test_call_mismatch"
    # test-vs-API call mismatch (TypeError, unexpected/multiple keyword args, wrong
    # signature) — a test/entrypoint defect, not a kernel assertion failure.
    if re.search(r"TypeError:|multiple values for argument|got an unexpected keyword|takes \d+ positional|missing \d+ required", text):
        return "test_call_mismatch"
    # tensor shape/broadcast mismatch in the test harness — test/API-shape defect.
    if re.search(r"shape mismatch|cannot be broadcast|size mismatch|Sizes of tensors must match|expected.*same size", text):
        return "test_call_mismatch"
    # pytest exit code 5: collection produced no runnable tests (arg-taking test
    # functions are not pytest fixtures) — a test-entrypoint mismatch.
    if re.search(r"no tests ran|no tests collected|No tests ran|collection failed|ERROR: no tests", text):
        return "test_entrypoint_mismatch"
    if re.search(r"AssertionError|assert\b|Mismatch|tensor_equal|not close|allclose", text):
        return "assertion_mismatch"
    return "unknown_nonzero"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def deep_gate() -> None:
    cp = subprocess.run([CTL, "deep"], capture_output=True, text=True, timeout=120)
    out = cp.stdout
    needed = ("ok: True", "contents_api_ok: True", "probe_exit_code: 0")
    if cp.returncode or not all(x in out for x in needed):
        raise SystemExit(f"8-card deep gate failed rc={cp.returncode}\n{out}\n{cp.stderr}")
    if GATE_EVIDENCE.exists() and "PASS" in GATE_EVIDENCE.read_text():
        return
    raise SystemExit("8-card HCCL gate evidence is missing or not passed")


def run_ctl(command: str, timeout: int = 3600, retries: int = 3) -> subprocess.CompletedProcess:
    """Run a remote command with bounded retries for transient connection loss."""
    last: subprocess.CompletedProcess | None = None
    for attempt in range(retries):
        try:
            last = subprocess.run([CTL, "run", command], capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            if attempt == retries - 1:
                raise
            time.sleep(5)
            continue
        if last.returncode == 0:
            return last
        # terminal did not reach ready state / connection lost: retry
        if attempt == retries - 1:
            return last
        time.sleep(5)
    return last  # type: ignore[return-value]


def poll_state(remote_dir: str, timeout: int) -> dict | None:
    """Short-connection poll of remote state.json until a terminal stage.

    Also declares evidence_incomplete when the worker's heartbeat stops
    advancing (e.g. the distributed NPU fault killed the worker before it
    could write a terminal record) so the batch does not idle for an hour.
    """
    deadline = time.monotonic() + timeout
    last = None
    stale_since = None
    last_updated = None
    while time.monotonic() < deadline:
        try:
            cp = run_ctl(f"cat {remote_dir}/state.json 2>/dev/null || echo STATE_PENDING", timeout=60)
        except subprocess.TimeoutExpired:
            time.sleep(8)
            continue
        out = cp.stdout
        m = re.search(r"\{.*\}", out, re.DOTALL)
        if m:
            try:
                last = json.loads(m.group(0))
            except json.JSONDecodeError:
                last = None
        if last and last.get("stage") == "finished":
            return last
        if last and last.get("status") in ("failed", "exception", "runner_exception"):
            return last
        # Heartbeat liveness: if the worker stopped writing state for 5 min,
        # treat the attempt as interrupted (evidence_incomplete).
        updated = (last or {}).get("updated_at")
        if updated:
            if updated != last_updated:
                last_updated = updated
                stale_since = time.monotonic()
            elif stale_since and (time.monotonic() - stale_since) > 300:
                last = last or {}
                last["stage"] = "finished"
                last["status"] = "evidence_incomplete"
                last["stop_reason"] = "worker heartbeat stopped; interrupted process"
                return last
        time.sleep(8)
    return last


def download(remote_path: str, local_path: Path, timeout: int = 120, retries: int = 3) -> bool:
    for attempt in range(retries):
        try:
            cp = subprocess.run(
                [AI4QZ, "--config", CONFIG, "download", TARGET, remote_path, str(local_path), "--via-terminal"],
                capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            cp = None
        if cp is not None and cp.returncode == 0:
            return True
        if attempt == retries - 1:
            return False
        time.sleep(5)
    return False


def terminal_status(row: dict) -> bool:
    return row.get("status") in TERMINAL


def latest_local_result(pr: int) -> dict | None:
    pr_dir = VALIDATION / f"PR-{pr}" / "multicard"
    if not pr_dir.is_dir():
        return None
    stamps = sorted([p.name for p in pr_dir.iterdir() if p.is_dir()])
    for stamp in reversed(stamps):
        r = pr_dir / stamp / "result.json"
        if r.exists():
            return json.loads(r.read_text())
    return None


def free_port_remote() -> int:
    cp = run_ctl("python3 -c \"import socket; s=socket.socket(); s.bind(('127.0.0.1',0)); print(s.getsockname()[1]); s.close()\"", timeout=60)
    m = re.search(r"(\d{4,5})", cp.stdout)
    if not m:
        raise RuntimeError(f"could not probe free port: {cp.stdout} {cp.stderr}")
    return int(m.group(1))


def reference_meta(pr: int) -> dict:
    d = load_json(REF_CLASSIFICATION)
    for item in d.get("items", []):
        if int(item.get("pr")) == pr:
            return {
                "reference_category": item.get("category"),
                "reference_strategy": item.get("verification_strategy"),
                "reference_description": item.get("description"),
            }
    return {}


def dispatch_one(item: dict, cls_row: dict, max_attempts: int) -> dict:
    pr = int(item["pr"])
    sha = str(item["merge_sha"])
    kind = str(cls_row.get("execution_kind"))
    launcher = cls_row.get("launcher")
    scope = item.get("validation_scope")
    state_label = item.get("state")

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    local = VALIDATION / f"PR-{pr}" / "multicard" / stamp
    local.mkdir(parents=True, exist_ok=True)
    remote = f"{REMOTE_ROOT}/PR-{pr}/{stamp}"

    # one MASTER_PORT per attempt, recorded before dispatch
    port = free_port_remote()
    (local / "dispatch.master_port.txt").write_text(str(port) + "\n")

    smoke_status = "probe_passed"
    smoke_reason = None
    if kind == "reference":
        smoke_status = "reference_required"
        smoke_reason = "no reference entrypoint; build/import smoke only"
    elif kind == "build_import_only":
        if cls_row.get("status") == "test_missing_at_target_sha":
            smoke_status = "not_applicable"
            smoke_reason = "mapped test missing at exact SHA; build/import probe only"
        elif cls_row.get("status") == "test_entrypoint_invalid":
            smoke_status = "test_entrypoint_invalid"
            smoke_reason = "mapped test empty/invalid at exact SHA; build/import probe only"
        else:
            smoke_status = "probe_passed"
            smoke_reason = "no correctness entrypoint defined; build/import probe only"

    cfg = {
        "pr": pr, "sha": sha, "run_dir": remote, "execution_kind": kind,
        "launcher": launcher, "master_port": port, "world_size": 8,
        "validation_scope": scope, "state": state_label,
        "compatibility": item.get("compatibility") or [],
        "compatibility_migration": True,
        "requires_deepep_runtime": bool((cls_row.get("markers") or {}).get("imports_deepep")),
        "smoke_status": smoke_status, "smoke_reason": smoke_reason,
        "heartbeat_every": 30,
    }
    if kind == "reference":
        cfg.update(reference_meta(pr))

    worker_b64 = base64.b64encode(REMOTE_WORKER.encode()).decode()
    cfg_b64 = base64.b64encode(json.dumps(cfg).encode()).decode()
    cmd = (
        "source /usr/local/Ascend/ascend-toolkit/set_env.sh; "
        "unset ASCEND_VISIBLE_DEVICES ASCEND_RT_VISIBLE_DEVICES; "
        "export HCCL_INTRA_ROCE_ENABLE=1; "
        f"mkdir -p {remote}; "
        f"echo {worker_b64} | base64 -d > {remote}/worker.py; "
        f"echo {cfg_b64} | base64 -d > {remote}/config.json; "
        f"setsid nohup python3 -u {remote}/worker.py {remote}/config.json "
        f">{remote}/launcher.log 2>&1 </dev/null & echo TASK_SUBMITTED_PID=$!"
    )
    cp = run_ctl(cmd, timeout=120)
    (local / "dispatch.stdout.log").write_text(cp.stdout)
    (local / "dispatch.stderr.log").write_text(cp.stderr)
    (local / "dispatch.cfg.json").write_text(json.dumps(cfg, indent=2) + "\n")

    summary = {
        "pr": pr, "merge_sha": sha, "target": TARGET, "execution_kind": kind,
        "dispatch_rc": cp.returncode, "remote_evidence": remote,
        "local_evidence": str(local), "status": "submitted", "master_port": port,
    }
    (local / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary, remote, local


def finalize(pr: int, remote: str, local: Path, summary: dict) -> dict:
    # Poll until terminal, then download result.json + key logs.
    poll_secs = 60 * 60  # 1h cap; long builds+tests
    final_state = poll_state(remote, timeout=poll_secs)
    result: dict = {}
    ok = download(f"{remote}/result.json", local / "result.json")
    if ok:
        try:
            result = json.loads((local / "result.json").read_text())
        except json.JSONDecodeError:
            result = {"status": "evidence_incomplete", "stop_reason": "result.json unparsable"}
    else:
        result = {"status": "evidence_incomplete", "stop_reason": "result.json not recoverable"}
    for name in ("state.json", "logs/git-clone.log", "logs/git-fetch.log",
                 "logs/git-checkout.log", "logs/configure.log", "logs/build.log",
                 "logs/deepep-build.log", "logs/import.log", "logs/correctness.log", "logs/runner-exception.log",
                 "environment.log", "runs/compat.patch"):
        fn = name.split("/")[-1]
        download(f"{remote}/{name}", local / fn, timeout=60)
    result["local_evidence"] = str(local)
    result["remote_evidence"] = remote
    result["final_state"] = final_state
    (local / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def update_ledger(result: dict, item: dict) -> None:
    ledger = load_json(LEDGER)
    ledger.setdefault("schema_version", 1)
    rows = ledger.setdefault("rows", {})
    pr = int(item["pr"])
    rows[str(pr)] = {
        "pr": pr,
        "merge_sha": item.get("merge_sha"),
        "validation_scope": item.get("validation_scope"),
        "execution_kind": result.get("execution_kind"),
        "status": result.get("status"),
        "stop_reason": result.get("stop_reason"),
        "correctness_log_classification": result.get("correctness_log_classification"),
        "configure_rc": result.get("configure.log_rc"),
        "build_rc": result.get("build.log_rc"),
        "import_rc": result.get("import.log_rc"),
        "correctness_rc": result.get("correctness.log_rc"),
        "repo_local_import": result.get("repo_local_import"),
        "master_port": result.get("master_port"),
        "local_evidence": result.get("local_evidence"),
        "remote_evidence": result.get("remote_evidence"),
        "updated_at": result.get("updated_at") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    LEDGER.write_text(json.dumps(ledger, indent=2) + "\n")


def load_batch_state() -> dict:
    return load_json(BATCH_STATE)


def save_batch_state(state: dict) -> None:
    state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    BATCH_STATE.write_text(json.dumps(state, indent=2) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="8-card batch coverage runner")
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--pr", type=int, action="append", help="run only these PRs")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-reference", action="store_true")
    ap.add_argument("--deepep-only", action="store_true",
                    help="run only manifest entries that import the vendored DeepEP package")
    ap.add_argument("--after-pr", type=int, default=0,
                    help="run only PR numbers greater than this checkpoint")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-attempts", type=int, default=2)
    args = ap.parse_args()

    manifest = load_json(args.manifest)
    items = manifest["items"]

    deep_gate()
    print("8-card deep gate: PASS", flush=True)

    if args.dry_run:
        sel = [x for x in items if not args.pr or x["pr"] in args.pr]
        if args.skip_reference:
            sel = [x for x in sel if x.get("execution_kind") != "reference"]
        print(f"dry-run: {len(sel)} bundles would run")
        from collections import Counter
        print(json.dumps(Counter(x.get("execution_kind") for x in sel), indent=2))
        return 0

    queue = load_json(MANIFEST.parent / "full-execution-queue.json")
    candidates = {int(x["pr"]): x for x in queue["candidates"]}

    selected = [x for x in items if not args.pr or x["pr"] in args.pr]
    if args.deepep_only:
        selected = [x for x in selected if (x.get("markers") or {}).get("imports_deepep")]
    if args.after_pr:
        selected = [x for x in selected if int(x["pr"]) > args.after_pr]
    if args.skip_reference:
        selected = [x for x in selected if x.get("execution_kind") != "reference"]
    if args.limit:
        selected = selected[: args.limit]

    bs = load_batch_state()
    in_flight = bs.setdefault("in_flight", {})

    print(f"=== {len(selected)} bundles selected ===", flush=True)
    # The ledger is the authoritative resume record: it is only written after a
    # run's evidence was downloaded locally.  Local result.json files alone are
    # not trustworthy (a mid-flight runner bug could leave a misleading status).
    ledger = load_json(LEDGER).get("rows", {})
    for idx, cls_row in enumerate(selected):
        pr = int(cls_row["pr"])
        item = candidates.get(pr, cls_row)

        if args.resume and str(pr) in ledger:
            row = ledger[str(pr)]
            if terminal_status(row):
                print(f"PR-{pr}: ledger has terminal status {row['status']}, skipping (resume)", flush=True)
                continue

        # re-poll an in-flight job instead of double-dispatching
        if str(pr) in in_flight:
            fl = in_flight[str(pr)]
            print(f"PR-{pr}: in-flight job exists, polling {fl['remote']}", flush=True)
            result = finalize(pr, fl["remote"], Path(fl["local"]), fl.get("summary", {}))
            update_ledger(result, item)
            if terminal_status(result):
                in_flight.pop(str(pr))
            save_batch_state(bs)
            print(f"PR-{pr}: {result.get('status')} {result.get('stop_reason','')}", flush=True)
            continue

        print(f"\n=== [{idx+1}/{len(selected)}] PR-{pr} {cls_row.get('execution_kind')} "
              f"sha={str(cls_row['merge_sha'])[:12]} ===", flush=True)

        summary, remote, local = dispatch_one(item, cls_row, args.max_attempts)
        in_flight[str(pr)] = {"remote": remote, "local": str(local), "summary": summary}
        save_batch_state(bs)

        result = finalize(pr, remote, local, summary)
        update_ledger(result, item)
        if terminal_status(result):
            in_flight.pop(str(pr), None)
        else:
            in_flight[str(pr)]["status"] = result.get("status")
        save_batch_state(bs)
        print(f"PR-{pr}: status={result.get('status')} stop={result.get('stop_reason','')} "
              f"evidence={local}", flush=True)

    print("\n=== batch done ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
