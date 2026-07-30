#!/usr/bin/env python3
"""V2 Ascend 910B validation orchestrator: preflight, resume, result sync.

Improvements over V1:
- Preflight gate (910bctl deep + NPU smoke) before any work.
- Bundle SHA integrity check per PR.
- Test-file existence verified at target SHA before pytest.
- Results downloaded to canonical local path after each PR.
- Checkpoint/resume: completed PRs are never re-run.
- Failure classifcation by gate.
- coverage_probe label preserved; heuristic passes never promote.
- Per-PR isolated compatibility patches.
"""
from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[3]   # kda-h200-workspace
CORPUS = WORKSPACE / "npu-kernelwiki"
CTL = Path("/home/kirin_14379/projects/ai4qz/scripts/910bctl")

# Canonical local results root
LOCAL_RESULTS = CORPUS / "validation" / "ascend-910b"

# Queue state file for resume
STATE_FILE = HERE / "queue-run-v2-state.json"

# Remote root for isolated per-PR work
REMOTE_ROOT = "/inspire/sj-ssd/project/qianghuaxuexi/s26043/npu-kernelwiki-validation"


# ---------------------------------------------------------------------------
# Preflight gate
# ---------------------------------------------------------------------------
def run_preflight() -> dict[str, object]:
    """Both local connectivity and remote NPU smoke. Fatal on failure."""
    result: dict[str, object] = {}

    # 1. Local connectivity probe
    cp = subprocess.run(
        [str(CTL), "deep"], capture_output=True, text=True, timeout=120
    )
    for line in cp.stdout.splitlines():
        line = line.strip()
        for field in ("ok", "xsrf_found", "contents_api_ok", "probe_exit_code"):
            if line.startswith(f"{field}:"):
                result[f"deep_{field}"] = line.split(":", 1)[1].strip()
    if cp.returncode != 0:
        raise SystemExit(f"PREFLIGHT FAIL: 910bctl deep exit {cp.returncode}")

    # 2. Remote NPU smoke — check device is usable
    cp = subprocess.run(
        [str(CTL), "run",
         "source /usr/local/Ascend/ascend-toolkit/set_env.sh; "
         "unset ASCEND_VISIBLE_DEVICES ASCEND_RT_VISIBLE_DEVICES; "
         "python3 -c \"import torch; import torch_npu; "
         "assert torch.npu.is_available(), 'NPU not available'; "
         "print('npu_smoke: device_count', torch.npu.device_count())\""],
        capture_output=True, text=True, timeout=120,
    )
    result["npu_smoke_rc"] = cp.returncode
    result["npu_smoke_output"] = cp.stdout.strip().splitlines()[-1] if cp.stdout.strip() else ""
    if cp.returncode != 0:
        raise SystemExit(
            f"PREFLIGHT FAIL: NPU smoke failed rc={cp.returncode}\n{cp.stdout}\n{cp.stderr}"
        )
    return result


# ---------------------------------------------------------------------------
# Remote checks (no full run)
# ---------------------------------------------------------------------------
def remote_test_exists(test_path: str, merge_sha: str, target_index: int) -> bool:
    """Check whether *test_path* exists inside a shallow clone at *merge_sha*."""
    script = (
        "source /usr/local/Ascend/ascend-toolkit/set_env.sh; "
        "unset ASCEND_VISIBLE_DEVICES ASCEND_RT_VISIBLE_DEVICES; "
        "TMPDIR=$(mktemp -d); "
        "git clone --filter=blob:none --no-checkout "
        "https://github.com/sgl-project/sgl-kernel-npu.git \"$TMPDIR\" 2>/dev/null; "
        "cd \"$TMPDIR\"; "
        "git checkout --detach " + merge_sha + " -- " + test_path + " 2>/dev/null; "
        "test -f \"" + test_path + "\" && echo 'TEST_EXISTS' || echo 'TEST_MISSING'; "
        "rm -rf \"$TMPDIR\""
    )
    cp = subprocess.run(
        [str(CTL), "run", script], capture_output=True, text=True, timeout=120
    )
    return "TEST_EXISTS" in cp.stdout


# ---------------------------------------------------------------------------
# Load / save state
# ---------------------------------------------------------------------------
def load_state() -> dict[str, object]:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"completed_prs": [], "failed_prs": {}, "in_progress": None}


def save_state(state: dict[str, object]) -> None:
    state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    STATE_FILE.write_text(json.dumps(state, indent=2) + "\n")


# ---------------------------------------------------------------------------
# Bundle integrity
# ---------------------------------------------------------------------------
def require_bundle(pr: int, sha: str) -> None:
    page = CORPUS / "sources/prs/sgl-kernel-npu" / f"PR-{pr}.md"
    bundle = CORPUS / "artifacts/prs/sgl-kernel-npu" / f"PR-{pr}"
    text = page.read_text()
    if f'merge_sha: "{sha}"' not in text:
        raise ValueError(f"PR-{pr}: manifest SHA differs from source page")
    for required in (bundle / "diff.patch", bundle / "PROVENANCE.yaml"):
        if not required.is_file():
            raise ValueError(f"PR-{pr}: missing {required.name}")
    if not any(p.is_file() for p in (bundle / "key-files").rglob("*")):
        raise ValueError(f"PR-{pr}: empty key-files")


# ---------------------------------------------------------------------------
# Remote runner template (runs on 910B)
# ---------------------------------------------------------------------------
REMOTE_RUNNER = r'''#!/usr/bin/env python3
import json, os, subprocess, sys, time
from pathlib import Path

cfg = json.loads(Path(sys.argv[1]).read_text())
base = Path(cfg["run_dir"])
repo = base / "repo" / "sgl-kernel-npu"
for d in (base / "logs", base / "runs", base / "outputs", base / "profile"):
    d.mkdir(parents=True, exist_ok=True)

env = os.environ.copy()
env["PYTHONPATH"] = "python/sgl_kernel_npu" + (
    ":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
)

res = {
    "pr": cfg["pr"],
    "merge_sha": cfg["merge_sha"],
    "environment": "Ascend910B2C/CANN-9.0.0 current server",
    "test_mapping": cfg.get("test_mapping", "reviewed"),
    "validation_scope": cfg.get("validation_scope", "correctness"),
    "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}


def run(cmd, name, cwd=None):
    cwd = cwd or repo
    start = time.monotonic()
    with (base / "logs" / name).open("w") as f:
        f.write("$ " + " ".join(cmd) + "\n")
        p = subprocess.run(cmd, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT, text=True)
        f.write(f"\n[exit_code={p.returncode}]\n")
    return p.returncode, round(time.monotonic() - start, 3)


# --- Clone & checkout ---
if not repo.exists():
    repo.parent.mkdir(parents=True, exist_ok=True)
    res["clone_rc"], _ = run(
        ["git", "clone", "--no-checkout",
         "https://github.com/sgl-project/sgl-kernel-npu.git", str(repo)],
        "git-clone.log", base)
else:
    res["clone_rc"] = 0

if res["clone_rc"] == 0:
    res["fetch_rc"], _ = run(
        ["git", "fetch", "--depth=1", "origin", cfg["merge_sha"]],
        "git-fetch.log")

if res.get("fetch_rc") == 0:
    res["checkout_rc"], _ = run(
        ["git", "checkout", "--detach", cfg["merge_sha"]],
        "git-checkout.log")

# --- Verify SHA ---
if res.get("checkout_rc") == 0:
    res["head"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if res["head"] != cfg["merge_sha"]:
        res["stop_reason"] = "checkout SHA mismatch"
        res["gate_failure"] = "checkout"

# --- Verify test file exists at this SHA ---
if "stop_reason" not in res and cfg.get("test"):
    test_file = repo / cfg["test"]
    if not test_file.is_file():
        res["stop_reason"] = f"test file {cfg['test']} missing at target SHA"
        res["gate_failure"] = "test_missing_at_target_sha"

# --- Compatibility patch (per PR, isolated) ---
if "stop_reason" not in res:
    compat = cfg.get("compatibility", [])
    if isinstance(compat, list) and len(compat) > 0:
        # Collect before-patch state
        cm_before = (repo / "CMakeLists.txt").read_text()
        cs_before = (repo / "csrc" / "CMakeLists.txt").read_text()

        if "gcc_acl" in compat:
            cm = repo / "CMakeLists.txt"
            cs = repo / "csrc" / "CMakeLists.txt"
            cm.write_text(
                cm.read_text().replace(
                    "-hno-unused-parameter -lno-unused-function ", ""))
            acl_path = "${TORCH_NPU_DIR}/include/third_party/acl/inc"
            cs_content = cs.read_text()
            if acl_path not in cs_content:
                cs.write_text(
                    cs_content.replace(
                        "${TORCH_NPU_DIR}/include",
                        "${TORCH_NPU_DIR}/include\n        ${TORCH_NPU_DIR}/include/third_party/acl/inc",
                        1))

        with (base / "runs" / "compat.patch").open("w") as f:
            subprocess.run(
                ["git", "diff", "--", "CMakeLists.txt", "csrc/CMakeLists.txt"],
                cwd=repo, stdout=f, text=True, check=True)
        res["compat_patch_applied"] = True
        res["compat_patch_tags"] = compat

# --- Configure ---
if "stop_reason" not in res:
    build = repo / "build"
    res["configure_rc"], _ = run(
        ["cmake", "-S", ".", "-B", str(build),
         "-DBUILD_DEEPEP_MODULE=OFF",
         "-DSOC_VERSION=Ascend910B2C",
         "-DASCEND_INCLUDE_DIR=/usr/local/Ascend/cann-9.0.0/include"],
        "configure.log")
    if res["configure_rc"] != 0:
        res["stop_reason"] = "configure failed"
        res["gate_failure"] = "configure"

# --- Build ---
if "stop_reason" not in res:
    res["build_rc"], _ = run(
        ["cmake", "--build", str(build), "--target", "sgl_kernel_npu", "-j2"],
        "build.log")
    if res["build_rc"] != 0:
        res["stop_reason"] = "build failed"
        res["gate_failure"] = "build"

# --- Import check ---
if "stop_reason" not in res:
    so_path = build / "python" / "sgl_kernel_npu" / "sgl_kernel_npu" / "lib" / "libsgl_kernel_npu.so"
    if not so_path.is_file():
        res["stop_reason"] = "library not produced"
        res["gate_failure"] = "build"
        res["library_absent"] = True
    else:
        import_cmd = [
            sys.executable, "-c",
            "import sys; sys.path.insert(0,'python/sgl_kernel_npu'); "
            "import sgl_kernel_npu; print('import_ok')"]
        res["import_rc"], _ = run(import_cmd, "import.log")
        if res["import_rc"] != 0:
            res["stop_reason"] = "repository-local import failed"
            res["gate_failure"] = "import"

# --- Correctness ---
if "stop_reason" not in res and cfg.get("test"):
    res["correctness_rc"], _ = run(
        [sys.executable, "-m", "pytest", "-q", cfg["test"]],
        "correctness.log")
    if res["correctness_rc"] != 0:
        res["stop_reason"] = "correctness failed"
        res["gate_failure"] = "correctness_mismatch"

# --- Benchmark (wall-time repeats) ---
if "stop_reason" not in res and cfg.get("test"):
    samples = []
    for i in range(cfg.get("benchmark_repeats", 3)):
        rc, sec = run(
            [sys.executable, "-m", "pytest", "-q", cfg["test"]],
            f"benchmark-{i + 1}.log")
        samples.append({"exit_code": rc, "seconds": sec})
        if rc:
            break
    res["benchmark_samples"] = samples
    res["benchmark_rc"] = 0 if all(s["exit_code"] == 0 for s in samples) else 1
    (base / "outputs" / "benchmark-walltime.json").write_text(
        json.dumps({
            "kind": "upstream-test wall-time only; not operator performance",
            "samples": samples,
        }, indent=2) + "\n")
    if res["benchmark_rc"] != 0:
        res["stop_reason"] = res.get("stop_reason", "benchmark repeat failed")

# --- Profile ---
if "stop_reason" not in res and cfg.get("profile_mode") == "pytest":
    profile_script = (
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "import pytest, torch, torch_npu\n"
        "cfg = json.loads(Path(sys.argv[1]).read_text())\n"
        "repo = Path(sys.argv[2])\n"
        "out = Path(sys.argv[3])\n"
        "os.chdir(repo)\n"
        "out.mkdir(parents=True, exist_ok=True)\n"
        "activities = [torch_npu.profiler.ProfilerActivity.CPU, "
        "torch_npu.profiler.ProfilerActivity.NPU]\n"
        "with torch_npu.profiler.profile("
        "activities=activities, "
        "schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1), "
        "on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(out)), "
        "record_shapes=True, profile_memory=True) as prof:\n"
        "    rc = pytest.main([cfg['test'], '-q'])\n"
        "    torch.npu.synchronize()\n"
        "    prof.step()\n"
        "print(f'PYTEST_RC={repr(rc)}')\n"
        "raise SystemExit(rc)\n"
    )
    script_path = base / "runs" / "profile_pytest.py"
    script_path.write_text(profile_script)
    res["profile_rc"], _ = run(
        [sys.executable, str(script_path),
         str(Path(sys.argv[1]).resolve()), str(repo), str(base / "profile")],
        "profile.log", repo)

res["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
(base / "result.json").write_text(json.dumps(res, indent=2) + "\n")
print(json.dumps(res, sort_keys=True))
'''


# ---------------------------------------------------------------------------
# Download results from remote
# ---------------------------------------------------------------------------
def download_results(pr: int) -> bool:
    """Pull remote logs, result.json, profile traces to canonical local dir."""
    local_pr = LOCAL_RESULTS / f"PR-{pr}"
    local_pr.mkdir(parents=True, exist_ok=True)

    remote_pr = f"{REMOTE_ROOT}/PR-{pr}"

    # Download result.json
    cp = subprocess.run(
        ["ai4qz", "download", "ascend_910b",
         f"{remote_pr}/result.json",
         str(local_pr / "result.json"),
         "--via-terminal"],
        capture_output=True, text=True, timeout=120,
    )
    if cp.returncode != 0:
        print(f"  WARN: download result.json failed for PR-{pr}")
        return False

    # Download logs directory
    for log_name in ["configure.log", "build.log", "import.log",
                      "correctness.log", "profile.log",
                      "git-clone.log", "git-fetch.log", "git-checkout.log"]:
        for suffix in ["", ".1", ".2", ".3"]:  # benchmark variants
            remote_log = f"{remote_pr}/logs/{log_name}"
            local_log_dir = local_pr / "evidence" / "logs"
            local_log_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["ai4qz", "download", "ascend_910b",
                 remote_log, str(local_log_dir / log_name),
                 "--via-terminal"],
                capture_output=True, text=True, timeout=60,
            )

    # Download compat patch if exists
    subprocess.run(
        ["ai4qz", "download", "ascend_910b",
         f"{remote_pr}/runs/compat.patch",
         str(local_pr / "evidence" / "runs" / "compat.patch"),
         "--via-terminal"],
        capture_output=True, text=True, timeout=60,
    )

    # Download benchmark data
    subprocess.run(
        ["ai4qz", "download", "ascend_910b",
         f"{remote_pr}/outputs/benchmark-walltime.json",
         str(local_pr / "evidence" / "outputs" / "benchmark-walltime.json"),
         "--via-terminal"],
        capture_output=True, text=True, timeout=60,
    )

    return True


# ---------------------------------------------------------------------------
# Run one PR on 910B
# ---------------------------------------------------------------------------
def run_one_pr(item: dict[str, object]) -> dict[str, object]:
    pr = int(item["pr"])
    sha = str(item["merge_sha"])

    remote_dir = f"{REMOTE_ROOT}/PR-{pr}"
    remote_cfg = {
        **item,
        "run_dir": remote_dir,
        "benchmark_repeats": 3,
        "profile_mode": item.get("profile_mode", "pytest"),
    }

    runner_b64 = base64.b64encode(REMOTE_RUNNER.encode()).decode()
    cfg_b64 = base64.b64encode(json.dumps(remote_cfg).encode()).decode()

    command = (
        "source /usr/local/Ascend/ascend-toolkit/set_env.sh; "
        "unset ASCEND_VISIBLE_DEVICES ASCEND_RT_VISIBLE_DEVICES; "
        f"mkdir -p {remote_dir}; "
        f"echo {runner_b64} | base64 -d > {remote_dir}/runner.py; "
        f"echo {cfg_b64} | base64 -d > {remote_dir}/config.json; "
        f"python3 {remote_dir}/runner.py {remote_dir}/config.json"
    )

    print(f"  PR-{pr}: dispatching to 910B...", flush=True)
    cp = subprocess.run(
        [str(CTL), "run", command], capture_output=True, text=True, timeout=3600,
    )

    # Try to parse remote runner's JSON output from 910bctl stdout
    result: dict[str, object] = {}
    for line in cp.stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                result = json.loads(line)
            except json.JSONDecodeError:
                pass
    if not result:
        result = {
            "pr": pr,
            "merge_sha": sha,
            "dispatch_stderr": cp.stderr[-2000:] if cp.stderr else "",
            "dispatch_rc": cp.returncode,
            "stop_reason": "no JSON result parsed from remote runner",
        }

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="V2 910B validation orchestrator")
    ap.add_argument("--manifest", type=Path,
                    default=HERE / "full-execution-queue.json",
                    help="queue manifest")
    ap.add_argument("--pr", type=int, action="append",
                    help="run only these PRs (repeatable)")
    ap.add_argument("--execute", action="store_true",
                    help="actually run on 910B (dry-run otherwise)")
    ap.add_argument("--resume", action="store_true",
                    help="skip PRs already completed in state file")
    ap.add_argument("--limit", type=int, default=0,
                    help="max PRs to run in this batch (0 = unlimited)")
    ap.add_argument("--compat-mode", action="store_true",
                    help="apply GCC/ACL CMake compatibility patch to every PR")
    ap.add_argument("--skip-reference-required", action="store_true",
                    help="skip 28 reference_required entries")
    ap.add_argument("--skip-heuristic", action="store_true",
                    help="skip 160 heuristic probe entries")
    args = ap.parse_args()

    # --- Gate 1: Load manifest ---
    queue = json.loads(args.manifest.read_text())
    items = [x for x in queue["candidates"] if not args.pr or x["pr"] in args.pr]

    if not items:
        raise SystemExit("no candidates selected")

    # --- Gate 2: Preflight ---
    print("=== Preflight gate ===", flush=True)
    pf = run_preflight()
    print(json.dumps(pf, indent=2), flush=True)

    # --- Gate 3: Filter & validate bundles ---
    state = load_state() if args.resume else {"completed_prs": [], "failed_prs": {}}

    to_run = []
    for item in items:
        pr = int(item["pr"])

        # Skip completed
        if args.resume and pr in state.get("completed_prs", []):
            print(f"PR-{pr}: already completed, skipping (resume)")
            continue

        # Skip reference_required
        if args.skip_reference_required and item.get("state") == "reference_required":
            print(f"PR-{pr}: reference_required, skipping")
            continue

        # Skip heuristic probes
        if args.skip_heuristic and item.get("validation_scope") == "coverage_probe":
            print(f"PR-{pr}: coverage_probe, skipping")
            continue

        # Validate bundle
        try:
            require_bundle(pr, str(item["merge_sha"]))
        except ValueError as e:
            print(f"PR-{pr}: bundle validation failed: {e}")
            state.setdefault("failed_prs", {})[str(pr)] = str(e)
            save_state(state)
            continue

        # Auto-apply compat for all builds (except PR-592)
        if args.compat_mode and pr != 592:
            item.setdefault("compatibility", []).append("gcc_acl")

        to_run.append(item)

    if args.limit > 0:
        to_run = to_run[: args.limit]

    print(f"\n=== {len(to_run)} PRs to run ===", flush=True)
    if not args.execute:
        print("dry-run only; add --execute to run on 910B")
        for item in to_run[:5]:
            print(f"  PR-{item['pr']}: {item.get('test','no test')} [{item.get('validation_scope','?')}]")
        if len(to_run) > 5:
            print(f"  ... and {len(to_run)-5} more")
        return 0

    # --- Gate 4: Execute ---
    for idx, item in enumerate(to_run):
        pr = int(item["pr"])
        sha = str(item["merge_sha"])
        scope = item.get("validation_scope", "correctness")
        tag = f"[{scope}]"
        print(f"\n=== [{idx+1}/{len(to_run)}] PR-{pr} {tag} sha={sha[:12]} ===", flush=True)

        # Pre-check: test file exists at target SHA?
        test_path = item.get("test")
        if test_path:
            exists = remote_test_exists(test_path, sha, pr)
            if not exists:
                record = {
                    "pr": pr, "merge_sha": sha,
                    "state": "test_missing_at_target_sha",
                    "test": test_path,
                    "validation_scope": scope,
                    "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
            else:
                record = run_one_pr(item)
        else:
            record = {
                "pr": pr, "merge_sha": sha,
                "state": "reference_required",
                "reason": "no test path — needs reference design",
                "validation_scope": scope,
                "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }

        # Save locally immediately
        local_pr = LOCAL_RESULTS / f"PR-{pr}"
        local_pr.mkdir(parents=True, exist_ok=True)
        (local_pr / "result.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n")

        # Download remote evidence
        if record.get("clone_rc") == 0 or record.get("checkout_rc") == 0:
            download_results(pr)

        # Update state
        state.setdefault("completed_prs", []).append(pr)
        if record.get("stop_reason"):
            state.setdefault("failed_prs", {})[str(pr)] = record["stop_reason"]
        save_state(state)

        # Summary
        status = record.get("stop_reason", "OK")
        print(f"  PR-{pr}: gate={'PASS' if 'stop_reason' not in record else record.get('gate_failure','?')} "
              f"stop={status}", flush=True)

    print(f"\n=== Done. {len(to_run)} PRs processed ===", flush=True)
    print(f"State file: {STATE_FILE}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
