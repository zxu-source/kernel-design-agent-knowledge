#!/usr/bin/env python3
"""Remote validation runner for sgl-kernel-npu on Ascend 910B.

Uploaded once, then executed with: python3 runner_v2.py <config.json>
Prints RESULT_JSON_START ... RESULT_JSON_END markers for easy parsing."""
import json, os, subprocess, sys, time
from pathlib import Path

cfg = json.loads(Path(sys.argv[1]).read_text())
base = Path(cfg["run_dir"])
repo = base / "repo" / "sgl-kernel-npu"
for d in (base / "logs", base / "runs", base / "outputs", base / "profile"):
    d.mkdir(parents=True, exist_ok=True)

env = os.environ.copy()
env["PYTHONPATH"] = "python/sgl_kernel_npu" + (
    ":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

res = {
    "pr": cfg["pr"],
    "merge_sha": cfg["merge_sha"],
    "environment": "Ascend910B2C/CANN-9.0.0 server",
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

# --- Clone ---
if not repo.exists():
    repo.parent.mkdir(parents=True, exist_ok=True)
    res["clone_rc"], _ = run(["git", "clone", "--no-checkout",
        "https://github.com/sgl-project/sgl-kernel-npu.git", str(repo)], "git-clone.log", base)
else:
    res["clone_rc"] = 0

if res["clone_rc"] == 0:
    res["fetch_rc"], _ = run(["git", "fetch", "--depth=1", "origin", cfg["merge_sha"]], "git-fetch.log")

if res.get("fetch_rc") == 0:
    res["checkout_rc"], _ = run(["git", "checkout", "--detach", cfg["merge_sha"]], "git-checkout.log")

# --- Verify SHA ---
if res.get("checkout_rc") == 0:
    res["head"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if res["head"] != cfg["merge_sha"]:
        res["stop_reason"] = "checkout SHA mismatch"
        res["gate_failure"] = "checkout"

# --- Test file existence ---
if "stop_reason" not in res and cfg.get("test"):
    test_file = repo / cfg["test"]
    if not test_file.is_file():
        res["stop_reason"] = f"test file {cfg['test']} missing at target SHA"
        res["gate_failure"] = "test_missing_at_target_sha"

# --- Compatibility patch ---
if "stop_reason" not in res:
    compat = cfg.get("compatibility", [])
    if isinstance(compat, list) and len(compat) > 0:
        cm_path = repo / "CMakeLists.txt"
        cs_path = repo / "csrc" / "CMakeLists.txt"
        if not cm_path.is_file() or not cs_path.is_file():
            res["stop_reason"] = "CMakeLists.txt missing at target SHA (pre-bootstrap?)"
            res["gate_failure"] = "configure"
        else:
            if "gcc_acl" in compat:
                cm_path.write_text(cm_path.read_text().replace("-hno-unused-parameter -lno-unused-function ", ""))
                acl_path = "${TORCH_NPU_DIR}/include/third_party/acl/inc"
                cs_content = cs_path.read_text()
                if acl_path not in cs_content:
                    cs_path.write_text(cs_content.replace(
                        "${TORCH_NPU_DIR}/include",
                        "${TORCH_NPU_DIR}/include\n        ${TORCH_NPU_DIR}/include/third_party/acl/inc", 1))
            with (base / "runs" / "compat.patch").open("w") as f:
                subprocess.run(["git", "diff", "--", "CMakeLists.txt", "csrc/CMakeLists.txt"],
                               cwd=repo, stdout=f, text=True)
            res["compat_patch_applied"] = True

# --- Configure ---
if "stop_reason" not in res:
    if not (repo / "CMakeLists.txt").is_file():
        res["stop_reason"] = "CMakeLists.txt missing at target SHA (pre-bootstrap?)"
        res["gate_failure"] = "configure"
    else:
        build = repo / "build"
        res["configure_rc"], _ = run([
            "cmake", "-S", ".", "-B", str(build),
            "-DBUILD_DEEPEP_MODULE=OFF",
            "-DSOC_VERSION=Ascend910B2C",
            "-DASCEND_INCLUDE_DIR=/usr/local/Ascend/cann-9.0.0/include"], "configure.log")
        if res["configure_rc"] != 0:
            res["stop_reason"] = "configure failed"
            res["gate_failure"] = "configure"

# --- Build ---
if "stop_reason" not in res:
    res["build_rc"], _ = run(["cmake", "--build", str(build), "--target", "sgl_kernel_npu", "-j2"], "build.log")
    if res["build_rc"] != 0:
        res["stop_reason"] = "build failed"
        res["gate_failure"] = "build"

# --- Import check ---
if "stop_reason" not in res:
    so = repo / "python" / "sgl_kernel_npu" / "sgl_kernel_npu" / "lib" / "libsgl_kernel_npu.so"
    if not so.is_file():
        res["stop_reason"] = "library not produced"
        res["gate_failure"] = "build"
        res["library_absent"] = True
    else:
        res["import_rc"], _ = run([sys.executable, "-c",
            "import sys; sys.path.insert(0,'python/sgl_kernel_npu'); import sgl_kernel_npu; print('import_ok')"],
            "import.log")
        if res["import_rc"] != 0:
            res["stop_reason"] = "import failed"
            res["gate_failure"] = "import"

# --- Correctness ---
if "stop_reason" not in res and cfg.get("test"):
    res["correctness_rc"], _ = run([sys.executable, "-m", "pytest", "-q", cfg["test"]], "correctness.log")
    if res["correctness_rc"] != 0:
        res["stop_reason"] = "correctness failed"
        res["gate_failure"] = "correctness_mismatch"

# --- Benchmark (wall-time repeats) ---
if "stop_reason" not in res and cfg.get("test"):
    samples = []
    for i in range(cfg.get("benchmark_repeats", 3)):
        rc, sec = run([sys.executable, "-m", "pytest", "-q", cfg["test"]], f"benchmark-{i+1}.log")
        samples.append({"exit_code": rc, "seconds": sec})
        if rc:
            break
    res["benchmark_samples"] = samples
    res["benchmark_rc"] = 0 if all(s["exit_code"] == 0 for s in samples) else 1
    (base / "outputs" / "benchmark-walltime.json").write_text(json.dumps(
        {"kind": "upstream wall-time only; not operator performance", "samples": samples}, indent=2) + "\n")

# --- Profile ---
if "stop_reason" not in res and cfg.get("profile_mode") == "pytest":
    profile_script = (
        "import json, os, sys\nfrom pathlib import Path\n"
        "import pytest, torch, torch_npu\n"
        "cfg = json.loads(Path(sys.argv[1]).read_text())\n"
        "repo = Path(sys.argv[2])\nout = Path(sys.argv[3])\n"
        "os.chdir(repo)\nout.mkdir(parents=True, exist_ok=True)\n"
        "activities = [torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU]\n"
        "with torch_npu.profiler.profile(activities=activities, "
        "schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1), "
        "on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(out)), "
        "record_shapes=True, profile_memory=True) as prof:\n"
        "    rc = pytest.main([cfg['test'], '-q'])\n"
        "    torch.npu.synchronize()\n    prof.step()\n"
        "print(f'PYTEST_RC={repr(rc)}')\nraise SystemExit(rc)\n")
    (base / "runs" / "profile_pytest.py").write_text(profile_script)
    res["profile_rc"], _ = run(
        [sys.executable, str(base / "runs" / "profile_pytest.py"),
         str(Path(sys.argv[1]).resolve()), str(repo), str(base / "profile")],
        "profile.log", repo)

res["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
(base / "result.json").write_text(json.dumps(res, indent=2) + "\n")
print("RESULT_JSON_START")
print(json.dumps(res, sort_keys=True))
print("RESULT_JSON_END")
