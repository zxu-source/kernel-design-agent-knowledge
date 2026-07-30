#!/usr/bin/env python3
"""Non-interactive 910B validation batch for five archived sgl-kernel-npu PRs.

Every candidate has isolated pristine and compatibility clones.  The script
never alters an archived source/artifact bundle and continues after failures.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(os.environ.get("NPU_BATCH_ROOT", "/inspire/sj-ssd/project/qianghuaxuexi/s26043/npu-kernelwiki-validation/batch-5-20260728-v1"))
REPO_URL = "https://github.com/sgl-project/sgl-kernel-npu.git"
CANDIDATES = [
    {"pr": 155, "sha": "d54224c4d16edddc9c0ea0e1ca08fd51e58fa2f9", "test": "tests/python/sgl_kernel_npu/test_swiglu_quant.py"},
    {"pr": 290, "sha": "fa9a7f62f6ea24a186fbaffa0adc7da560cacbde", "test": "tests/python/sgl_kernel_npu/test_split_qkv_rmsnorm_rope.py"},
    {"pr": 404, "sha": "80b9cd70da2ba5bbc5e6e6f06b4efc167cb07d15", "test": "tests/python/sgl_kernel_npu/test_split_qkv_rmsnorm_rope_pos_cache_half_npu.py"},
    {"pr": 507, "sha": "68946d3f166790f225367f03cac62fc8714e3d18", "test": "tests/python/sgl_kernel_npu/test_swiglu_quant.py"},
    {"pr": 557, "sha": "9765e27532b8f854717eba1d3fb9a0bb97a0c887", "test": "tests/python/sgl_kernel_npu/test_fused_rope_qk_mqa.py"},
]


def run(command: list[str], cwd: Path, log: Path, env: dict[str, str] | None = None) -> tuple[int, float]:
    started = time.monotonic()
    merged = os.environ.copy()
    if env:
        merged.update(env)
    with log.open("w") as stream:
        stream.write("$ " + " ".join(command) + "\n")
        stream.flush()
        result = subprocess.run(command, cwd=cwd, env=merged, stdout=stream, stderr=subprocess.STDOUT, text=True)
        stream.write(f"\n[exit_code={result.returncode}]\n")
    return result.returncode, round(time.monotonic() - started, 3)


def checkout(path: Path, sha: str, logs: Path) -> tuple[int, str]:
    logs.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        rc, _ = run(["git", "clone", "--no-checkout", REPO_URL, str(path)], ROOT, logs / "git-clone.log")
        if rc:
            return rc, ""
    rc, _ = run(["git", "fetch", "--depth=1", "origin", sha], path, logs / "git-fetch.log")
    if rc:
        return rc, ""
    rc, _ = run(["git", "checkout", "--detach", sha], path, logs / "git-checkout.log")
    if rc:
        return rc, ""
    got = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()
    return (0 if got == sha else 1), got


def patch_compat(repo: Path, patch_path: Path) -> tuple[bool, str]:
    root_cmake = repo / "CMakeLists.txt"
    csrc_cmake = repo / "csrc" / "CMakeLists.txt"
    before_root = root_cmake.read_text()
    before_csrc = csrc_cmake.read_text()
    after_root = before_root.replace("-hno-unused-parameter -lno-unused-function ", "")
    acl = "${TORCH_NPU_DIR}/include/third_party/acl/inc"
    after_csrc = before_csrc
    if acl not in after_csrc:
        after_csrc = after_csrc.replace("${TORCH_NPU_DIR}/include", "${TORCH_NPU_DIR}/include\n        ${TORCH_NPU_DIR}/include/third_party/acl/inc", 1)
    if after_root == before_root or after_csrc == before_csrc:
        return False, "expected CMake compatibility context was not found"
    root_cmake.write_text(after_root)
    csrc_cmake.write_text(after_csrc)
    with patch_path.open("w") as stream:
        subprocess.run(["git", "diff", "--", "CMakeLists.txt", "csrc/CMakeLists.txt"], cwd=repo, stdout=stream, check=True, text=True)
    return True, "applied CMake-only GCC/ACL compatibility patch"


def profile_wrapper(path: Path) -> None:
    path.write_text(
        "import os, sys, torch, torch_npu.profiler as profiler, pytest\n"
        "test = sys.argv[1]\ntrace = sys.argv[2]\n"
        "with profiler.profile(activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU], "
        "schedule=profiler.schedule(wait=0, warmup=0, active=1, repeat=1), "
        "on_trace_ready=profiler.tensorboard_trace_handler(trace), record_shapes=True, profile_memory=True) as p:\n"
        "    rc = pytest.main([test, '-q'])\n"
        "    torch.npu.synchronize()\n"
        "    p.step()\n"
        "print('PYTEST_RC=%s' % rc)\n"
        "raise SystemExit(rc)\n"
    )


def validate(candidate: dict[str, object]) -> dict[str, object]:
    pr = int(candidate["pr"])
    sha = str(candidate["sha"])
    test = str(candidate["test"])
    base = ROOT / f"PR-{pr}"
    logs = base / "logs"
    runs = base / "runs"
    outputs = base / "outputs"
    profile = base / "profile"
    for item in (logs, runs, outputs, profile):
        item.mkdir(parents=True, exist_ok=True)
    result: dict[str, object] = {"pr": pr, "sha": sha, "test": test, "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    pristine = base / "pristine" / "sgl-kernel-npu"
    pristine.parent.mkdir(parents=True, exist_ok=True)
    rc, got = checkout(pristine, sha, logs / "pristine")
    result["pristine_checkout_rc"] = rc
    result["pristine_head"] = got
    if rc == 0:
        native_build = pristine / "build"
        rc, _ = run(["cmake", "-S", ".", "-B", str(native_build), "-DBUILD_DEEPEP_MODULE=OFF", "-DSOC_VERSION=Ascend910B2C", "-DASCEND_INCLUDE_DIR=/usr/local/Ascend/cann-9.0.0/include"], pristine, logs / "native-configure.log")
        result["native_configure_rc"] = rc
        if rc == 0:
            rc, _ = run(["cmake", "--build", str(native_build), "--target", "sgl_kernel_npu", "-j2"], pristine, logs / "native-build.log")
        result["native_build_rc"] = rc

    compat = base / "compat" / "sgl-kernel-npu"
    compat.parent.mkdir(parents=True, exist_ok=True)
    rc, got = checkout(compat, sha, logs / "compat")
    result["compat_checkout_rc"] = rc
    result["compat_head"] = got
    if rc != 0:
        result["stop_reason"] = "compat checkout failed"
        return result
    applied, detail = patch_compat(compat, runs / "compat.patch")
    result["compat_patch_applied"] = applied
    result["compat_patch_detail"] = detail
    if not applied:
        result["stop_reason"] = detail
        return result
    build = compat / "build"
    rc, _ = run(["cmake", "-S", ".", "-B", str(build), "-DBUILD_DEEPEP_MODULE=OFF", "-DSOC_VERSION=Ascend910B2C", "-DASCEND_INCLUDE_DIR=/usr/local/Ascend/cann-9.0.0/include"], compat, logs / "compat-configure.log")
    result["compat_configure_rc"] = rc
    if rc != 0:
        result["stop_reason"] = "compat configure failed"
        return result
    rc, _ = run(["cmake", "--build", str(build), "--target", "sgl_kernel_npu", "-j2"], compat, logs / "compat-build.log")
    result["compat_build_rc"] = rc
    if rc != 0:
        result["stop_reason"] = "compat build failed"
        return result

    env = {"PYTHONPATH": "python/sgl_kernel_npu"}
    rc, duration = run([sys.executable, "-m", "pytest", "-q", test], compat, logs / "correctness.log", env)
    result["correctness_rc"] = rc
    result["correctness_seconds"] = duration
    if rc != 0:
        result["stop_reason"] = "upstream correctness failed"
        return result

    samples: list[float] = []
    for index in range(3):
        rc, duration = run([sys.executable, "-m", "pytest", "-q", test], compat, logs / f"benchmark-{index + 1}.log", env)
        samples.append(duration)
        if rc != 0:
            result["benchmark_rc"] = rc
            result["stop_reason"] = "benchmark repeat failed"
            return result
    result["benchmark_rc"] = 0
    result["benchmark_wall_seconds"] = samples
    (outputs / "benchmark-walltime.json").write_text(json.dumps({"kind": "three upstream-test wall-time samples; not an operator performance comparison", "seconds": samples}, indent=2) + "\n")

    wrapper = runs / "profile_upstream_test.py"
    profile_wrapper(wrapper)
    trace = profile / "torch_npu"
    rc, duration = run([sys.executable, str(wrapper), test, str(trace)], compat, logs / "profile.log", env)
    result["profile_rc"] = rc
    result["profile_seconds"] = duration
    result["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return result


def main() -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    results = []
    for candidate in CANDIDATES:
        try:
            result = validate(candidate)
        except Exception as exc:  # retain the batch even on unexpected errors
            result = {"pr": candidate["pr"], "sha": candidate["sha"], "unhandled_exception": repr(exc)}
        results.append(result)
        (ROOT / f"PR-{candidate['pr']}" / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, sort_keys=True), flush=True)
    (ROOT / "batch-results.json").write_text(json.dumps(results, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
