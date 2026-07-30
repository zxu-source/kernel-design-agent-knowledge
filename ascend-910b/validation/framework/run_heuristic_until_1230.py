#!/usr/bin/env python3
"""Time-bounded single-NPU heuristic probe runner with immediate evidence pullback."""
from __future__ import annotations

import base64
import json
import subprocess
import tarfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[3]
CORPUS = WORKSPACE / "npu-kernelwiki"
VALIDATION = CORPUS / "validation/ascend-910b"
QUEUE = HERE / "full-execution-queue.json"
RUNNER = HERE / "runner_v2.py"
STATE = HERE / "heuristic-until-1230-state.json"
HEARTBEAT = HERE / "heuristic-until-1230-heartbeat.json"
LOG = HERE / "heuristic-until-1230.log"
STOP = HERE / "STOP"
CTL = "/home/kirin_14379/projects/ai4qz/scripts/910bctl"
AI4QZ = "/home/kirin_14379/projects/ai4qz/.venv/bin/ai4qz"
CONFIG = "/home/kirin_14379/projects/ai4qz/configs/notebooks-910b.yaml"
TARGET = "ascend_910b"
REMOTE_BASE = "/inspire/sj-ssd/project/qianghuaxuexi/s26043/npu-kernelwiki-validation/work"
DEADLINE = datetime(2026, 7, 29, 12, 30, tzinfo=timezone(timedelta(hours=8)))
MULTI = ("test_intranode", "test_low_latency", "test_dispatch_ffn_combine", "test_normal_and_low_latency", "test_internode", "test_fused_deep_moe", "test_shmem_intranode")


def now() -> datetime:
    return datetime.now(timezone.utc).astimezone(DEADLINE.tzinfo)


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def log(message: str) -> None:
    line = f"[{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}] {message}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def save(data: dict) -> None:
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    tmp.replace(STATE)
    HEARTBEAT.write_text(json.dumps({"updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "deadline": DEADLINE.isoformat(), "processed": len(data["results"])}, indent=2) + "\n")


def bundle_ok(pr: int, sha: str) -> None:
    page = CORPUS / "sources/prs/sgl-kernel-npu" / f"PR-{pr}.md"
    bundle = CORPUS / "artifacts/prs/sgl-kernel-npu" / f"PR-{pr}"
    if f'merge_sha: "{sha}"' not in page.read_text():
        raise ValueError("source SHA mismatch")
    if not (bundle / "diff.patch").is_file() or not (bundle / "PROVENANCE.yaml").is_file():
        raise ValueError("artifact provenance missing")
    if not any(p.is_file() for p in (bundle / "key-files").rglob("*")):
        raise ValueError("key-files empty")


def parse_result(stdout: str) -> dict:
    if "RESULT_JSON_START" not in stdout or "RESULT_JSON_END" not in stdout:
        return {"dispatch_error": "runner result markers missing", "stdout_tail": stdout[-2000:]}
    start = stdout.index("RESULT_JSON_START") + len("RESULT_JSON_START")
    end = stdout.index("RESULT_JSON_END", start)
    return json.loads(stdout[start:end].strip())


def main() -> int:
    if now() >= DEADLINE:
        log("deadline already reached")
        return 0
    gate = subprocess.run([CTL, "deep"], capture_output=True, text=True, timeout=120)
    if "ok: True" not in gate.stdout:
        log("FATAL: deep gate failed")
        return 1
    smoke = subprocess.run([CTL, "run", "source /usr/local/Ascend/ascend-toolkit/set_env.sh && unset ASCEND_VISIBLE_DEVICES ASCEND_RT_VISIBLE_DEVICES && python3 -c 'import torch,torch_npu; assert torch.npu.is_available(); print(\"NPU_OK\")'"], capture_output=True, text=True, timeout=120)
    if "NPU_OK" not in smoke.stdout:
        log("FATAL: NPU smoke failed")
        return 1
    queue = json.loads(QUEUE.read_text())["candidates"]
    state = json.loads(STATE.read_text()) if STATE.exists() else {"started_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "results": []}
    done = {x["pr"] for x in state["results"]}
    candidates = [x for x in queue if x.get("validation_scope") == "coverage_probe" and x["pr"] not in done]
    runner_b64 = base64.b64encode(RUNNER.read_bytes()).decode()
    log(f"START deadline={DEADLINE.isoformat()} queued={len(candidates)}")
    for item in candidates:
        if now() >= DEADLINE:
            log("deadline reached")
            break
        if STOP.exists():
            log("STOP found")
            break
        pr, sha, test = int(item["pr"]), item["merge_sha"], item.get("test") or ""
        record = {"pr": pr, "merge_sha": sha, "category": "heuristic_probe", "test": test, "started_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
        try:
            bundle_ok(pr, sha)
        except Exception as exc:
            record.update(status="probe_preflight_failed", reason=str(exc))
            state["results"].append(record); save(state); continue
        if not test or any(token in test for token in MULTI):
            record.update(status="probe_multi_device_or_no_test", reason="single-card probe not applicable")
            state["results"].append(record); save(state); log(f"PR-{pr}: {record['status']}"); continue
        attempt = stamp()
        remote = f"{REMOTE_BASE}/PR-{pr}/{attempt}"
        cfg = {"pr": pr, "merge_sha": sha, "test": test, "test_mapping": item.get("test_mapping", "heuristic_name_overlap"), "validation_scope": "coverage_probe", "compatibility": ["gcc_acl"], "run_dir": remote, "benchmark_repeats": 3, "profile_mode": "pytest"}
        cfg_b64 = base64.b64encode(json.dumps(cfg).encode()).decode()
        command = (
            "source /usr/local/Ascend/ascend-toolkit/set_env.sh && unset ASCEND_VISIBLE_DEVICES ASCEND_RT_VISIBLE_DEVICES && "
            f"mkdir -p {remote} && echo {runner_b64} | base64 -d > {remote}/runner.py && "
            f"echo {cfg_b64} | base64 -d > {remote}/config.json && python3 {remote}/runner.py {remote}/config.json; "
            f"tar -C {remote} -czf {remote}/evidence.tar.gz logs runs outputs profile result.json config.json 2>/dev/null || true"
        )
        log(f"PR-{pr}: running {test}")
        started = time.monotonic()
        try:
            cp = subprocess.run([CTL, "run", command], capture_output=True, text=True, timeout=900)
            result = parse_result(cp.stdout)
            record["dispatch_returncode"] = cp.returncode
        except subprocess.TimeoutExpired:
            result = {"dispatch_timeout": True}
        except Exception as exc:
            result = {"dispatch_exception": str(exc)}
        record["seconds"] = round(time.monotonic() - started, 2)
        local = VALIDATION / f"PR-{pr}" / "evidence" / f"attempt-{attempt}"
        local.mkdir(parents=True, exist_ok=True)
        (local / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        archive = local / "remote-evidence.tar.gz"
        dl = subprocess.run([AI4QZ, "--config", CONFIG, "--timeout", "600", "download", "--via-terminal", TARGET, f"{remote}/evidence.tar.gz", str(archive)], capture_output=True, text=True, timeout=660)
        record["evidence_download_rc"] = dl.returncode
        if archive.is_file() and archive.stat().st_size > 0:
            record["evidence_archive"] = str(archive.relative_to(VALIDATION))
        else:
            record["evidence_download_error"] = (dl.stderr or dl.stdout)[-1000:]
        if result.get("build_rc") == 0 and result.get("import_rc") == 0 and result.get("correctness_rc") == 0 and result.get("benchmark_rc") == 0 and result.get("profile_rc") == 0:
            record["status"] = "probe_passed"
        else:
            record["status"] = "probe_failed"
            record["failure_gate"] = result.get("gate_failure", result.get("stop_reason", "dispatch"))
        record["finished_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        state["results"].append(record); save(state)
        log(f"PR-{pr}: {record['status']} ({record['seconds']}s)")
    state["finished_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    save(state); log("FINISHED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
