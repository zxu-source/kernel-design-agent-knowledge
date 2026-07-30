#!/usr/bin/env python3
"""Persistent 910B batch validation supervisor — V5.2 minimal, no download.

Key design:
- Uses verified base64 config + 910bctl run pattern (proven working in V4)
- Parses RESULT_JSON markers from stdout — no file download
- Saves parsed result to local canonical evidence dir
- STOP file checked between PRs
- Resumes from state + local evidence
"""
import base64, json, os, signal, subprocess, sys, time, traceback
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
CORPUS = HERE.parents[3] / "npu-kernelwiki"
LOCAL_VAL = CORPUS / "validation" / "ascend-910b"
CTL = "/home/kirin_14379/projects/ai4qz/scripts/910bctl"
QUEUE_PATH = HERE / "full-execution-queue.json"
STATE_PATH = HERE / "batch-state-v5.json"
HEARTBEAT_PATH = HERE / "heartbeat.json"
SUMMARY_JSON = HERE / "batch-summary-v5.json"
SUMMARY_CSV = HERE / "batch-summary-v5.csv"
STOP_PATH = HERE / "STOP"
PID_PATH = HERE / "supervisor.pid"
LOG_PATH = HERE / "supervisor.log"
REMOTE_BASE = "/inspire/sj-ssd/project/qianghuaxuexi/s26043/npu-kernelwiki-validation"
REMOTE_RUNNER = f"{REMOTE_BASE}/runner_v2.py"
DISK_SPACE_GATE_GB = 1.0  # minimum free GB before pausing

def log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with LOG_PATH.open("a") as f:
        f.write(line + "\n")

def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {"started_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "phase": "correctness_candidates", "completed": [], "failed": [],
            "skipped": [], "current_pr": None, "total_processed": 0}

def save_state(state: dict) -> None:
    state["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")

def heartbeat(state: dict) -> None:
    HEARTBEAT_PATH.write_text(json.dumps({
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pid": os.getpid(), "phase": state.get("phase"),
        "current_pr": state.get("current_pr"),
        "total_processed": state.get("total_processed", 0),
    }, indent=2) + "\n")

def update_summary(state: dict) -> None:
    rows = []
    for entry in (state.get("completed", []) + state.get("failed", []) +
                  state.get("skipped", [])):
        rows.append({k: entry.get(k, "") for k in
            ["pr","merge_sha","category","attempt","status","failure_gate",
             "compat_patch","test","started_at","finished_at"]})
    SUMMARY_JSON.write_text(json.dumps(rows, indent=2) + "\n")
    keys = ["pr","merge_sha","category","attempt","status","failure_gate",
            "compat_patch","test","started_at","finished_at"]
    with SUMMARY_CSV.open("w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(k,"")).replace(",",";") for k in keys) + "\n")

def preflight_gate() -> bool:
    log("=== PREFLIGHT ===")
    cp = subprocess.run([CTL, "deep"], capture_output=True, text=True, timeout=120)
    checks = {}
    for line in cp.stdout.splitlines():
        for f in ("ok","xsrf_found","contents_api_ok","probe_exit_code"):
            if line.strip().startswith(f"{f}:"):
                checks[f] = line.split(":",1)[1].strip()
    log(f"  deep: ok={checks.get('ok')} xsrf={checks.get('xsrf_found')} contents={checks.get('contents_api_ok')} probe={checks.get('probe_exit_code')}")
    if not (checks.get("ok") == "True" and checks.get("xsrf_found") == "True" and
            checks.get("contents_api_ok") == "True" and checks.get("probe_exit_code") == "0"):
        return False
    cp = subprocess.run([CTL, "run",
        "source /usr/local/Ascend/ascend-toolkit/set_env.sh && "
        "unset ASCEND_VISIBLE_DEVICES ASCEND_RT_VISIBLE_DEVICES && "
        "python3 -c 'import torch; import torch_npu; "
        "assert torch.npu.is_available(); print(\"SMOKE_OK\")'"],
        capture_output=True, text=True, timeout=120)
    ok = "SMOKE_OK" in cp.stdout
    log(f"  NPU smoke: {'OK' if ok else 'FAIL'}")
    return ok

def check_bundle(pr: int, sha: str) -> bool:
    page = CORPUS / "sources/prs/sgl-kernel-npu" / f"PR-{pr}.md"
    bundle = CORPUS / "artifacts/prs/sgl-kernel-npu" / f"PR-{pr}"
    if not page.exists():
        return False
    text = page.read_text()
    if f'merge_sha: "{sha}"' not in text:
        return False
    for fn in ("diff.patch", "PROVENANCE.yaml"):
        if not (bundle / fn).is_file():
            return False
    return any(p.is_file() for p in (bundle / "key-files").rglob("*"))

def is_already_done(pr: int) -> bool:
    ev_dir = LOCAL_VAL / f"PR-{pr}" / "evidence"
    for attempt_dir in sorted(ev_dir.glob("attempt-*"), reverse=True):
        rf = attempt_dir / "result.json"
        if rf.exists():
            try:
                r = json.loads(rf.read_text())
                if (r.get("stop_reason") is None and
                    r.get("build_rc") == 0 and r.get("import_rc") == 0 and
                    r.get("correctness_rc") == 0 and r.get("benchmark_rc") == 0 and
                    r.get("profile_rc") == 0):
                    return True
            except: pass
    return False

def classify_failure(result: dict) -> str:
    gate = result.get("gate_failure", "")
    stop = result.get("stop_reason", "")
    if "pre-bootstrap" in stop or "CMakeLists.txt missing" in stop:
        return "prebootstrap_no_cmake"
    if "test_missing" in gate or "test file" in stop:
        return "test_missing_at_target_sha"
    return (gate or stop or "unknown").replace(" ", "_")

# ── Core dispatch ───────────────────────────────────────────────────

# Tests known to hang on single-card 910B (DeepEP intranode/low_latency need 2+ devices)
SINGLE_CARD_SKIP_TESTS = (
    "test_intranode", "test_low_latency", "test_dispatch_ffn_combine",
    "test_normal_and_low_latency", "test_internode", "test_fused_deep_moe",
)

def run_one_pr(item: dict, attempt_ts: str) -> dict | None:
    pr = int(item["pr"])
    sha = str(item["merge_sha"])
    test = item.get("test")
    scope = item.get("validation_scope", "correctness")
    mapping = item.get("test_mapping", "")

    # Detect single-card-incompatible tests early
    if test and any(skip_token in test for skip_token in SINGLE_CARD_SKIP_TESTS):
        return {
            "pr": pr, "merge_sha": sha, "test": test,
            "stop_reason": "test requires multi-device (DeepEP intranode/low_latency); single-card 910B skip",
            "gate_failure": "multi_device_required",
            "validation_scope": scope,
            "dispatched_at": attempt_ts, "attempt": 0,
        }

    remote_dir = f"{REMOTE_BASE}/work/PR-{pr}/{attempt_ts}"

    compat = []
    if pr not in (1, 5, 15, 60, 66, 71, 77, 95):
        compat = ["gcc_acl"]

    cfg = {"pr": pr, "merge_sha": sha, "test": test,
           "test_mapping": mapping, "validation_scope": scope,
           "compatibility": compat, "run_dir": remote_dir,
           "benchmark_repeats": 3, "profile_mode": "pytest"}
    cfg_b64 = base64.b64encode(json.dumps(cfg).encode()).decode()

    cmd = (
        f"source /usr/local/Ascend/ascend-toolkit/set_env.sh && "
        f"unset ASCEND_VISIBLE_DEVICES ASCEND_RT_VISIBLE_DEVICES && "
        f"mkdir -p {remote_dir}/logs {remote_dir}/runs {remote_dir}/outputs {remote_dir}/profile && "
        f"echo {cfg_b64} | base64 -d > {remote_dir}/config.json && "
        f"python3 {REMOTE_RUNNER} {remote_dir}/config.json"
    )

    timeout = 900 if test and "deep" not in test and "test_fused" not in test else 300

    for attempt in range(2):
        try:
            cp = subprocess.run([CTL, "run", cmd], capture_output=True,
                              text=True, timeout=timeout)
            out = cp.stdout
            if "RESULT_JSON_START" in out:
                s = out.index("RESULT_JSON_START") + 18
                e = out.index("RESULT_JSON_END") if "RESULT_JSON_END" in out else len(out)
                result = json.loads(out[s:e].strip())
                result["dispatched_at"] = attempt_ts
                result["attempt"] = attempt
                return result
            # Check if runner produced output at all (even without markers)
            if "runner_v2.py" not in out and attempt == 1:
                log(f"  no runner output — possible runner failure")
        except subprocess.TimeoutExpired:
            log(f"  timeout {attempt+1} — killing stuck subprocess")
            # Kill any remaining ai4qz child to free resources
            subprocess.run(["pkill", "-f", f"ai4qz.*PR-{pr}"], capture_output=True)
        except Exception as exc:
            log(f"  exception attempt {attempt+1}: {exc}")
        time.sleep(10)
    return {"pr": pr, "dispatch_failed": True, "stop_reason": "dispatch failed after 2 attempts"}

def main() -> int:
    PID_PATH.write_text(str(os.getpid()))
    log("=== SUPERVISOR V5.2 STARTED ===")

    if not preflight_gate():
        log("FATAL: preflight failed")
        state = load_state(); state["infra_blocked"] = True; save_state(state)
        return 1

    queue = json.loads(QUEUE_PATH.read_text())
    all_c = queue["candidates"]
    correctness = [x for x in all_c if x.get("validation_scope") == "correctness_candidate"]
    heuristic = [x for x in all_c if x.get("validation_scope") == "coverage_probe"]
    reference = [x for x in all_c if x.get("state") == "reference_required"]
    log(f"Queue: {len(correctness)} correctness + {len(heuristic)} heuristic + {len(reference)} reference")

    state = load_state()
    save_state(state); heartbeat(state)

    # Build skip set from state + existing evidence
    already = set()
    for e in (state.get("completed",[]) + state.get("failed",[]) + state.get("skipped",[])):
        already.add(e.get("pr"))
    for d in LOCAL_VAL.glob("PR-*"):
        try:
            p = int(d.name[3:])
            if is_already_done(p): already.add(p)
        except: pass
    log(f"Already done/skipped: {len(already)} PRs")

    phases = [("correctness_candidates", correctness),
              ("coverage_probes", heuristic),
              ("reference_required", reference)]

    for phase_name, phase_items in phases:
        if STOP_PATH.exists(): break
        state["phase"] = phase_name; save_state(state)
        log(f"\n=== PHASE: {phase_name} ({len(phase_items)} items) ===")

        for idx, item in enumerate(phase_items):
            pr = int(item["pr"])
            sha = str(item["merge_sha"])
            if pr in already: continue
            if STOP_PATH.exists():
                log("STOP file found — exiting"); return 0

            state["current_pr"] = pr
            state["total_processed"] = state.get("total_processed", 0) + 1
            save_state(state); heartbeat(state)

            test_str = item.get("test", "none")
            scope = item.get("validation_scope", "correctness")
            log(f"\n[{idx+1}/{len(phase_items)}] PR-{pr} [{scope}] sha={sha[:12]} test={test_str}")

            if not check_bundle(pr, sha):
                entry = {"pr":pr,"merge_sha":sha,"category":scope,"status":"bundle_incomplete",
                         "failure_gate":"bundle","test":test_str,"attempt":"","compat_patch":"",
                         "started_at":"","finished_at":""}
                state["skipped"].append(entry); already.add(pr)
                save_state(state); update_summary(state); continue

            attempt_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            entry = {"pr":pr,"merge_sha":sha,"category":scope,"attempt":attempt_ts,
                     "test":test_str,
                     "compat_patch":"gcc_acl" if pr not in (1,5,15,60,66,71,77,95) else "none",
                     "started_at":datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}

            result = run_one_pr(item, attempt_ts)
            if not result or result.get("dispatch_failed"):
                entry.update({"status":"dispatch_failed","failure_gate":"dispatch",
                              "finished_at":datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")})
                state["failed"].append(entry); already.add(pr)
                save_state(state); update_summary(state); heartbeat(state); continue

            entry["finished_at"] = result.get("finished_at",
                datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))

            # Save result locally
            local_attempt = LOCAL_VAL / f"PR-{pr}" / "evidence" / f"attempt-{attempt_ts}"
            local_attempt.mkdir(parents=True, exist_ok=True)
            (local_attempt / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            (LOCAL_VAL / f"PR-{pr}" / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

            stop = result.get("stop_reason")
            if stop is None:
                if (result.get("build_rc") == 0 and result.get("import_rc") == 0 and
                    result.get("correctness_rc") == 0 and result.get("benchmark_rc") == 0 and
                    result.get("profile_rc") == 0):
                    status = "FULL_PASS"
                    entry.update({"status": status, "failure_gate": ""})
                    state["completed"].append(entry)
                    log(f"  → FULL_PASS (build+import+correctness+bench+profile)")
                elif result.get("build_rc") == 0 and result.get("import_rc") == 0 and result.get("correctness_rc") == 0:
                    status = "CORRECTNESS_PASS"
                    entry.update({"status": status, "failure_gate": ""})
                    state["completed"].append(entry)
                    log(f"  → CORRECTNESS_PASS")
                else:
                    status = "PARTIAL"
                    entry.update({"status": status, "failure_gate": ""})
                    state["completed"].append(entry)
                    log(f"  → PARTIAL")
            else:
                failure = classify_failure(result)
                entry.update({"status": failure, "failure_gate": result.get("gate_failure","unknown")})
                if failure in ("test_missing_at_target_sha", "prebootstrap_no_cmake"):
                    state["skipped"].append(entry)
                else:
                    state["failed"].append(entry)
                log(f"  → {failure}")

            already.add(pr); save_state(state); update_summary(state)

            if state["total_processed"] % 10 == 0:
                c = len(state.get("completed",[])); f = len(state.get("failed",[]))
                s = len(state.get("skipped",[]))
                log(f"  ── SUMMARY: {c} pass / {f} fail / {s} skip ──")

            heartbeat(state)

    state["current_pr"] = None
    state["finished_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    save_state(state); update_summary(state); heartbeat(state)
    log("=== SUPERVISOR V5.2 FINISHED ===")
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log("SIGINT"); sys.exit(0)
    except Exception:
        log(f"FATAL: {traceback.format_exc()}")
        sys.exit(1)
