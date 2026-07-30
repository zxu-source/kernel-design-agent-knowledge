#!/usr/bin/env python3
"""Nightly correctness-candidate replay from PR-310 onward.
Lean design: one shot per PR via 910bctl, result saved locally, no downloads, no retries.
Resumes from batch-state-v5.json.
"""
import base64, json, os, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
CORPUS = HERE.parents[3] / "npu-kernelwiki"
LOCAL_VAL = CORPUS / "validation" / "ascend-910b"
CTL = "/home/kirin_14379/projects/ai4qz/scripts/910bctl"
QUEUE_PATH = HERE / "full-execution-queue.json"
STATE_PATH = HERE / "batch-state-v5.json"
HB_PATH = HERE / "heartbeat.json"
LOG_PATH = HERE / "nightly_replay.log"
STOP_PATH = HERE / "STOP"
PID_PATH = HERE / "nightly_replay.pid"
BASE = "/inspire/sj-ssd/project/qianghuaxuexi/s26043/npu-kernelwiki-validation"
RUNNER = f"{BASE}/runner_v2.py"

SKIP_TESTS = ("test_intranode","test_low_latency","test_dispatch_ffn_combine",
              "test_normal_and_low_latency","test_internode","test_fused_deep_moe",
              "test_shmem_intranode")

def log(msg):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with LOG_PATH.open("a") as f: f.write(line+"\n")

PID_PATH.write_text(str(os.getpid()))
LOG_PATH.write_text("")
log("=== NIGHTLY REPLAY STARTED ===")

# Gate
cp = subprocess.run([CTL, "deep"], capture_output=True, text=True, timeout=120)
if "ok: True" not in cp.stdout:
    log("FATAL: gate failed")
    sys.exit(1)
cp = subprocess.run([CTL, "run",
    "source /usr/local/Ascend/ascend-toolkit/set_env.sh && "
    "unset ASCEND_VISIBLE_DEVICES ASCEND_RT_VISIBLE_DEVICES && "
    "python3 -c 'import torch; import torch_npu; assert torch.npu.is_available(); print(\"SMOKE_OK\")'"],
    capture_output=True, text=True, timeout=120)
if "SMOKE_OK" not in cp.stdout:
    log("FATAL: NPU smoke failed")
    sys.exit(1)
log("gate OK")

# Load queue
q = json.loads(QUEUE_PATH.read_text())
correctness = [x for x in q["candidates"] if x.get("validation_scope") == "correctness_candidate"]
log(f"Total correctness: {len(correctness)}")

s = json.loads(STATE_PATH.read_text())
already = set()
for e in (s.get("completed",[]) + s.get("failed",[]) + s.get("skipped",[])):
    already.add(e.get("pr"))
log(f"Already processed: {len(already)}")

for idx, item in enumerate(correctness):
    pr = int(item["pr"])
    if pr in already:
        continue
    if STOP_PATH.exists():
        log("STOP found — exiting")
        break

    sha = str(item["merge_sha"])
    test = item.get("test","")
    scope = item.get("validation_scope","correctness")

    if test and any(t in test for t in SKIP_TESTS):
        entry = {"pr":pr,"merge_sha":sha,"category":scope,"status":"multi_device_required",
                 "failure_gate":"multi_device_required","test":test,
                 "attempt":"","compat_patch":"",
                 "started_at":"","finished_at":""}
        s["failed"].append(entry)
        already.add(pr)
        json.dump(s, open(STATE_PATH,"w"), indent=2)
        log(f"[{idx+1}/{len(correctness)}] PR-{pr} → multi_device_required (skip)")
        continue

    attempt_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    remote_dir = f"{BASE}/work/PR-{pr}/{attempt_ts}"
    compat = ["gcc_acl"] if pr not in (1,5,15,60,66,71,77,95) else []

    cfg = {"pr":pr,"merge_sha":sha,"test":test,
           "test_mapping":item.get("test_mapping",""),
           "validation_scope":scope,"compatibility":compat,
           "run_dir":remote_dir,
           "benchmark_repeats":3,
           "profile_mode":"pytest"}
    cfg_b64 = base64.b64encode(json.dumps(cfg).encode()).decode()

    cmd = (f"source /usr/local/Ascend/ascend-toolkit/set_env.sh && "
           f"unset ASCEND_VISIBLE_DEVICES ASCEND_RT_VISIBLE_DEVICES && "
           f"mkdir -p {remote_dir}/logs {remote_dir}/runs {remote_dir}/outputs {remote_dir}/profile && "
           f"echo {cfg_b64} | base64 -d > {remote_dir}/config.json && "
           f"python3 {RUNNER} {remote_dir}/config.json")

    log(f"[{idx+1}/{len(correctness)}] PR-{pr} sha={sha[:12]} test={test}")
    entry = {"pr":pr,"merge_sha":sha,"category":scope,"attempt":attempt_ts,
             "test":test,"compat_patch":"gcc_acl" if compat else "none",
             "started_at":datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}

    timeout = 900
    try:
        cp = subprocess.run([CTL, "run", cmd], capture_output=True, text=True, timeout=timeout)
        out = cp.stdout
        result = {}
        if "RESULT_JSON_START" in out:
            si = out.index("RESULT_JSON_START") + 18
            ei = out.index("RESULT_JSON_END") if "RESULT_JSON_END" in out else len(out)
            result = json.loads(out[si:ei].strip())
        else:
            result = {"dispatch_failed": True, "rc": cp.returncode, "out_tail": out[-500:]}
    except subprocess.TimeoutExpired:
        result = {"dispatch_timeout": True}
    except Exception as exc:
        result = {"dispatch_exception": str(exc)}

    # Save locally
    local_attempt = LOCAL_VAL / f"PR-{pr}" / "evidence" / f"attempt-{attempt_ts}"
    local_attempt.mkdir(parents=True, exist_ok=True)
    (local_attempt / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    (LOCAL_VAL / f"PR-{pr}" / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")

    entry["finished_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    stop = result.get("stop_reason")
    if stop is None and result.get("build_rc") == 0 and result.get("import_rc") == 0 and result.get("correctness_rc") == 0 and result.get("benchmark_rc") == 0 and result.get("profile_rc") == 0:
        status = "FULL_PASS"
        entry.update({"status": status, "failure_gate": ""})
        s["completed"].append(entry)
        log(f"  → FULL_PASS")
    elif result.get("build_rc") == 0 and result.get("import_rc") == 0 and result.get("correctness_rc") == 0:
        status = "CORRECTNESS_OK"
        entry.update({"status": status, "failure_gate": ""})
        s["completed"].append(entry)
        log(f"  → CORRECTNESS_OK")
    elif stop:
        gate = result.get("gate_failure","unknown")
        entry.update({"status": (gate or stop).replace(" ","_"), "failure_gate": gate})
        if "pre-bootstrap" in str(stop) or "test_missing" in str(gate):
            s["skipped"].append(entry)
        else:
            s["failed"].append(entry)
        log(f"  → {(gate or stop).replace(' ','_')}")
    else:
        entry.update({"status":"partial","failure_gate":""})
        s["completed"].append(entry)
        log(f"  → partial")

    s["total_processed"] = s.get("total_processed",0) + 1
    s["current_pr"] = pr
    already.add(pr)
    json.dump(s, open(STATE_PATH,"w"), indent=2)
    HB_PATH.write_text(json.dumps({
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pid": os.getpid(), "phase": "correctness_candidates",
        "current_pr": pr, "total_processed": s.get("total_processed",0),
    }, indent=2)+"\n")

    if s["total_processed"] % 10 == 0:
        c = len(s.get("completed",[])); f = len(s.get("failed",[]))
        sk = len(s.get("skipped",[]))
        log(f"  SUMMARY: {c} pass / {f} fail / {sk} skip")

log("=== NIGHTLY REPLAY FINISHED ===")
