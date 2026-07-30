#!/usr/bin/env python3
"""Simple sequential PR runner — one 910bctl invocation per PR, no subprocess trees.
Each PR result saved immediately to local evidence dir + state updated on the spot.
Resumes from batch-state-v5.json. STOP file checked between PRs.
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
LOG_PATH = HERE / "seq_runner.log"
STOP_PATH = HERE / "STOP"
PID_PATH = HERE / "seq_runner.pid"
BASE = "/inspire/sj-ssd/project/qianghuaxuexi/s26043/npu-kernelwiki-validation"
RUNNER = f"{BASE}/runner_v2.py"

SKIP_TESTS = ("test_intranode","test_low_latency","test_dispatch_ffn_combine",
              "test_normal_and_low_latency","test_internode","test_fused_deep_moe",
              "test_shmem_intranode")
NO_COMPAT = {1,5,15,60,66,71,77,95}

PID_PATH.write_text(str(os.getpid()))

def log(msg):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with LOG_PATH.open("a") as f: f.write(line+"\n")

log("=== SEQ RUNNER ===")

# Preflight
cp = subprocess.run([CTL, "deep"], capture_output=True, text=True, timeout=120)
if "ok: True" not in cp.stdout:
    log("FATAL: deep check failed")
    sys.exit(1)
cp = subprocess.run([CTL, "run",
    "source /usr/local/Ascend/ascend-toolkit/set_env.sh && "
    "unset ASCEND_VISIBLE_DEVICES ASCEND_RT_VISIBLE_DEVICES && "
    "python3 -c 'import torch; import torch_npu; assert torch.npu.is_available(); print(\"OK\")'"],
    capture_output=True, text=True, timeout=120)
if "OK" not in cp.stdout:
    log("FATAL: NPU smoke failed")
    sys.exit(1)
log("gate OK")

# Load state
q = json.loads(QUEUE_PATH.read_text())
s = json.loads(STATE_PATH.read_text())
done_entries = s.get("completed",[]) + s.get("failed",[]) + s.get("skipped",[])
done = {e.get("pr") for e in done_entries}

candidates = [x for x in q["candidates"] if x.get("validation_scope")=="correctness_candidate" and x["pr"] not in done]
log(f"{len(candidates)} remaining correctness candidates")

for idx, item in enumerate(candidates):
    pr = int(item["pr"])
    sha = str(item["merge_sha"])
    test = item.get("test","")
    scope = item.get("validation_scope","correctness")
    mapping = item.get("test_mapping","")

    if STOP_PATH.exists():
        log("STOP found — exiting")
        break

    # Skip multi-device
    if test and any(t in test for t in SKIP_TESTS):
        entry = {"pr":pr,"merge_sha":sha,"category":scope,"status":"multi_device_required",
                 "failure_gate":"multi_device_required","test":test,"attempt":"",
                 "compat_patch":"","started_at":"","finished_at":""}
        s.setdefault("failed",[]).append(entry)
        json.dump(s, open(STATE_PATH,"w"), indent=2, sort_keys=True)
        log(f"[{idx+1}/{len(candidates)}] PR-{pr} → multi_device_required")
        continue

    ts_str = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    remote_dir = f"{BASE}/work/PR-{pr}/{ts_str}"
    compat = [] if pr in NO_COMPAT else ["gcc_acl"]

    cfg = {"pr":pr,"merge_sha":sha,"test":test,"test_mapping":mapping,
           "validation_scope":scope,"compatibility":compat,"run_dir":remote_dir,
           "benchmark_repeats":3,"profile_mode":"pytest"}
    cfg_b64 = base64.b64encode(json.dumps(cfg).encode()).decode()

    cmd = (f"source /usr/local/Ascend/ascend-toolkit/set_env.sh && "
           f"unset ASCEND_VISIBLE_DEVICES ASCEND_RT_VISIBLE_DEVICES && "
           f"mkdir -p {remote_dir}/logs {remote_dir}/runs {remote_dir}/outputs {remote_dir}/profile && "
           f"echo {cfg_b64} | base64 -d > {remote_dir}/config.json && "
           f"python3 {RUNNER} {remote_dir}/config.json")

    log(f"[{idx+1}/{len(candidates)}] PR-{pr} sha={sha[:12]} test={test}")

    t0 = time.monotonic()
    try:
        cp = subprocess.run([CTL, "run", cmd], capture_output=True, text=True, timeout=900)
        dt = time.monotonic() - t0
        out = cp.stdout
        result = {}
        if "RESULT_JSON_START" in out:
            si = out.index("RESULT_JSON_START") + 18
            ei = out.index("RESULT_JSON_END") if "RESULT_JSON_END" in out else len(out)
            result = json.loads(out[si:ei].strip())
    except subprocess.TimeoutExpired:
        dt = time.monotonic() - t0
        result = {"dispatch_timeout": True, "elapsed": dt}
    except Exception as exc:
        dt = time.monotonic() - t0
        result = {"dispatch_exception": str(exc), "elapsed": dt}

    # Save locally
    local_attempt = LOCAL_VAL / f"PR-{pr}" / "evidence" / f"attempt-{ts_str}"
    local_attempt.mkdir(parents=True, exist_ok=True)
    (local_attempt / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    (LOCAL_VAL / f"PR-{pr}" / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")

    entry = {"pr":pr,"merge_sha":sha,"category":scope,"attempt":ts_str,
             "test":test,"compat_patch":"gcc_acl" if compat else "none",
             "started_at":datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
             "finished_at":result.get("finished_at",datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))}

    stop_reason = result.get("stop_reason")
    if stop_reason is None and result.get("build_rc")==0 and result.get("import_rc")==0 and result.get("correctness_rc")==0 and result.get("benchmark_rc")==0 and result.get("profile_rc")==0:
        entry.update({"status":"FULL_PASS","failure_gate":""})
        s.setdefault("completed",[]).append(entry)
        log(f"  → FULL_PASS ({dt:.0f}s)")
    elif stop_reason is None and result.get("build_rc")==0 and result.get("import_rc")==0 and result.get("correctness_rc")==0:
        entry.update({"status":"CORRECTNESS_OK","failure_gate":""})
        s.setdefault("completed",[]).append(entry)
        log(f"  → CORRECTNESS_OK ({dt:.0f}s)")
    elif stop_reason:
        gate = result.get("gate_failure","unknown")
        status = (gate or stop_reason).replace(" ","_")
        entry.update({"status":status,"failure_gate":gate})
        s.setdefault("failed",[]).append(entry)
        log(f"  → {status} ({dt:.0f}s)")
    else:
        entry.update({"status":"partial","failure_gate":""})
        s.setdefault("completed",[]).append(entry)
        log(f"  → partial ({dt:.0f}s)")

    s["total_processed"] = s.get("total_processed",0) + 1
    s["current_pr"] = pr
    json.dump(s, open(STATE_PATH,"w"), indent=2, sort_keys=True)

    total = len(s.get("completed",[])) + len(s.get("failed",[])) + len(s.get("skipped",[]))
    if total % 5 == 0:
        c = len(s.get("completed",[])); f = len(s.get("failed",[])); sk = len(s.get("skipped",[]))
        log(f"  SUMMARY: {c} pass / {f} fail / {sk} skip ({total} total)")

log("=== DONE ===")
