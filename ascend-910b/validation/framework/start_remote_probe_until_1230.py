#!/usr/bin/env python3
"""Start a 910B-resident, deadline-bounded heuristic probe supervisor."""
from __future__ import annotations

import base64
import gzip
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
QUEUE = HERE / "full-execution-queue.json"
RUNNER = HERE / "runner_v2.py"
CTL = "/home/kirin_14379/projects/ai4qz/scripts/910bctl"
REMOTE = "/inspire/sj-ssd/project/qianghuaxuexi/s26043/npu-kernelwiki-validation/background-probes-until-1230-v2"

queue = json.loads(QUEUE.read_text())["candidates"]
candidates = [x for x in queue if x.get("validation_scope") == "coverage_probe"]
remote_script = f'''#!/usr/bin/env python3
import base64, json, os, subprocess, sys, time
from datetime import datetime, timedelta, timezone
from pathlib import Path

BASE=Path({REMOTE!r})
DEADLINE=datetime(2026,7,29,12,30,tzinfo=timezone(timedelta(hours=8)))
CANDIDATES=json.loads(base64.b64decode({base64.b64encode(json.dumps(candidates).encode()).decode()!r}))
RUNNER=base64.b64decode({base64.b64encode(RUNNER.read_bytes()).decode()!r})
MULTI=("test_intranode","test_low_latency","test_dispatch_ffn_combine","test_normal_and_low_latency","test_internode","test_fused_deep_moe","test_shmem_intranode")
BASE.mkdir(parents=True,exist_ok=True)
state_path=BASE/"state.json"; log_path=BASE/"supervisor.log"; hb_path=BASE/"heartbeat.json"
state=json.loads(state_path.read_text()) if state_path.exists() else {{"started_at":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"results":[]}}
done={{x["pr"] for x in state["results"]}}
def save():
    tmp=BASE/"state.tmp"; tmp.write_text(json.dumps(state,indent=2)+"\\n"); tmp.replace(state_path)
    hb_path.write_text(json.dumps({{"updated_at":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"processed":len(state["results"]),"deadline":DEADLINE.isoformat()}},indent=2)+"\\n")
def log(s):
    line=f"[{{time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}}] {{s}}"; print(line,flush=True); open(log_path,"a").write(line+"\\n")
log(f"START candidates={{len(CANDIDATES)}} deadline={{DEADLINE.isoformat()}}")
for item in CANDIDATES:
    if datetime.now(DEADLINE.tzinfo) >= DEADLINE: log("DEADLINE"); break
    pr=int(item["pr"])
    if pr in done: continue
    test=item.get("test") or ""
    rec={{"pr":pr,"merge_sha":item["merge_sha"],"category":"heuristic_probe","test":test,"started_at":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())}}
    if not test or any(x in test for x in MULTI):
        rec.update(status="probe_multi_device_or_no_test",reason="single-card probe not applicable")
        state["results"].append(rec); save(); log(f"PR-{{pr}} {{rec['status']}}"); continue
    attempt=time.strftime("%Y%m%dT%H%M%SZ",time.gmtime()); run=BASE/f"PR-{{pr}}"/attempt; run.mkdir(parents=True,exist_ok=True)
    cfg={{"pr":pr,"merge_sha":item["merge_sha"],"test":test,"test_mapping":item.get("test_mapping","heuristic_name_overlap"),"validation_scope":"coverage_probe","compatibility":["gcc_acl"],"run_dir":str(run),"benchmark_repeats":3,"profile_mode":"pytest"}}
    (run/"config.json").write_text(json.dumps(cfg)+"\\n"); (run/"runner.py").write_bytes(RUNNER)
    try:
        with (run/"dispatch.log").open("w") as f: p=subprocess.run([sys.executable,str(run/"runner.py"),str(run/"config.json")],stdout=f,stderr=subprocess.STDOUT,text=True,timeout=900)
        result=json.loads((run/"result.json").read_text()) if (run/"result.json").is_file() else {{"dispatch_returncode":p.returncode}}
    except subprocess.TimeoutExpired: result={{"dispatch_timeout":True}}
    except Exception as e: result={{"dispatch_exception":str(e)}}
    rec["result_path"]=str((run/"result.json").relative_to(BASE))
    if all(result.get(k)==0 for k in ("build_rc","import_rc","correctness_rc","benchmark_rc","profile_rc")):
        rec["status"]="probe_passed"
    else:
        rec["status"]="probe_failed"; rec["failure_gate"]=result.get("gate_failure",result.get("stop_reason","dispatch"))
    rec["finished_at"]=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()); state["results"].append(rec); save(); log(f"PR-{{pr}} {{rec['status']}}")
state["finished_at"]=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()); save(); log("FINISHED")
'''
generated = HERE / "generated_remote_probe_until_1230.py"
generated.write_text(remote_script)
mkdir = subprocess.run([CTL, "run", f"mkdir -p {REMOTE}"], text=True, capture_output=True, timeout=120)
if mkdir.returncode:
    print(mkdir.stdout); print(mkdir.stderr, file=sys.stderr); raise SystemExit(mkdir.returncode)
payload = base64.b64encode(gzip.compress(remote_script.encode())).decode()
command = (
    f"echo {payload} | base64 -d | gzip -d > {REMOTE}/supervisor.py; "
    f"source /usr/local/Ascend/ascend-toolkit/set_env.sh; unset ASCEND_VISIBLE_DEVICES ASCEND_RT_VISIBLE_DEVICES; "
    f"nohup /opt/mamba/bin/python3 -u {REMOTE}/supervisor.py > {REMOTE}/supervisor-stdout.log 2>&1 < /dev/null & "
    f"echo $! > {REMOTE}/supervisor.pid; sleep 2; cat {REMOTE}/supervisor.pid; ps -p $(cat {REMOTE}/supervisor.pid) -o pid=,stat=,etime=,cmd="
)
cp = subprocess.run([CTL, "run", command], text=True, capture_output=True, timeout=120)
print(cp.stdout)
print(cp.stderr, file=sys.stderr)
raise SystemExit(cp.returncode)
