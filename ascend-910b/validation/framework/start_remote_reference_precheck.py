#!/usr/bin/env python3
"""Start a 910B-resident build/import precheck for 28 reference-required PRs."""
from __future__ import annotations

import base64
import gzip
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
QUEUE = HERE / "full-execution-queue.json"
CLASSIFICATION = HERE / "reference-test-classification-v2.json"
RUNNER = HERE / "runner_v2.py"
CTL = "/home/kirin_14379/projects/ai4qz/scripts/910bctl"
REMOTE = "/inspire/sj-ssd/project/qianghuaxuexi/s26043/npu-kernelwiki-validation/background-reference-precheck"

items = {x["pr"]: x for x in json.loads(CLASSIFICATION.read_text())["items"]}
queue = {x["pr"]: x for x in json.loads(QUEUE.read_text())["candidates"]}
tasks = []
for pr, meta in sorted(items.items()):
    q = queue[pr]
    tasks.append({"pr": pr, "merge_sha": q["merge_sha"], "strategy": meta["verification_strategy"], "category": meta["category"]})

script = f'''#!/usr/bin/env python3
import base64,json,subprocess,sys,time
from pathlib import Path
BASE=Path({REMOTE!r}); BASE.mkdir(parents=True,exist_ok=True)
TASKS=json.loads(base64.b64decode({base64.b64encode(json.dumps(tasks).encode()).decode()!r}))
RUNNER=base64.b64decode({base64.b64encode(RUNNER.read_bytes()).decode()!r})
state_path=BASE/"state.json"; log_path=BASE/"supervisor.log"; hb_path=BASE/"heartbeat.json"
state=json.loads(state_path.read_text()) if state_path.exists() else {{"started_at":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"results":[]}}
done={{x["pr"] for x in state["results"]}}
def save():
 tmp=BASE/"state.tmp"; tmp.write_text(json.dumps(state,indent=2)+"\\n"); tmp.replace(state_path)
 hb_path.write_text(json.dumps({{"updated_at":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"processed":len(state["results"])}},indent=2)+"\\n")
def log(x):
 line=f"[{{time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}}] {{x}}"; print(line,flush=True); open(log_path,"a").write(line+"\\n")
log(f"START tasks={{len(TASKS)}}")
for task in TASKS:
 pr=task["pr"]
 if pr in done: continue
 attempt=time.strftime("%Y%m%dT%H%M%SZ",time.gmtime()); run=BASE/f"PR-{{pr}}"/attempt; run.mkdir(parents=True,exist_ok=True)
 cfg={{"pr":pr,"merge_sha":task["merge_sha"],"test":"","test_mapping":"reference_precheck","validation_scope":"reference_precheck","compatibility":["gcc_acl"],"run_dir":str(run),"benchmark_repeats":0,"profile_mode":"none"}}
 (run/"config.json").write_text(json.dumps(cfg)+"\\n"); (run/"runner.py").write_bytes(RUNNER)
 log(f"PR-{{pr}} {{task['strategy']}}")
 try:
  with (run/"dispatch.log").open("w") as f: p=subprocess.run([sys.executable,str(run/"runner.py"),str(run/"config.json")],stdout=f,stderr=subprocess.STDOUT,text=True,timeout=900)
  result=json.loads((run/"result.json").read_text()) if (run/"result.json").is_file() else {{"dispatch_returncode":p.returncode}}
 except subprocess.TimeoutExpired: result={{"dispatch_timeout":True}}
 except Exception as e: result={{"dispatch_exception":str(e)}}
 success=result.get("build_rc")==0 and result.get("import_rc")==0
 status=("build_smoke_passed" if task["strategy"]=="build_smoke_only" else "reference_precheck_passed") if success else "precheck_failed"
 rec={{**task,"attempt":attempt,"status":status,"failure_gate":result.get("gate_failure",result.get("stop_reason","")),"result_path":str((run/"result.json").relative_to(BASE)),"finished_at":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())}}
 state["results"].append(rec); save(); log(f"PR-{{pr}} {{status}}")
state["finished_at"]=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime());save();log("FINISHED")
'''
generated = HERE / "generated_remote_reference_precheck.py"
generated.write_text(script)
mkdir = subprocess.run([CTL, "run", f"mkdir -p {REMOTE}"], text=True, capture_output=True, timeout=120)
if mkdir.returncode:
    print(mkdir.stdout); print(mkdir.stderr, file=sys.stderr); raise SystemExit(mkdir.returncode)
payload = base64.b64encode(gzip.compress(script.encode())).decode()
command = (
    f"echo {payload} | base64 -d | gzip -d > {REMOTE}/supervisor.py; "
    f"source /usr/local/Ascend/ascend-toolkit/set_env.sh; unset ASCEND_VISIBLE_DEVICES ASCEND_RT_VISIBLE_DEVICES; "
    f"nohup /opt/mamba/bin/python3 -u {REMOTE}/supervisor.py > {REMOTE}/supervisor-stdout.log 2>&1 < /dev/null & "
    f"echo $! > {REMOTE}/supervisor.pid; sleep 2; cat {REMOTE}/supervisor.pid; ps -p $(cat {REMOTE}/supervisor.pid) -o pid=,stat=,etime=,cmd="
)
cp = subprocess.run([CTL, "run", command], text=True, capture_output=True, timeout=120)
print(cp.stdout); print(cp.stderr, file=sys.stderr)
raise SystemExit(cp.returncode)
