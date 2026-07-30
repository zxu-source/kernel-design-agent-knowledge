#!/usr/bin/env python3
"""Config-driven, resumable 910B validation orchestrator.

The local process plans and invokes 910bctl; the generated remote runner owns
checkout, build, correctness, benchmark and profile gates.  It never updates a
source-page validation status: promotion remains a review decision.
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
WORKSPACE = HERE.parents[3]
CORPUS = WORKSPACE / "npu-kernelwiki"
CTL = Path("/home/kirin_14379/projects/ai4qz/scripts/910bctl")


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


REMOTE_RUNNER = r'''#!/usr/bin/env python3
import json, os, subprocess, sys, time
from pathlib import Path

cfg=json.loads(Path(sys.argv[1]).read_text())
base=Path(cfg["run_dir"]); repo=base/"repo"/"sgl-kernel-npu"
for d in (base/"logs", base/"runs", base/"outputs", base/"profile"): d.mkdir(parents=True,exist_ok=True)
env=os.environ.copy(); env["PYTHONPATH"]="python/sgl_kernel_npu" + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
res={"pr":cfg["pr"],"merge_sha":cfg["merge_sha"],"environment":"Ascend910B2C/CANN-9.0.0 current server","test_mapping":cfg.get("test_mapping","reviewed"),"validation_scope":cfg.get("validation_scope","correctness")}
def run(cmd,name,cwd=None):
    start=time.monotonic(); cwd=cwd or repo
    with (base/"logs"/name).open("w") as f:
        f.write("$ "+" ".join(cmd)+"\n"); p=subprocess.run(cmd,cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT,text=True); f.write(f"\n[exit_code={p.returncode}]\n")
    return p.returncode, round(time.monotonic()-start,3)
if not repo.exists():
    repo.parent.mkdir(parents=True,exist_ok=True); res["clone_rc"],_=run(["git","clone","--no-checkout","https://github.com/sgl-project/sgl-kernel-npu.git",str(repo)],"git-clone.log",base)
else: res["clone_rc"]=0
if res["clone_rc"]==0: res["fetch_rc"],_=run(["git","fetch","--depth=1","origin",cfg["merge_sha"]],"git-fetch.log")
if res.get("fetch_rc")==0: res["checkout_rc"],_=run(["git","checkout","--detach",cfg["merge_sha"]],"git-checkout.log")
if res.get("checkout_rc")==0:
    res["head"]=subprocess.check_output(["git","rev-parse","HEAD"],cwd=repo,text=True).strip()
    if res["head"] != cfg["merge_sha"]: res["stop_reason"]="checkout SHA mismatch"
if "stop_reason" not in res:
    if "gcc_acl" in cfg.get("compatibility",[]):
        cm=repo/"CMakeLists.txt"; cs=repo/"csrc/CMakeLists.txt"
        cm.write_text(cm.read_text().replace("-hno-unused-parameter -lno-unused-function ",""))
        if "third_party/acl/inc" not in cs.read_text(): cs.write_text(cs.read_text().replace("${TORCH_NPU_DIR}/include","${TORCH_NPU_DIR}/include\n        ${TORCH_NPU_DIR}/include/third_party/acl/inc",1))
        with (base/"runs"/"compat.patch").open("w") as f: subprocess.run(["git","diff","--","CMakeLists.txt","csrc/CMakeLists.txt"],cwd=repo,stdout=f,text=True,check=True)
    build=repo/"build"; res["configure_rc"],_=run(["cmake","-S",".","-B",str(build),"-DBUILD_DEEPEP_MODULE=OFF","-DSOC_VERSION=Ascend910B2C","-DASCEND_INCLUDE_DIR=/usr/local/Ascend/cann-9.0.0/include"],"configure.log")
    if res["configure_rc"]==0: res["build_rc"],_=run(["cmake","--build",str(build),"--target","sgl_kernel_npu","-j2"],"build.log")
    if res.get("build_rc")==0:
        res["correctness_rc"],_=run([sys.executable,"-m","pytest","-q",cfg["test"]],"correctness.log")
    if res.get("correctness_rc")==0:
        samples=[]
        for i in range(cfg.get("benchmark_repeats",3)):
            rc,sec=run([sys.executable,"-m","pytest","-q",cfg["test"]],f"benchmark-{i+1}.log"); samples.append({"exit_code":rc,"seconds":sec})
            if rc: break
        res["benchmark_samples"]=samples; res["benchmark_rc"]=0 if all(x["exit_code"]==0 for x in samples) else 1
        (base/"outputs"/"benchmark-walltime.json").write_text(json.dumps({"kind":"upstream-test wall-time only; not operator performance","samples":samples},indent=2)+"\n")
        if res["benchmark_rc"]==0 and cfg.get("profile_mode") == "pytest":
            profile_script = r"""import json, os, sys
from pathlib import Path
import pytest, torch, torch_npu
cfg=json.loads(Path(sys.argv[1]).read_text()); repo=Path(sys.argv[2]); out=Path(sys.argv[3])
os.chdir(repo); out.mkdir(parents=True, exist_ok=True)
activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU]
with torch_npu.profiler.profile(activities=activities, schedule=torch_npu.profiler.schedule(wait=0,warmup=0,active=1,repeat=1), on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(out)), record_shapes=True, profile_memory=True) as prof:
    rc=pytest.main([cfg["test"], "-q"])
    prof.step()
raise SystemExit(rc)
"""
            script_path=base/"runs"/"profile_pytest.py"; script_path.write_text(profile_script)
            res["profile_rc"],_=run([sys.executable,str(script_path),str(Path(sys.argv[1]).resolve()),str(repo),str(base/"profile")],"profile.log",repo)
res["finished_at"]=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())
(base/"result.json").write_text(json.dumps(res,indent=2)+"\n"); print(json.dumps(res,sort_keys=True))
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=HERE / "manifest.json")
    ap.add_argument("--pr", type=int, action="append")
    ap.add_argument("--execute", action="store_true", help="actually invoke the 910B runner")
    args = ap.parse_args()
    cfg = json.loads(args.manifest.read_text())
    items = [x for x in cfg["candidates"] if not args.pr or x["pr"] in args.pr]
    if not items: raise SystemExit("no candidates selected")
    for item in items: require_bundle(item["pr"], item["merge_sha"])
    plan = HERE / "planned-runs.json"
    plan.write_text(json.dumps(items, indent=2) + "\n")
    print(f"validated manifest entries: {len(items)}; plan: {plan}")
    if not args.execute:
        print("dry-run only; append --execute --pr <N> to run one PR on 910B")
        return 0
    local_results = HERE / "queue-results"
    local_results.mkdir(exist_ok=True)
    for item in items:
        if not item.get("test"):
            record = {
                "pr": item["pr"], "merge_sha": item["merge_sha"],
                "state": "needs_test_mapping", "reason": "no upstream correctness test path in archived source page",
                "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            (local_results / f"PR-{item['pr']}.json").write_text(json.dumps(record, indent=2) + "\n")
            print(f"PR-{item['pr']}: recorded needs_test_mapping; no remote build/test started")
            continue
        remote_dir = f'{cfg["remote_root"]}/PR-{item["pr"]}'
        remote_cfg = {
            **item,
            "run_dir": remote_dir,
            "benchmark_repeats": cfg.get("defaults", {}).get("benchmark_repeats", 3),
            "profile_mode": item.get("profile_mode", cfg.get("defaults", {}).get("profile_mode", "none")),
        }
        runner_b64 = base64.b64encode(REMOTE_RUNNER.encode()).decode()
        cfg_b64 = base64.b64encode(json.dumps(remote_cfg).encode()).decode()
        command = (
            "source /usr/local/Ascend/ascend-toolkit/set_env.sh; unset ASCEND_VISIBLE_DEVICES ASCEND_RT_VISIBLE_DEVICES; "
            f"mkdir -p {remote_dir}; echo {runner_b64} | base64 -d > {remote_dir}/runner.py; "
            f"echo {cfg_b64} | base64 -d > {remote_dir}/config.json; python3 {remote_dir}/runner.py {remote_dir}/config.json"
        )
        subprocess.run([str(CTL), "run", command], check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
