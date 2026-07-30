#!/usr/bin/env python3
"""Finish PR-557 benchmark and torch_npu profile after its passing upstream test."""
from __future__ import annotations
import json, os, subprocess, sys, time
from pathlib import Path

root = Path(os.environ["PR557_ROOT"])
repo = root / "compat" / "sgl-kernel-npu"
out = root / "repair-v1"; logs = out / "logs"; outputs = out / "outputs"; profile_dir = out / "profile" / "torch_npu"
for p in (logs, outputs, profile_dir.parent): p.mkdir(parents=True, exist_ok=True)
env = os.environ.copy(); env["PYTHONPATH"] = "python/sgl_kernel_npu"
test = "tests/python/sgl_kernel_npu/test_fused_rope_qk_mqa.py"
samples=[]
if os.environ.get("PR557_PROFILE_ONLY") != "1":
    for index in range(3):
        started=time.monotonic()
        with (logs / f"benchmark-{index + 1}.log").open("w") as f:
            p=subprocess.run([sys.executable,"-m","pytest","-q",test],cwd=repo,env=env,stdout=f,stderr=subprocess.STDOUT,text=True)
            f.write(f"\n[exit_code={p.returncode}]\n")
        samples.append({"seconds":round(time.monotonic()-started,3),"exit_code":p.returncode})
        if p.returncode: raise SystemExit(p.returncode)
    (outputs / "benchmark-walltime.json").write_text(json.dumps({"kind":"three upstream-test wall-time samples; not an operator performance comparison","samples":samples},indent=2)+"\n")

import torch
import torch_npu.profiler as profiler
import pytest
os.chdir(repo)
with (logs / "profile-v2.log").open("w") as f:
    with profiler.profile(activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU], schedule=profiler.schedule(wait=0,warmup=0,active=1,repeat=1), on_trace_ready=profiler.tensorboard_trace_handler(str(profile_dir)), record_shapes=True, profile_memory=True) as prof:
        rc=pytest.main([test,"-q"])
        torch.npu.synchronize()
        prof.step()
    f.write(f"PYTEST_RC={rc}\n")
if rc: raise SystemExit(rc)
print(json.dumps({"benchmark":samples,"profile_rc":rc,"profile_dir":str(profile_dir)}))
