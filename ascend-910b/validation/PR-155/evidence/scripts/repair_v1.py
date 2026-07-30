#!/usr/bin/env python3
"""PR-155 isolated 910B repair: legacy host integer-header compatibility."""
from __future__ import annotations
import json, os, subprocess, time
from pathlib import Path

root = Path(os.environ["REPAIR_ROOT"])
repo = root / "repo" / "sgl-kernel-npu"
logs = root / "logs"; runs = root / "runs"; outputs = root / "outputs"; profile = root / "profile"
for p in (logs, runs, outputs, profile): p.mkdir(parents=True, exist_ok=True)
sha = "d54224c4d16edddc9c0ea0e1ca08fd51e58fa2f9"
env = os.environ.copy(); env["PYTHONPATH"] = "python/sgl_kernel_npu"
result = {"pr": 155, "sha": sha, "repair": "CMake GCC/ACL compatibility plus cstdint/climits legacy host header includes"}

def run(cmd, cwd, name, env_override=None):
    started=time.monotonic()
    merged=env.copy(); merged.update(env_override or {})
    with (logs/name).open("w") as f:
        f.write("$ "+" ".join(cmd)+"\n"); f.flush()
        p=subprocess.run(cmd,cwd=cwd,env=merged,stdout=f,stderr=subprocess.STDOUT,text=True)
        f.write(f"\n[exit_code={p.returncode}]\n")
    return p.returncode, round(time.monotonic()-started,3)

if not repo.exists():
    repo.parent.mkdir(parents=True, exist_ok=True)
    result["clone_rc"],_ = run(["git","clone","--no-checkout","https://github.com/sgl-project/sgl-kernel-npu.git",str(repo)],root,"git-clone.log")
else: result["clone_rc"] = 0
if result["clone_rc"] == 0:
    result["fetch_rc"],_ = run(["git","fetch","--depth=1","origin",sha],repo,"git-fetch.log")
if result.get("fetch_rc") == 0:
    result["checkout_rc"],_ = run(["git","checkout","--detach",sha],repo,"git-checkout.log")
    result["head"] = subprocess.check_output(["git","rev-parse","HEAD"],cwd=repo,text=True).strip()
if result.get("head") == sha:
    cmake=(repo/"CMakeLists.txt"); csrc=(repo/"csrc/CMakeLists.txt"); common=(repo/"csrc/utils/common.h")
    cmake.write_text(cmake.read_text().replace("-hno-unused-parameter -lno-unused-function ",""))
    src=csrc.read_text(); csrc.write_text(src.replace("${TORCH_NPU_DIR}/include","${TORCH_NPU_DIR}/include\n        ${TORCH_NPU_DIR}/include/third_party/acl/inc",1))
    header=common.read_text(); common.write_text(header.replace("#define UTILS_COMMON_H\n", "#define UTILS_COMMON_H\n\n#include <cstdint>\n#include <climits>\n",1))
    with (runs/"repair-v1.patch").open("w") as f: subprocess.run(["git","diff","--","CMakeLists.txt","csrc/CMakeLists.txt","csrc/utils/common.h"],cwd=repo,stdout=f,check=True,text=True)
    build=repo/"build"
    result["configure_rc"],_=run(["cmake","-S",".","-B",str(build),"-DBUILD_DEEPEP_MODULE=OFF","-DSOC_VERSION=Ascend910B2C","-DASCEND_INCLUDE_DIR=/usr/local/Ascend/cann-9.0.0/include"],repo,"configure.log")
    if result["configure_rc"] == 0:
        result["build_rc"],_=run(["cmake","--build",str(build),"--target","sgl_kernel_npu","-j2"],repo,"build.log")
    if result.get("build_rc") == 0:
        test="tests/python/sgl_kernel_npu/test_swiglu_quant.py"
        result["correctness_rc"],result["correctness_seconds"]=run(["python3","-m","pytest","-q",test],repo,"correctness.log")
result["finished_at"]=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())
(root/"result.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,sort_keys=True))
