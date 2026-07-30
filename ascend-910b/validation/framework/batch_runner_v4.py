#!/usr/bin/env python3
"""910B batch runner — v4 with base64 config (bulletproof quoting)."""
import base64, json, os, subprocess, sys, time

QUEUE_PATH = "/home/kirin_14379/projects/ai4qz/kda-h200-workspace/npu-kernelwiki/validation/ascend-910b/framework/full-execution-queue.json"
CTL = "/home/kirin_14379/projects/ai4qz/scripts/910bctl"
LOCAL = "/home/kirin_14379/projects/ai4qz/kda-h200-workspace/npu-kernelwiki/validation/ascend-910b"
BASE = "/inspire/sj-ssd/project/qianghuaxuexi/s26043/npu-kernelwiki-validation"
RUNNER = f"{BASE}/runner_v2.py"

q = json.loads(open(QUEUE_PATH).read())

# HAS_CMAKE correctness_candidates
PRS = [35, 38, 41, 43, 46]

for pr_num in PRS:
    item = next((x for x in q["candidates"] if x["pr"] == pr_num), None)
    if not item:
        continue
    sha = item["merge_sha"]
    test = item.get("test")
    scope = item.get("validation_scope", "correctness_candidate")
    mapping = item.get("test_mapping", "source_page_test_path")

    print(f"\n=== PR-{pr_num} sha={sha[:12]} test={test} ===", flush=True)

    cfg = {
        "pr": pr_num, "merge_sha": sha, "test": test,
        "test_mapping": mapping, "validation_scope": scope,
        "compatibility": ["gcc_acl"],
        "run_dir": f"{BASE}/PR-{pr_num}",
        "benchmark_repeats": 3, "profile_mode": "pytest"
    }
    cfg_b64 = base64.b64encode(json.dumps(cfg).encode()).decode()

    # safe: echo <base64> | base64 -d > config.json
    full_cmd = (
        f"source /usr/local/Ascend/ascend-toolkit/set_env.sh && "
        f"unset ASCEND_VISIBLE_DEVICES ASCEND_RT_VISIBLE_DEVICES && "
        f"rm -rf {BASE}/PR-{pr_num} && "
        f"mkdir -p {BASE}/PR-{pr_num}/logs {BASE}/PR-{pr_num}/runs "
        f"{BASE}/PR-{pr_num}/outputs {BASE}/PR-{pr_num}/profile && "
        f"echo {cfg_b64} | base64 -d > {BASE}/PR-{pr_num}/config.json && "
        f"python3 {RUNNER} {BASE}/PR-{pr_num}/config.json"
    )

    t0 = time.monotonic()
    cp = subprocess.run([CTL, "run", full_cmd], capture_output=True, text=True, timeout=900)
    dt = time.monotonic() - t0
    out = cp.stdout

    result = {}
    if "RESULT_JSON_START" in out:
        s = out.index("RESULT_JSON_START") + 18
        e = out.index("RESULT_JSON_END") if "RESULT_JSON_END" in out else len(out)
        try:
            result = json.loads(out[s:e].strip())
        except Exception:
            result = {"parse_error": out[s:e].strip()[:300]}
    else:
        result = {"no_json": True, "rc": cp.returncode, "dt": dt,
                  "stdout_tail": out[-500:] if out else "",
                  "stderr_tail": cp.stderr[-500:] if cp.stderr else ""}

    os.makedirs(f"{LOCAL}/PR-{pr_num}/evidence", exist_ok=True)
    json.dump(result, open(f"{LOCAL}/PR-{pr_num}/result.json", "w"), indent=2, sort_keys=True)

    stop = result.get("stop_reason", "PASS")
    gate = result.get("gate_failure", "OK")
    print(f"  PR-{pr_num}: gate={gate} stop={stop} "
          f"build={result.get('build_rc','?')} import={result.get('import_rc','?')} "
          f"correct={result.get('correctness_rc','?')} profile={result.get('profile_rc','?')} "
          f"({dt:.0f}s)", flush=True)

print("\n=== Done ===", flush=True)
