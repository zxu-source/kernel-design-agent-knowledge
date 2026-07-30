# PR-592 manual server runbook

Run this on the 910B server. It intentionally creates a separate worktree and
does not alter PR-572's recorded checkout. Stop at the first failed gate and
download the resulting `runs/` and `logs/` directory instead of patching code.

## 1. Set paths and capture the preflight

```bash
set -euo pipefail

BASE=/inspire/sj-ssd/project/qianghuaxuexi/s26043/npu-kernelwiki-validation
WORK="$BASE/pr592-20260727"
REPO="$WORK/repo/sgl-kernel-npu"
RUN="$WORK/runs"
LOG="$WORK/logs"
MERGE=8f13e502e5bbbd027f5e677e3a99ba1ab1095fce

test ! -e "$WORK" || { echo "Refusing to reuse $WORK"; exit 1; }
mkdir -p "$RUN" "$LOG" "$WORK/repo"

source /usr/local/Ascend/ascend-toolkit/set_env.sh
unset ASCEND_VISIBLE_DEVICES ASCEND_RT_VISIBLE_DEVICES

{
  date -Is
  npu-smi info
  python -c 'import torch, torch_npu; print("torch=", torch.__version__); print("torch_npu=", torch_npu.__version__); print("available=", torch.npu.is_available()); print("count=", torch.npu.device_count()); print("name=", torch.npu.get_device_name(0) if torch.npu.device_count() else None)'
} 2>&1 | tee "$RUN/pr592-environment.txt"
```

Proceed only when `available=True`, `count=1`, and the logical device name is
`Ascend910B2C`.

## 2. Exact checkout and source compatibility preflight

```bash
git clone --recursive https://github.com/sgl-project/sgl-kernel-npu.git "$REPO" 2>&1 | tee "$LOG/git-clone.log"
git -C "$REPO" checkout --detach "$MERGE" 2>&1 | tee "$LOG/git-checkout.log"
git -C "$REPO" submodule update --init --recursive 2>&1 | tee "$LOG/git-submodule.log"

{
  git -C "$REPO" rev-parse HEAD
  git -C "$REPO" status --short
  rg -n -i 'SOC_VERSION|Ascend910|arch35|causal_conv1d' "$REPO/CMakeLists.txt" "$REPO/csrc" "$REPO/build.sh"
  find "$REPO/tests" -type f \( -iname '*conv1d*' -o -iname '*gdn*' \) -print
} 2>&1 | tee "$RUN/pr592-source-preflight.txt"

test "$(git -C "$REPO" rev-parse HEAD)" = "$MERGE"
```

If `arch35` or the build configuration excludes `Ascend910B2C`, stop here:
PR-592 remains `captured/unverified`. Do not copy PR-572's compatibility patch
into this separate checkout.

## 3. Configure and compile without source edits

```bash
TORCH_DIR=$(python -c 'import torch; print(torch.__path__[0])')
TORCH_NPU_DIR=$(python -c 'import torch_npu; print(torch_npu.__path__[0])')
PYBIND11_DIR=$(python -c 'import pybind11; print(pybind11.__path__[0])')

cmake -S "$REPO" -B "$REPO/build" \
  -DSOC_VERSION=Ascend910B2C \
  -DASCEND_CANN_PACKAGE_PATH="$ASCEND_HOME_PATH" \
  -DASCEND_HOME_PATH="$ASCEND_HOME_PATH" \
  -DASCEND_INCLUDE_DIR="$ASCEND_HOME_PATH/include" \
  -DTORCH_DIR="$TORCH_DIR" \
  -DTORCH_NPU_DIR="$TORCH_NPU_DIR" \
  -DPYBIND11_DIR="$PYBIND11_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  2>&1 | tee "$LOG/cmake-config.log"

set -o pipefail
cmake --build "$REPO/build" -j"$(nproc)" 2>&1 | tee "$LOG/build.log"
echo "build_exit_code=${PIPESTATUS[0]}" | tee "$RUN/pr592-build-exit-code.txt"
```

If either command fails, preserve these logs and stop. Do not remove compiler
flags or add headers yet; that would be a separately recorded compatibility
experiment, not a clean PR-592 validation.

## 4. Prove repository-local import and identify the test

```bash
export PYTHONPATH="$REPO/python/sgl_kernel_npu${PYTHONPATH:+:$PYTHONPATH}"

python - <<'PY' 2>&1 | tee "$RUN/pr592-repo-package-import.txt"
import pathlib
import torch
import torch_npu
import sgl_kernel_npu
print("package=", sgl_kernel_npu.__file__)
print("package_is_repo=", str(pathlib.Path(sgl_kernel_npu.__file__).resolve()).startswith("/inspire/sj-ssd/project/qianghuaxuexi/s26043/npu-kernelwiki-validation/pr592-20260727/repo/"))
print("npu_available=", torch.npu.is_available())
print("npu_count=", torch.npu.device_count())
print("causal_conv1d_registered=", hasattr(torch.ops.npu, "causal_conv1d"))
PY

find "$REPO/tests" -type f \( -iname '*conv1d*' -o -iname '*gdn*' \) -print | tee "$RUN/pr592-candidate-tests.txt"
```

At this point, do **not** choose a test by filename alone. Send back
`pr592-source-preflight.txt`, `cmake-config.log`, `build.log`,
`pr592-repo-package-import.txt`, and `pr592-candidate-tests.txt`. We will map
the test signature to PR-592's operator schema before selecting the correctness
command. No benchmark or profiler should run in this round.
