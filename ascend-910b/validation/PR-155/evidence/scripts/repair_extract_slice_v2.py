#!/usr/bin/env python3
"""Patch the PR-155 repair clone without shell-dependent multiline matching."""
from pathlib import Path
import subprocess
import os

repo = Path(os.environ["PR155_REPO"])
out = Path(os.environ["PR155_OUT"])
(out / "runs").mkdir(parents=True, exist_ok=True)
(out / "logs").mkdir(parents=True, exist_ok=True)
p = repo / "python/sgl_kernel_npu/sgl_kernel_npu/activation/swiglu_quant.py"
s = p.read_text()
old = '''        # swiglu
        x_offsets = row_idx * TOTAL_COLS + tl.arange(0, TOTAL_COLS)
        cur_x = tl.load(x_ptr + x_offsets)
        x1 = tl.extract_slice(cur_x, offsets=(0,), sizes=(HALF_COLS,), strides=(1,))
        x2 = tl.extract_slice(
            cur_x, offsets=(HALF_COLS,), sizes=(HALF_COLS,), strides=(1,)
        )
        out = x1 * tl.sigmoid(x1) * x2
'''
new = '''        # Triton 3.2 compatibility: load the two halves directly.
        half_offsets = tl.arange(0, HALF_COLS)
        x1 = tl.load(x_ptr + row_idx * TOTAL_COLS + half_offsets)
        x2 = tl.load(x_ptr + row_idx * TOTAL_COLS + HALF_COLS + half_offsets)
        out = x1 * tl.sigmoid(x1) * x2
'''
assert old in s, "SwiGLU input slice block did not match"
s = s.replace(old, new, 1)
old = '''                tmp_out = tl.extract_slice(
                    out, offsets=(col_blk_idx,), sizes=(COL_BLOCK_SIZE,), strides=(1,)
                )
'''
new = '''                block_offsets = col_blk_idx + tl.arange(0, COL_BLOCK_SIZE)
                block_mask = block_offsets < HALF_COLS
                block_x1 = tl.load(x_ptr + row_idx * TOTAL_COLS + block_offsets, mask=block_mask, other=0.0)
                block_x2 = tl.load(x_ptr + row_idx * TOTAL_COLS + HALF_COLS + block_offsets, mask=block_mask, other=0.0)
                tmp_out = block_x1 * tl.sigmoid(block_x1) * block_x2
'''
assert old in s, "SwiGLU quant slice block did not match"
s = s.replace(old, new, 1)
old = "                tl.store(out_ptr + o_offsets, tmp_out.to(out_ptr.dtype.element_ty))\n"
new = "                tl.store(out_ptr + o_offsets, tmp_out.to(out_ptr.dtype.element_ty), mask=block_mask)\n"
assert old in s, "SwiGLU store block did not match"
s = s.replace(old, new, 1)
assert "extract_slice" not in s
p.write_text(s)
with (out / "runs/repair-v2.patch").open("w") as f:
    subprocess.run(["git", "diff", "--", "CMakeLists.txt", "csrc/CMakeLists.txt", "csrc/utils/common.h", str(p.relative_to(repo))], cwd=repo, stdout=f, check=True, text=True)
env = os.environ.copy(); env["PYTHONPATH"] = "python/sgl_kernel_npu"
with (out / "logs/correctness.log").open("w") as f:
    r = subprocess.run(["python3", "-m", "pytest", "-q", "tests/python/sgl_kernel_npu/test_swiglu_quant.py"], cwd=repo, env=env, stdout=f, stderr=subprocess.STDOUT, text=True)
(out / "runs/correctness-exit-code.txt").write_text(f"{r.returncode}\n")
raise SystemExit(r.returncode)
