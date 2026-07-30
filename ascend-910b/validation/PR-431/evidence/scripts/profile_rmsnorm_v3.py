#!/usr/bin/env python3
"""Scheduled NPU trace for the repository-local PR-431 fused RMSNorm."""

import os

import torch
import torch_npu.profiler as profiler

from sgl_kernel_npu.norm.rmsnorm_without_weight import fused_rmsnorm_without_weight


trace_dir = os.environ["PR431_PROFILE_DIR"]
x = torch.randn(1, 130, 2048, dtype=torch.float32, device="npu")
eps = 1e-6

for _ in range(5):
    fused_rmsnorm_without_weight(x, eps)
torch.npu.synchronize()

with profiler.profile(
    activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU],
    schedule=profiler.schedule(wait=0, warmup=0, active=3, repeat=1),
    on_trace_ready=profiler.tensorboard_trace_handler(trace_dir),
    record_shapes=True,
    profile_memory=True,
) as prof:
    for _ in range(3):
        fused_rmsnorm_without_weight(x, eps)
        torch.npu.synchronize()
        prof.step()

print(f"PROFILE_TRACE_DIR={trace_dir}")
