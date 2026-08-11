"""Minimal 8-rank HCCL gate for the Ascend 910B 8-card target.

The launcher must set WORLD_SIZE=8.  The environment setup belongs to the
calling command so that the visible-device variables are cleared before
torch_npu import.
"""

import os

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401 -- registers the NPU backend


rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])

assert world_size == 8, world_size
assert torch.npu.is_available(), "NPU unavailable"
assert torch.npu.device_count() == 8, torch.npu.device_count()

torch.npu.set_device(local_rank)
dist.init_process_group(backend="hccl")
value = torch.tensor([float(rank + 1)], device=f"npu:{local_rank}")
dist.all_reduce(value, op=dist.ReduceOp.SUM)
torch.npu.synchronize()
assert value.item() == 36.0, (rank, value.item())

if rank == 0:
    print("HCCL_ALL_REDUCE_8_RANK_PASS world_size=8 sum=36", flush=True)
