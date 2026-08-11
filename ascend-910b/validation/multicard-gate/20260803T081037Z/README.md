# Ascend 910B 8-card P0 gate — 2026-08-03

## Scope and target

- Target: `ascend_910b_8card`
- Wrapper: `/home/kirin_14379/projects/ai4qz/scripts/910b8ctl`
- Configuration: `/home/kirin_14379/projects/ai4qz/configs/notebooks-910b-8card.yaml`
- This run never used the single-card `ascend_910b` target or `910bctl` wrapper.

## Gate verdict

**PASSED with a process-scoped HCCL topology setting.**

The target-level connectivity, physical device, CANN environment, PyTorch
device-count, 8-rank launch, and HCCL all-reduce gates passed.  The successful
smoke used `HCCL_INTRA_ROCE_ENABLE=1` in its command environment; it did not
modify the Notebook, the target configuration, or any shared environment.
Without that variable the same all-reduce failed because device 0 and device
12 are on different planes.  See `hccl-smoke.log`.

## Passed evidence

- Deep connectivity: `ok: True`, `xsrf_found: True`, `contents_api_ok: True`,
  and `probe_exit_code: 0` (`deep.log`).
- `npu-smi info`: eight healthy `910B2C` devices, physical NPU ids
  `0, 3, 5, 7, 12, 13, 14, 15` (`npu-smi.log`).
- CANN setup loaded from `/usr/local/Ascend/ascend-toolkit/set_env.sh`.
- `ASCEND_VISIBLE_DEVICES` and `ASCEND_RT_VISIBLE_DEVICES` were unset before
  importing PyTorch.
- `torch.npu.is_available() == True` and `torch.npu.device_count() == 8`
  (`environment.log`).
- Eight independent torchrun ranks each saw `WORLD_SIZE=8` and eight logical
  NPU devices (`torchrun-probe.log`).
- The final HCCL all-reduce returned rc=0 and validated the sum 1+...+8=36
  (`hccl-smoke.log`).

## Smoke design

The versioned local script `hccl_all_reduce_smoke.py` asserted `WORLD_SIZE ==
8` and logical device count 8, set each rank's NPU device, initialized
`torch.distributed` with backend `hccl`, all-reduced rank values 1..8,
synchronized, and asserted the sum was 36.  It used:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=8 --tee 3 <temporary-script>
```

No PR source, build, import, test, benchmark, or profile was run.  No PR
compatibility patch was used.  There is no performance conclusion.

## Next P0 action

Before each multi-card PR run, source CANN, unset the two visible-device
variables, and set `HCCL_INTRA_ROCE_ENABLE=1` for the child process.  Keep the
exact setting and its scope in the PR's environment snapshot.  Do not
substitute the single-card target.
