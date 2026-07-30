# PR-572 local validation record

## Result

PR-572 (`causal_conv1d` PTO-ISA rewrite) is validated on the recorded Ascend
910B2C / CANN 9.0.0 environment. This means the exact merge commit built with
the recorded compatibility patch, the repository-local extension registered,
and the upstream `test_conv1d_prefill.py` suite passed with exit code 0.

It is not benchmarked, profiled, or KDA-ready. Nothing in this record supports
a local performance conclusion.

## Provenance

- Merge SHA: `d5630dff41c8108216f835597e63f6d3a7445908`
- Hardware: Ascend 910B2C (container logical device `npu:0`)
- CANN: 9.0.0
- Python: 3.13.13; torch: `2.11.0+cpu`; torch_npu: `2.11.0.rc4`
- Archive SHA-256: `db4ed41d7886397d4abb787f0f6d37e589106bb524500d208294a273578987d7`

`evidence/` contains the original downloaded archive, its extracted contents,
the exact upstream test source, build/configure logs, library hash, merge SHA,
repository status, and compatibility patch. The patch altered two CMake files;
inspect it before treating the build as portable.

## Reproduction status

The passing run used the repository package through `PYTHONPATH`, rather than
the installed site-packages copy. It also unsets both Ascend visible-device
variables because this container exposes the physical NPU 3 as logical NPU 0.
