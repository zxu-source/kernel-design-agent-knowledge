# H200 benchmark upgrade: all experimental kernel pages

Date: 2026-07-21  
Target: `h200_ncu` (NVIDIA H200, Triton 3.6.0)

All 62 original `op_*.py` harnesses completed with return code 0 under an
isolated batch runner (180-second per-harness timeout). The raw per-harness
stdout/stderr and structured status summary are preserved in
`replay-2026-07-21-all-experimental-raw.tar.gz`.

- Archive SHA256: `b3442f98f2978e42d49cf26c61853c9f7be210d6450d2d5596551d84e7fbe2a9`
- Archive contents: `summary.json`, `driver.log`, and 62 `logs/op_*.log` files.
- Batch outcome: 62 passed, 0 failed, 0 timed out.
- The causal-mask log in that archive is retained as historical output but is
  superseded by `replay-2026-07-21-causal-mask-corrected-raw.md` because its old
  `-inf - -inf` error calculation was invalid.
- Non-causal FA2 was not an `op_*.py` harness and is recorded separately in
  `replay-2026-07-21-fa2-hopper-raw.md`.

These are local derived-implementation benchmarks, not claims about upstream
source performance. A ratio greater than 1 means only that the recorded torch
reference was slower for that tested shape and toolchain.
