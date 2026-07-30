#!/usr/bin/env python3
"""H200 validation for Triton PR #6394 (PDL) — CUDA-graph variant.

Same kernel as triton6394_pdl_chain.py, but the launch chain is captured into a
CUDA graph and replayed, removing Python/CPU dispatch overhead from the timing.
This isolates the GPU-side ramp-down/ramp-up overlap that PDL is supposed to
provide. If PDL helps, it shows up here; if it stays ~1.0x under graph replay,
that is a strong negative result for this kernel class on H200 / Triton 3.6.0.
"""
import json, statistics
import torch, triton, triton.language as tl

@triton.jit
def chain_step(x_ptr, n, alpha, beta, BLOCK: tl.constexpr, PDL: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    if PDL:
        tl.extra.cuda.gdc_wait()
    x = tl.load(x_ptr + offs, mask=m)
    tl.store(x_ptr + offs, x * alpha + beta, mask=m)
    if PDL:
        tl.extra.cuda.gdc_launch_dependents()

def build_graph(x, alpha, beta, n_launches, BLOCK, pdl):
    grid = ((x.numel() + BLOCK - 1) // BLOCK,)
    # pre-compile
    chain_step[grid](x, x.numel(), alpha, beta, BLOCK=BLOCK, PDL=pdl, launch_pdl=pdl)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n_launches):
            chain_step[grid](x, x.numel(), alpha, beta, BLOCK=BLOCK, PDL=pdl, launch_pdl=pdl)
    return g

def time_graph(g, trials=10):
    for _ in range(5):          # warmup replays
        g.replay(); torch.cuda.synchronize()
    ts = []
    for _ in range(trials):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); g.replay(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return min(ts), statistics.median(ts)

def main():
    print(f"triton={triton.__version__} dev={torch.cuda.get_device_name(0)} "
          f"SMs={torch.cuda.get_device_properties(0).multi_processor_count}")
    n_elem = 1 << 20
    alpha, beta = 0.999, 1.0
    configs = [(BLOCK, N) for BLOCK in (1024, 2048, 4096) for N in (256, 1024, 4096)]
    rows = []
    for BLOCK, N in configs:
        xo = torch.zeros(n_elem, device="cuda", dtype=torch.float32)
        xn = torch.zeros(n_elem, device="cuda", dtype=torch.float32)
        try:
            go = build_graph(xo, alpha, beta, N, BLOCK, pdl=False)
            gn = build_graph(xn, alpha, beta, N, BLOCK, pdl=True)
        except Exception as ex:
            print(f"BLOCK={BLOCK} N={N} graph-build failed: {repr(ex)[:120]}")
            continue
        off_min, off_med = time_graph(go)
        on_min, on_med = time_graph(gn)
        # correctness: compare final buffers
        xo.zero_(); xn.zero_()
        go.replay(); gn.replay(); torch.cuda.synchronize()
        delta = (xo - xn).abs().max().item()
        aN = alpha ** N
        expect = beta * (aN - 1.0) / (alpha - 1.0)
        speedup = off_min / on_min if on_min > 0 else float("nan")
        per_launch_off = off_min * 1e3 / N   # us
        per_launch_on = on_min * 1e3 / N
        rows.append(dict(BLOCK=BLOCK, N=N, off_ms=round(off_min,4), on_ms=round(on_min,4),
                         speedup=round(speedup,3), per_launch_off_us=round(per_launch_off,3),
                         per_launch_on_us=round(per_launch_on,3), delta=delta, expect=round(expect,3)))
        print(f"BLOCK={BLOCK:5d} N={N:5d}  off={off_min:8.4f}ms ({per_launch_off:6.3f}us/launch)  "
              f"on={on_min:8.4f}ms ({per_launch_on:6.3f}us/launch)  speedup={speedup:5.3f}x  delta={delta:.1e}")
    print("RESULT_JSON", json.dumps(rows))
    print("max_delta=", max((r["delta"] for r in rows), default=float("nan")))

if __name__ == "__main__":
    main()
