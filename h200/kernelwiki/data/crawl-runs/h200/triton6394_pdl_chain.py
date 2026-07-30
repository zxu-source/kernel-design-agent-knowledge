#!/usr/bin/env python3
"""H200 validation for Triton PR #6394 (Programmatic Dependent Launch).

Source-informed microbenchmark (not a reproduction of upstream tutorial 11):
measures whether PDL (launch_pdl=True + griddepcontrol intrinsics) reduces the
wall-clock time of a chain of short, data-dependent Triton kernels on H200
(SM90). PDL overlaps a kernel's ramp-down (store drain / teardown) with the next
kernel's ramp-up (launch / scheduling / first loads).

Hardware: NVIDIA H200, 132 SMs, cc 9.0. Triton 3.6.0.
Compile: JIT by Triton (sm_90).  Timing: CUDA events, min of TRIALS trials.
"""
import json, statistics
import torch, triton, triton.language as tl

@triton.jit
def chain_step(x_ptr, n, alpha, beta, BLOCK: tl.constexpr, PDL: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    if PDL:
        tl.extra.cuda.gdc_wait()                 # dependent waits for prior grid's signal + data visibility
    x = tl.load(x_ptr + offs, mask=m)
    tl.store(x_ptr + offs, x * alpha + beta, mask=m)
    if PDL:
        tl.extra.cuda.gdc_launch_dependents()    # signal next grid may start its ramp-up

def run_chain(x, alpha, beta, n_launches, BLOCK, pdl):
    grid = ((x.numel() + BLOCK - 1) // BLOCK,)
    # PDL is a compile (launch) option in Triton 3.6.0; pass as launch kwarg.
    for _ in range(n_launches):
        chain_step[grid](x, x.numel(), alpha, beta, BLOCK=BLOCK, PDL=pdl, launch_pdl=pdl)

def time_chain(x, alpha, beta, n_launches, BLOCK, pdl, trials=7):
    # warmup
    for _ in range(3):
        run_chain(x, alpha, beta, n_launches, BLOCK, pdl)
    torch.cuda.synchronize()
    times = []
    for _ in range(trials):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); run_chain(x, alpha, beta, n_launches, BLOCK, pdl); e.record()
        torch.cuda.synchronize(); times.append(s.elapsed_time(e))
    return min(times), statistics.median(times)

def correctness(n_elem, BLOCK, n_launches, alpha, beta):
    """Verify PDL chain produces the same result as a non-PDL reference chain."""
    ref = torch.zeros(n_elem, device="cuda", dtype=torch.float32)
    tst = torch.zeros(n_elem, device="cuda", dtype=torch.float32)
    run_chain(ref, alpha, beta, n_launches, BLOCK, pdl=False)
    run_chain(tst, alpha, beta, n_launches, BLOCK, pdl=True)
    # closed form: x_k = a^k * x0 + beta*(a^{k-1}+...+1), x0=0 => beta*(a^k-1)/(a-1)
    aN = alpha ** n_launches
    expect_val = beta * (aN - 1.0) / (alpha - 1.0)
    err_ref = (ref - expect_val).abs().max().item()
    err_tst = (tst - expect_val).item() if False else (tst - ref).abs().max().item()
    return err_ref, err_tst, expect_val

def main():
    print(f"triton={triton.__version__} dev={torch.cuda.get_device_name(0)} "
          f"SMs={torch.cuda.get_device_properties(0).multi_processor_count}")
    n_elem = 1 << 20  # 1M floats
    alpha, beta = 0.999, 1.0
    configs = [(BLOCK, N) for BLOCK in (1024, 4096) for N in (128, 512, 1024)]
    rows = []
    for BLOCK, N in configs:
        x = torch.zeros(n_elem, device="cuda", dtype=torch.float32)
        off_min, off_med = time_chain(x, alpha, beta, N, BLOCK, pdl=False)
        x = torch.zeros(n_elem, device="cuda", dtype=torch.float32)
        on_min, on_med = time_chain(x, alpha, beta, N, BLOCK, pdl=True)
        er_ref, er_cmp, ev = correctness(n_elem, BLOCK, N, alpha, beta)
        speedup = off_min / on_min if on_min > 0 else float("nan")
        rows.append(dict(BLOCK=BLOCK, N=N, off_ms=round(off_min,4), on_ms=round(on_min,4),
                         speedup=round(speedup,3), off_med=round(off_med,4), on_med=round(on_med,4),
                         err_vs_ref=er_cmp, expect=round(ev,4)))
        print(f"BLOCK={BLOCK:5d} N={N:5d}  off={off_min:8.4f}ms  on={on_min:8.4f}ms  "
              f"speedup={speedup:5.3f}x  |delta_vs_nonpdl|={er_cmp:.2e}")
    # also single-launch latency (launch-overhead bound) for context
    for BLOCK in (1024, 4096):
        x = torch.zeros(n_elem, device="cuda", dtype=torch.float32)
        one_off, _ = time_chain(x, alpha, beta, 1, BLOCK, pdl=False)
        x = torch.zeros(n_elem, device="cuda", dtype=torch.float32)
        one_on, _ = time_chain(x, alpha, beta, 1, BLOCK, pdl=True)
        print(f"single-launch BLOCK={BLOCK}: off={one_off:.5f}ms on={one_on:.5f}ms (launch-overhead reference)")
    print("RESULT_JSON", json.dumps(rows))
    print("CORRECTNESS expect_val_sample=", round(rows[0]['expect'],4),
          "max_delta_vs_nonpdl=", max(r['err_vs_ref'] for r in rows))

if __name__ == "__main__":
    main()
