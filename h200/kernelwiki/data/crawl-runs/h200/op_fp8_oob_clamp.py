#!/usr/bin/env python3
"""H200 FP8 OOB-Clamp Robustness validation.
FP8 e4m3 max = 448. Inputs with large outliers / Inf overflow without clamping
(-> NaN/undefined on cast). This validates that the clamp in fp8-quant is a
ROBUSTNESS guard: clamped kernel produces no NaN (all in [-448,448]); naive
(no-clamp) produces NaN. Also measures clamp overhead (should be ~free).
PURPOSE = robustness. H200, Triton 3.6.
"""
import json, statistics
import torch, triton, triton.language as tl
def cdiv(a,b): return (a+b-1)//b
@triton.jit
def fp8q_clamp(x_ptr, o_ptr, inv_scale, N, BLOCK: tl.constexpr):
    pid=tl.program_id(0); offs=pid*BLOCK+tl.arange(0,BLOCK); mask=offs<N
    x=tl.load(x_ptr+offs, mask=mask, other=0.0).to(tl.float32)
    q=x*inv_scale
    q=tl.minimum(tl.maximum(q, -448.0), 448.0)   # ROBUST: clamp to fp8 range
    tl.store(o_ptr+offs, q.to(tl.float8e4nv), mask=mask)
@triton.jit
def fp8q_naive(x_ptr, o_ptr, inv_scale, N, BLOCK: tl.constexpr):
    pid=tl.program_id(0); offs=pid*BLOCK+tl.arange(0,BLOCK); mask=offs<N
    x=tl.load(x_ptr+offs, mask=mask, other=0.0).to(tl.float32)
    q=x*inv_scale                               # NO clamp: OOB -> undefined/NaN on cast
    tl.store(o_ptr+offs, q.to(tl.float8e4nv), mask=mask)
def run(kernel, x, inv, BLOCK=4096):
    o=torch.empty((x.numel(),),device=x.device,dtype=torch.float8_e4m3fn)
    kernel[(cdiv(x.numel(),BLOCK),)](x, o, inv, x.numel(), BLOCK=BLOCK); return o
def time_fn(fn,trials=50):
    for _ in range(10): fn(); torch.cuda.synchronize()
    ts=[]
    for _ in range(trials):
        s,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    return min(ts), statistics.median(ts)
def main():
    print(f"triton={triton.__version__} dev={torch.cuda.get_device_name(0)} SMs={torch.cuda.get_device_properties(0).multi_processor_count}")
    N=1<<24
    torch.manual_seed(0)
    x=torch.randn(N,device="cuda",dtype=torch.float32)*0.5
    # inject 0.1% extreme outliers + some Inf/NaN
    out_idx=torch.randperm(N)[:N//1000]
    x[out_idx]=1e4
    inf_idx=torch.randperm(N)[:64]; x[inf_idx]=float('inf')
    nan_idx=torch.randperm(N)[:64]; x[nan_idx]=float('nan')
    # use a SMALL scale so most values are in range but outliers/inf overflow
    inv=1.0   # nominal scale (x*1.0); outliers 1e4 -> far above 448 -> clamp needed
    o_clamp=run(fp8q_clamp, x, inv); o_naive=run(fp8q_naive, x, inv)
    oc=o_clamp.float(); on=o_naive.float()
    clamp_nan=torch.isnan(oc).sum().item(); clamp_inf=torch.isinf(oc).sum().item()
    naive_nan=torch.isnan(on).sum().item(); clamp_max=oc.abs().max().item()
    print(f"clamp: NaN={clamp_nan} Inf={clamp_inf} max_abs={clamp_max} (expect NaN=0, max<=448)")
    print(f"naive (no-clamp): NaN={naive_nan} (expect >0 from inf/outlier overflow)")
    # also with a proper per-tensor scale (amax/448) so clamped = exact, naive may NaN at inf
    amax=x[~torch.isnan(x)&~torch.isinf(x)].abs().max()
    inv2=(448.0/amax).item()
    o2=run(fp8q_clamp, x, inv2)
    print(f"with proper scale (amax/448): clamp NaN={torch.isnan(o2.float()).sum().item()} (inf clamps to 448, not NaN)")
    tc,_=time_fn(lambda: run(fp8q_clamp,x,inv)); tn,_=time_fn(lambda: run(fp8q_naive,x,inv))
    print(f"overhead: clamp={tc:.4f}ms naive={tn:.4f}ms clamp/naive={tc/tn:.3f}x (clamp ~free)")
    print("RESULT_JSON", json.dumps(dict(N=int(N),clamp_nan=int(clamp_nan),clamp_max=float(clamp_max),
        naive_nan=int(naive_nan),with_scale_nan=int(torch.isnan(o2.float()).sum().item()),
        clamp_ms=round(tc,4),naive_ms=round(tn,4),clamp_over_naive=round(tc/tn,3))))
if __name__=="__main__": main()
