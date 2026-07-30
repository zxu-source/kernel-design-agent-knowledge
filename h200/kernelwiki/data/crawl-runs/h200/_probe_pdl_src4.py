import os, triton, triton.language as tl, torch, inspect
# show knobs.nvidia fields
try:
    kn = triton.knobs.nvidia
    print("KNOBS.nvidia attrs:", [a for a in dir(kn) if "pdl" in a.lower() or "coop" in a.lower()])
except Exception as e:
    print("KNOBS err", repr(e))

@triton.jit
def k(x_ptr, n, BLOCK: tl.constexpr, USE_GDC: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid*BLOCK + tl.arange(0,BLOCK); m = offs < n
    if USE_GDC: tl.extra.cuda.gdc_wait()
    tl.store(x_ptr+offs, tl.load(x_ptr+offs, mask=m)+1.0, mask=m)
    if USE_GDC: tl.extra.cuda.gdc_launch_dependents()

x = torch.ones(1<<16, device="cuda"); grid=(x.numel()//1024,)
def try_call(label, **kw):
    try:
        k[grid](x, x.numel(), BLOCK=1024, USE_GDC=True, **kw)
        torch.cuda.synchronize(); print(label, "OK", kw)
    except Exception as e:
        print(label, "ERR", repr(e)[:120])

try_call("kwarg launch_pdl=True", launch_pdl=True)
# compile explicitly and inspect metadata
try:
    kern = k.warmup(x, x.numel(), BLOCK=1024, USE_GDC=True, grid=grid)
    md = kern.metadata if hasattr(kern,"metadata") else None
    print("metadata type", type(md))
    if md is not None:
        print("metadata.launch_pdl =", getattr(md, "launch_pdl", "<absent>"))
except Exception as e:
    print("warmup err", repr(e)[:160])
# env var route
for ev in ("TRITON_PDL","TRITON_LAUNCH_PDL","TRITON_NVIDIA_LAUNCH_PDL"):
    os.environ[ev]="1"
try:
    k[grid](x, x.numel(), BLOCK=1024, USE_GDC=True); torch.cuda.synchronize()
    print("env-var launch OK; re-check metadata.launch_pdl after env set:")
    kern2 = k.warmup(x, x.numel(), BLOCK=1024, USE_GDC=True, grid=grid)
    print("  metadata.launch_pdl =", getattr(kern2.metadata, "launch_pdl", "<absent>"))
except Exception as e:
    print("env-var launch ERR", repr(e)[:160])
