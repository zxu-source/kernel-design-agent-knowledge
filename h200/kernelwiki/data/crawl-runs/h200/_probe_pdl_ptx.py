import triton, triton.language as tl, torch, re

@triton.jit
def k(x_ptr, n, BLOCK: tl.constexpr, USE_GDC: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid*BLOCK + tl.arange(0,BLOCK); m = offs < n
    if USE_GDC: tl.extra.cuda.gdc_wait()
    tl.store(x_ptr+offs, tl.load(x_ptr+offs, mask=m)+1.0, mask=m)
    if USE_GDC: tl.extra.cuda.gdc_launch_dependents()

x = torch.ones(1<<16, device="cuda"); grid=(x.numel()//1024,)
kern = k.warmup(x, x.numel(), BLOCK=1024, USE_GDC=True, launch_pdl=True, grid=grid)
print("type(kern)=", type(kern))
print("kern attrs:", [a for a in dir(kern) if not a.startswith("_")][:40])
print("metadata.launch_pdl =", getattr(kern.metadata,"launch_pdl",None))
# locate PTX
ptx = ""
if hasattr(kern, "asm") and isinstance(kern.asm, dict):
    ptx = kern.asm.get("ptx","")
    print("kern.asm keys:", list(kern.asm.keys()))
if not ptx and hasattr(kern, "metadata"):
    md = kern.metadata
    print("metadata attrs:", [a for a in dir(md) if not a.startswith("_")][:40])
    for a in ("asm","ptx","binary","cubin"):
        if hasattr(md, a):
            v = getattr(md,a); print(f"  md.{a} type:", type(v), (list(v.keys()) if isinstance(v,dict) else getattr(v,'__len__',lambda:0)()))
print("ptx len:", len(ptx))
found = sorted(set(m.group(0) for m in re.finditer(r"griddepcontrol\.\w+", ptx)))
print("griddepcontrol found:", found)
