import triton, triton.language as tl, torch

@triton.jit
def k(x_ptr, n, BLOCK: tl.constexpr, USE_GDC: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid*BLOCK + tl.arange(0,BLOCK); m = offs < n
    if USE_GDC: tl.extra.cuda.gdc_wait()
    tl.store(x_ptr+offs, tl.load(x_ptr+offs, mask=m)+1.0, mask=m)
    if USE_GDC: tl.extra.cuda.gdc_launch_dependents()

x = torch.ones(1<<16, device="cuda"); grid=(x.numel()//1024,)
# compile BOTH variants and inspect metadata + PTX
for lp in (False, True):
    for gdc in (False, True):
        kern = k.warmup(x, x.numel(), BLOCK=1024, USE_GDC=gdc, launch_pdl=lp, grid=grid)
        asm = kern.metadata
        ptx = (asm.asm.get("ptx","") if hasattr(asm,"asm") else "")
        has_gdc = "griddepcontrol" in ptx
        print(f"launch_pdl={lp} USE_GDC={gdc} -> metadata.launch_pdl={getattr(asm,'launch_pdl',None)}  griddepcontrol_in_ptx={has_gdc}")
