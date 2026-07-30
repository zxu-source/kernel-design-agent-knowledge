import triton, triton.language as tl, torch

@triton.jit
def k_keep(x_ptr, n, BLOCK: tl.constexpr, USE_GDC: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid*BLOCK + tl.arange(0,BLOCK); m = offs < n
    g = tl.zeros((BLOCK,), dtype=tl.int32)
    if USE_GDC:
        g = tl.extra.cuda.gdc_wait()           # keep result alive
    x = tl.load(x_ptr+offs, mask=m)
    tl.store(x_ptr+offs, x + (g & 0).to(tl.float32) + 1.0, mask=m)
    if USE_GDC:
        d = tl.extra.cuda.gdc_launch_dependents()
        # force keep by writing dummy into a dedicated scratch slot (offs 0)
        tl.store(x_ptr + 0, tl.load(x_ptr+0) + (d & 0).to(tl.float32))

x = torch.ones(1<<16, device="cuda"); grid=(x.numel()//1024,)
kern = k_keep.warmup(x, x.numel(), BLOCK=1024, USE_GDC=True, launch_pdl=True, grid=grid)
asm = kern.metadata
print("asm keys:", list(asm.asm.keys()) if hasattr(asm,"asm") else "n/a")
ptx = asm.asm.get("ptx","") if hasattr(asm,"asm") else ""
print("ptx len:", len(ptx))
import re
for m in re.finditer(r"griddepcontrol\.\w+", ptx):
    print("  found PTX:", m.group(0))
# also dump a small window around any 'griddep' if present
i = ptx.find("griddep")
print("griddep index:", i)
if i>=0:
    print(ptx[max(0,i-80):i+120])
