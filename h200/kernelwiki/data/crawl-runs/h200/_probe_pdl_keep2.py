import triton, triton.language as tl, torch, re

@triton.jit
def k_keep(x_ptr, n, BLOCK: tl.constexpr, USE_GDC: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid*BLOCK + tl.arange(0,BLOCK); m = offs < n
    if USE_GDC:
        w = tl.inline_asm_elementwise("griddepcontrol.wait;", "=r", [], dtype=tl.int32, is_pure=False, pack=1)
        tl.store(x_ptr+0, tl.load(x_ptr+0) + (w & 0).to(tl.float32))
    x = tl.load(x_ptr+offs, mask=m)
    tl.store(x_ptr+offs, x + 1.0, mask=m)
    if USE_GDC:
        d = tl.inline_asm_elementwise("griddepcontrol.launch_dependents;", "=r", [], dtype=tl.int32, is_pure=False, pack=1)
        tl.store(x_ptr+0, tl.load(x_ptr+0) + (d & 0).to(tl.float32))

x = torch.ones(1<<16, device="cuda"); grid=(x.numel()//1024,)
kern = k_keep.warmup(x, x.numel(), BLOCK=1024, USE_GDC=True, launch_pdl=True, grid=grid)
asm = kern.metadata
print("asm keys:", list(asm.asm.keys()))
ptx = asm.asm.get("ptx","")
print("ptx len:", len(ptx))
found = [m.group(0) for m in re.finditer(r"griddepcontrol\.\w+", ptx)]
print("griddepcontrol found:", found)
