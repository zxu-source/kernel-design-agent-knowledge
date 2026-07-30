import triton, triton.language as tl, torch, inspect

@triton.jit
def _k(x_ptr, n, BLOCK: tl.constexpr, USE_GDC: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    if USE_GDC:
        tl.extra.cuda.gdc_wait()
    x = tl.load(x_ptr + offs, mask=m)
    tl.store(x_ptr + offs, x + 1.0, mask=m)
    if USE_GDC:
        tl.extra.cuda.gdc_launch_dependents()

x = torch.ones(1 << 16, device="cuda")
grid = (x.numel() // 1024,)

# Discover how PDL is enabled at launch in this Triton version.
print("triton", triton.__version__)
# 1) inspect JITFunction signature for run/launch kwargs
for meth in ("run", "__call__"):
    try:
        sig = str(inspect.signature(getattr(_k, meth)))
        print(f"JIT.{meth} sig:", sig[:300])
    except Exception as e:
        print(f"JIT.{meth} err:", repr(e))

# 2) Check for an enable_pdl attribute / launch hint in the compiled kernel cache
ok_kw = None
for kw in ("enable_pdl",):
    try:
        _k[grid](x, x.numel(), BLOCK=1024, USE_GDC=True, **{kw: True})
        torch.cuda.synchronize()
        ok_kw = kw + "=True (launch kwarg accepted)"
    except Exception as e:
        ok_kw = f"{kw}=True REJECTED: {repr(e)[:160]}"
print("LAUNCH_kwarg:", ok_kw)

# 3) Check decorator option
try:
    @triton.jit(enable_pdl=True)
    def _k2(x_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        tl.store(x_ptr + offs, tl.load(x_ptr + offs) + 1.0, mask=offs < n)
    _k2[grid](x, x.numel(), BLOCK=1024)
    torch.cuda.synchronize()
    print("DECORATOR_enable_pdl: accepted")
except Exception as e:
    print("DECORATOR_enable_pdl: rejected ->", repr(e)[:160])
