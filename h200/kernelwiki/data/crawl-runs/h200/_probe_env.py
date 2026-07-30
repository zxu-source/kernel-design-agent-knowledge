import sys, importlib.util, os
print("PY", sys.version.split()[0])
# Triton
try:
    import triton
    print("TRITON", triton.__version__)
    import triton.language as tl
    has_gdc = hasattr(tl.extra, "cuda") and hasattr(tl.extra.cuda, "gdc_wait")
    print("TRITON_gdc_wait", has_gdc)
    print("TRITON_gdc_launch_dependents", hasattr(tl.extra, "cuda") and hasattr(tl.extra.cuda, "gdc_launch_dependents"))
    # JIT attributes for enable_pdl
    print("TRITON_jit_has_enable_pdl", hasattr(triton.runtime.jit.JITFunction, "enable_pdl") if hasattr(triton, "runtime") else "n/a")
except Exception as e:
    print("TRITON_ERR", repr(e))
# triton_kernels
print("TRITON_KERNELS_INSTALLED", importlib.util.find_spec("triton_kernels") is not None)
# CUTLASS
try:
    import cutlass
    print("CUTLASS", cutlass.__version__)
    print("CUTLASS_path", os.path.dirname(cutlass.__file__))
    # Look for SM90 array TMA GEMM header and PDL helpers
    cdir = os.path.dirname(cutlass.__file__)
    import subprocess
    # find included cutlass C++ headers if bundled
    hits = subprocess.run(["find", cdir, "-name", "sm90_gemm_array_tma*.hpp"], capture_output=True, text=True).stdout.strip()
    print("CUTLASS_hdr_sm90_array_tma", hits or "(none bundled)")
    pdl = subprocess.run(["find", cdir, "-name", "*.hpp", "-o", "-name", "*.h"], capture_output=True, text=True).stdout
    print("CUTLASS_hdr_count", len(pdl.splitlines()))
except Exception as e:
    print("CUTLASS_ERR", repr(e))
# Device
try:
    import torch
    p = torch.cuda.get_device_properties(0)
    print("DEV", p.name, "SMs=", p.multi_processor_count, "cc=", f"{p.major}.{p.minor}", "max_smem=", getattr(p, "shared_memory_per_block_optin", -1))
except Exception as e:
    print("DEV_ERR", repr(e))
