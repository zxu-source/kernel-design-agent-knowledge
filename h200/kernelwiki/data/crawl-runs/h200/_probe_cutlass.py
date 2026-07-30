import os, subprocess, glob
print("=== search whole site-packages + /usr/local for cutlass C++ headers ===")
for base in ["/usr/local/lib/python3.12/dist-packages", "/usr/local/include", "/usr/include"]:
    for pat in ("**/sm90_gemm_array_tma*.hpp", "**/cutlass/arch/grid*.h", "**/grid_dependency.h"):
        try:
            hits = glob.glob(os.path.join(base, pat), recursive=True)
        except Exception:
            hits = []
        if hits:
            print(f"{base}/{pat}: {len(hits)} hits; e.g.", hits[0])
# Does the cutlass DSL bundle headers in its package data?
import cutlass
cdir = os.path.dirname(cutlass.__file__)
print("cutlass pkg:", cdir)
# walk up to find the nvidia_cutlass_dsl root and any 'include' dir
for d, _, files in os.walk(os.path.dirname(cdir)):
    if any(f.endswith(".hpp") for f in files):
        # only print dirs whose path mentions cutlass and contain gemm/sm90
        if "cutlass" in d.lower():
            rel = d.replace(os.path.dirname(cdir), "")
            sample = [f for f in files if f.endswith((".hpp",".h"))][:3]
            if any("sm90" in f or "grid" in f or "gemm" in f for f in files):
                print("HAS_HEADERS:", rel, "n=", len(files), "sample=", sample)
print("=== DSL API surface for GEMM + PDL ===")
import cutlass
print("cutlass attrs:", [a for a in dir(cutlass) if a.lower() in ("op","invoke","cute","epilogue") or "gemm" in a.lower()][:20])
try:
    import cutlass.cute as cute
    print("cute attrs:", [a for a in dir(cute) if not a.startswith("_")][:25])
except Exception as e:
    print("cute import err", repr(e)[:120])
# Search DSL python for PDL / launch_dependent
import re
for d, _, files in os.walk(cdir):
    for f in files:
        if not f.endswith(".py"): continue
        p = os.path.join(d, f)
        try: t = open(p, errors="replace").read()
        except: continue
        if re.search(r"launch_dependent|wait_on_dependent|grid_dependency|PDL|pdl", t):
            print("DSL-PDL:", p.replace(cdir,""), "->", [ln.strip()[:90] for ln in t.splitlines() if re.search(r"launch_dependent|wait_on_dependent|grid_dependency|\bPDL\b|\bpdl\b", ln)][:3])
