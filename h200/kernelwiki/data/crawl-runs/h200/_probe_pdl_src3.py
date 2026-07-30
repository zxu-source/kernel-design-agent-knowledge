import os, triton, subprocess, re
base = os.path.dirname(triton.__file__)
# 1) every assignment / kw involving launch_pdl across the package
print("=== launch_pdl usages ===")
for root, _, files in os.walk(base):
    for fn in files:
        if not fn.endswith((".py",)):
            continue
        p = os.path.join(root, fn)
        try: txt = open(p, errors="replace").read()
        except: continue
        for i, ln in enumerate(txt.splitlines(), 1):
            if "launch_pdl" in ln or "enable_pdl" in ln or "TRITON_PDL" in ln:
                print(f"  {p.replace(base,'')}/{i}: {ln.strip()[:150]}")
# 2) compiler.py context around launch_pdl default
print("\n=== compiler.py context (launch_pdl) ===")
txt = open(os.path.join(base,"backends/nvidia/compiler.py"), errors="replace").read()
lines = txt.splitlines()
for i, ln in enumerate(lines, 1):
    if "launch_pdl" in ln:
        a, b = max(1,i-6), min(len(lines), i+6)
        for j in range(a, b+1):
            print(f"  {j}: {lines[j-1].rstrip()[:150]}")
        print("  ---")
