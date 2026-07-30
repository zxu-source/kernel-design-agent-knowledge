import os, triton, subprocess, re
base = os.path.dirname(triton.__file__)
for rel in ("backends/nvidia/driver.py", "backends/nvidia/compiler.py", "language/extra/cuda/gdc.py"):
    f = os.path.join(base, rel)
    txt = open(f, errors="replace").read()
    print("="*30, rel, "="*30)
    for i, ln in enumerate(txt.splitlines(), 1):
        if re.search(r"pdl|griddepcontrol|launch_dependent|programmatic|gdc|PDL|GDC", ln, re.I):
            print(f"{i:4d}: {ln.rstrip()[:150]}")
