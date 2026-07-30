import os, triton, subprocess, re
base = os.path.dirname(triton.__file__)
print("triton base", base)
# search python sources
pyhits = subprocess.run(["grep","-rIl","-E","pdl|griddepcontrol|programmatic|launch_dependent|gdc", base],
                        capture_output=True, text=True).stdout.strip().splitlines()
print("FILES mentioning pdl/gdc:")
for f in pyhits[:40]:
    print("  ", f.replace(base,""))
# show lines with 'pdl' (case-insensitive) limited
print("\nLINES with pdl:")
for f in pyhits:
    try:
        txt = open(f, errors="replace").read()
    except Exception:
        continue
    for ln in txt.splitlines():
        if re.search(r"\bpdl\b", ln, re.I):
            print(f"  {os.path.basename(f)}: {ln.strip()[:140]}")
