import os, triton, re
base = os.path.dirname(triton.__file__)
print("=== warp_specialization / enable_ws in compiler options & jit ===")
for root,_,files in os.walk(base):
    for f in files:
        if not f.endswith(".py"): continue
        p=os.path.join(root,f)
        try: t=open(p,errors="replace").read()
        except: continue
        for i,ln in enumerate(t.splitlines(),1):
            if re.search(r"enable_warp_specialization|warp_specialization\s*[:=]|num_warps.*ws|enable_ws|launch_pdl", ln):
                print(f"  {p.replace(base,'')}/{i}: {ln.strip()[:140]}")
