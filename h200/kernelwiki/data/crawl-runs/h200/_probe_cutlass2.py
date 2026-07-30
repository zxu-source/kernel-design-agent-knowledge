import os, re
base = "/usr/local/lib/python3.12/dist-packages/vllm/third_party/deep_gemm/include/cutlass"
# 1) grid_dependency_control.h — the PDL primitive API
g = os.path.join(base, "arch/grid_dependency_control.h")
print("=== arch/grid_dependency_control.h ===")
print(open(g, errors="replace").read()[:2500])
# 2) Does sm90_gemm_array_tma_warpspecialized_cooperative.hpp reference launch/wait dependent grids (i.e., is PDL already wired)?
f = os.path.join(base, "gemm/kernel/sm90_gemm_array_tma_warpspecialized_cooperative.hpp")
txt = open(f, errors="replace").read()
print("\n=== PDL refs in sm90_gemm_array_tma_...cooperative.hpp ===")
for i, ln in enumerate(txt.splitlines(), 1):
    if re.search(r"launch_dependent|wait_on_dependent|grid_dependency|PDL|GridDependency", ln, re.I):
        print(f"{i}: {ln.strip()[:130]}")
print("file len:", len(txt.splitlines()))
# 3) Is there a ready CUTLASS device-layer grouped GEMM (CollectiveBuilder) header we can instantiate?
import glob
print("\n=== collective builder headers present? ===")
for h in ("gemm/collective/_sm90_scheduler.hpp","gemm/collective/builders/sm90_gmma_builder.inl"):
    print(h, os.path.exists(os.path.join(base,h)))
print("total cutlass headers:", len(glob.glob(base+"/**/*.hpp", recursive=True))+len(glob.glob(base+"/**/*.h", recursive=True)))
