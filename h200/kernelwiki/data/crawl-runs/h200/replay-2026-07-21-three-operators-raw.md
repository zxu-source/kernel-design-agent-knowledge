# H200 replay: three representative derived operators

Date: 2026-07-21  
Target: `h200_ncu`  
GPU: NVIDIA H200  
Toolchain: Triton 3.6.0  
Command: `python3 /tmp/kernelwiki_h200_replay.py`  
Warmup/measurement: 3 warmup calls, 10 CUDA-event measurements; minimum latency reported.

This is the raw stdout returned by the remote H200 run. It is evidence for this
replay only; it does not turn the implementations into upstream source snapshots.

```text
[h200_ncu] status=ok rc=0 seconds=4.86
GPU NVIDIA H200 Triton 3.6.0
RESULT_JSON {"fused_qkt": [{"err_finite": 4.470348358154297e-08, "shape": [512, 512, 64], "torch_ms": 0.045791998505592346, "torch_over_triton": 1.9549180067151921, "triton_ms": 0.023423999547958374}, {"err_finite": 7.450580596923828e-08, "shape": [1024, 1024, 128], "torch_ms": 0.04575999826192856, "torch_over_triton": 1.9972066825648378, "triton_ms": 0.022911999374628067}, {"err_finite": 6.705522537231445e-08, "shape": [2048, 2048, 128], "torch_ms": 0.08419200032949448, "torch_over_triton": 3.2321866329186193, "triton_ms": 0.026048000901937485}], "softmax": [{"err": 7.62939453125e-06, "shape": [1024, 1024], "torch_ms": 0.01071999967098236, "torch_over_triton": 0.6203703753607392, "triton_ms": 0.01727999933063984}, {"err": 1.9073486328125e-06, "shape": [4096, 4096], "torch_ms": 0.0597120001912117, "torch_over_triton": 1.9829968062489667, "triton_ms": 0.030112000000000003}, {"err": 3.814697265625e-06, "shape": [4096, 32000], "torch_ms": 0.3131519854068756, "torch_over_triton": 1.277545612464646, "triton_ms": 0.24512000000000003}], "tiled_transpose": [{"err": 0.0, "shape": [1024, 1024], "torch_ms": 0.01651199907064438, "torch_over_triton": 0.8500822914370205, "triton_ms": 0.019424000000000002}, {"err": 0.0, "shape": [4096, 4096], "torch_ms": 0.1281599998474121, "torch_over_triton": 4.163201697291843, "triton_ms": 0.0307839997112751}, {"err": 0.0, "shape": [8192, 4096], "torch_ms": 0.22819200158119202, "torch_over_triton": 4.828029972887131, "triton_ms": 0.04726399853825569}]}
```

Interpretation: fused QKᵀ is faster than the torch reference in all three
shapes (about 1.95x–3.23x by `torch_ms / triton_ms`); tiled transpose is slower
at 1024² but faster at the two larger shapes (about 4.16x–4.83x); softmax is
slower at 1024² and faster at the larger two shapes (about 1.28x–1.98x).
