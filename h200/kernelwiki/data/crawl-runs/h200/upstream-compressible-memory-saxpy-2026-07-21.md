# H200 upstream source run: compressible-memory SAXPY

- Source: `mirrors/cuda-samples@b7c5481c556c3fe98db060207ecaa41a4b9a9abc`
- Files: `saxpy.cu`, `compMalloc.cpp`, `compMalloc.h`
- SHA256: `3db9c6024336fb21a4b55ceba53487209dbd11aca360d6702fea72ee23be4cd6`,
  `d0e357aea0317771b64f5f5ee3e313c88ebcd83c406dbae724803fbfae81bd7b`,
  `dcab4824a59672c46824e55e06dff0b1ba6c35f5e190b85c59729dfa5b765494`
- Build: `nvcc -O3 -arch=sm_90 -I Common saxpy.cu compMalloc.cpp -lcuda`

The unmodified multi-file upstream sample compiled and ran on NVIDIA H200
(SM90). Generic memory compression support was available. Its own single-run
comparison reported:

```text
Compressible memory:     0.078 ms, 6.430 TB/s
Non-compressible memory: 0.127 ms, 3.962 TB/s
```

The observed source-internal ratio is about 1.63x in this one scenario. The
upstream source explicitly states CUDA Samples are not performance
measurements, and the run contains neither repeated trials nor an external
reference. It is therefore `runnable` evidence, not a benchmarked compression
speedup claim.
