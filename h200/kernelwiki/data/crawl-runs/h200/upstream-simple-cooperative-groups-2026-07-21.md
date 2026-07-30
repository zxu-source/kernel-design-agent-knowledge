# H200 upstream source probe: simpleCooperativeGroups

- Source: `mirrors/cuda-samples@b7c5481c556c3fe98db060207ecaa41a4b9a9abc`
- File SHA256: `cd2851e9e2534cf4704816c97864e3951662abc0a42bcbd0c9f68749e70dc3c1`
- Command: `nvcc -O3 -arch=sm_90 simpleCooperativeGroups.cu && ./simpleCooperativeGroups`
- Result: compile and run passed on NVIDIA H200 (SM90). This sample has no latency comparison, so it is a smoke test rather than a performance claim.

```text
Launching a single block with 64 threads...
Sum of all ranks 0..63 in threadBlockGroup is 2016 (expected 2016)
Now creating 4 groups, each of size 16 threads:
Sum of all ranks 0..15 in this tiledPartition16 group is 120 (expected 120)
...Done.
```
