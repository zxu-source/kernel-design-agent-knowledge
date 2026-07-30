# Iteration log and stopping rule

## Stopping rule

Do not terminate after a single regression. Before each new candidate, run a
new KernelWiki query and inspect an applicable page/provenance bundle. Stop
only when a candidate is correct and its median latency is no worse than
`candidate_00_baseline` on every representative benchmark case, or after three
consecutive correct new candidates fail to improve the current best result.

## Candidate sequence

| Version | Pre-implementation KernelWiki query | Outcome |
| --- | --- | --- |
| `candidate_00_baseline` | N/A baseline | Correct. One custom CUDA launch per nonempty expert. |
| `candidate_01_wmma_persistent_prototype` | Original required grouped-GEMM/Hopper queries; `kernel-grouped-gemm`, `pr-cutlass-3091` | Failed correctness; per-thread ticket violated CTA cooperative ownership; retained. |
| `candidate_01_kernelwiki` | Same initial evidence set | Correct CTA-level persistent queue. Improved 3 cases but regressed mixed 4096x4096; continue required. |
| `candidate_02_static_queue` | `Hopper grouped GEMM static tile scheduler persistent atomic overhead variable M`; `kernel-grouped-gemm`, `pr-cutlass-3091`, provenance `artifacts/prs/cutlass/PR-3091/PROVENANCE.yaml` | Correct. Replaced dynamic atomic queue with host-precomputed one-CTA-per-tile static queue. Meets stopping rule. |

## Median latency by version (us)

| Case | Baseline | Persistent | Static queue | Best |
| --- | ---: | ---: | ---: | --- |
| uniform, E=8, K=N=1024 | 1945.90 | 1342.41 | 724.31 | static, 2.69x vs baseline |
| skewed, E=16, K=2048,N=1024 | 9955.76 | 4820.32 | 1666.35 | static, 5.97x |
| small, E=32, K=4096,N=1024 | 36012.30 | 6448.93 | 1609.48 | static, 22.38x |
| mixed, E=8, K=N=4096 | 18755.60 | 40294.60 | 7849.02 | static, 2.39x |

`candidate_02_static_queue` is correct and beats baseline in all four cases;
therefore the first stopping condition is met. Raw rows are preserved in
`runs/candidate_02_static_queue_benchmark.csv`.
