# Kernel Design Agent Knowledge Base

This repository is a versioned evidence base for kernel-design work. It keeps upstream PR provenance, extracted implementation evidence, derived H200 experiments, and Ascend 910B validation records separate.

## Scope of v0.1

- `h200/kernelwiki/`: vLLM and SGLang PR source pages and artifact bundles, plus the KernelWiki retrieval material used by H200 work.
- `h200/derived-validations/`: independently implemented H200 experiments informed by upstream evidence. These are not claims that the corresponding upstream PRs were reproduced verbatim.
- `ascend-910b/corpus/`: isolated `sgl-project/sgl-kernel-npu` source and artifact corpus.
- `ascend-910b/validation/`: local 910B runner, result records, evidence, classification output, and reports.
- `ascend-910b/runs/`: crawl status and review decisions needed to interpret the corpus.

## Evidence boundaries

- A `source page` records PR metadata, merge SHA, description, and source URL.
- An `artifact bundle` records `diff.patch`, selected `key-files/`, and `PROVENANCE.yaml`.
- `FULL_PASS` is a runner outcome, not automatically a strict validation claim.
- `validated` requires exact SHA, build, repository-local import, a correct correctness/reference invocation, and complete local evidence.
- Benchmark or profile data are never used as a correctness substitute, and no performance claim is made without the corresponding evidence.

## Current snapshot

| Lane | Current contents |
|---|---|
| H200 vLLM | 1,364 source pages; 1,298 complete artifact bundles |
| H200 SGLang | 1,289 source pages; 1,130 complete artifact bundles |
| Ascend sgl-kernel-npu | 384 merged-PR source pages; 281 complete implementation bundles; 103 source-only/context PRs |
| Ascend validation | 93 correctness candidates, 160 heuristic probes, and 28 reference-required items have recorded handling states |

The remote 910B working directory contained additional checkout and build-cache material at the time of this snapshot. It is intentionally not committed here. Some original remote logs/profile artifacts remain to be recovered; affected records must retain `evidence_incomplete` rather than be promoted to `validated`.

## Status vocabulary

See [docs/status-definitions.md](docs/status-definitions.md). The authoritative current 910B matrix is under `ascend-910b/validation/reports/`.

## Safety

No credentials, browser captures, remote clone caches, or build products are included. One captured upstream source page contained temporary GitHub image signed-query parameters; those query parameters were removed in this publication copy while preserving the image path and PR provenance. Upstream provenance and license information are retained in the artifact bundles; review relevant upstream licenses before redistributing this repository publicly.
