---
id: technique-repository-source-snapshots
title: Provider API source snapshots for CUDA implementation evidence
type: technique
architectures: []
tags:
- cuda-cpp
confidence: source-reported
reproducibility: runnable
prerequisites: []
related:
- technique-jit-compilation
sources:
- repo-gitee-cuda-samples
- repo-gitcode-cuda-samples
- repo-gitcode-dgl-ascend
artifact_dir: artifacts/repos/gitcode/AI4Science__dgl-ascend
---

## Purpose

When a repository host blocks rendered pages, collect implementation evidence
through its documented content API, pin every request to a resolved commit,
and retain only selected source files with checksums. This preserves a
reproducible input for operator generation without treating a mirror as a
performance result.

## Minimal import contract

```python
def accept_snapshot(path, expected_sha256):
    import hashlib
    data = open(path, "rb").read()
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected_sha256:
        raise ValueError("source changed after capture")
    return data
```

## Caveats

- A fixed source commit is evidence of implementation, not proof of numerical
  correctness, target-SM compatibility, or performance.
- Public content endpoints and Git transport can have different access rules.
- Do not use HTML/WAF responses as source artifacts; record API failures and
  use a read-only platform token for authorized bulk discovery.
