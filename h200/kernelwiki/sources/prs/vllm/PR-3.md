---
id: pr-vllm-3
repo: vllm-project/vllm
pr: 3
title: Implement `single_query_cached_kv_attention` kernel
author: WoosukKwon
date: '2023-03-01'
url: https://github.com/vllm-project/vllm/pull/3
source_category: upstream-code
architectures:
- sm100
tags:
- attention
techniques: []
hardware_features: []
kernel_types:
- attention
languages:
- cuda-cpp
captured_at: '2026-07-23'
status: merged
merge_sha: 0deacbce
inclusion_reason: kernel file changes
changed_paths:
- cacheflow/master/block_manager.py
- cacheflow/models/attention.py
- cacheflow/worker/cache_engine.py
- cacheflow/worker/worker.py
- csrc/attention_kernels.cu
- csrc/cache_kernels.cu
- setup.py
- tests/kernels/attention.py
- tests/kernels/cache.py
artifact_dir: artifacts/prs/vllm/PR-3
---

## Summary

This PR adds the `single_query_cached_kv_attention` kernel.

Supported data types:
* `half`
* `float`

Tested models:
* OPT-125M
* OPT-350M
* OPT-1.3B
* OPT-2.7B
* OPT-6.7B
* OPT-13B

Tested GPUs:
* A100

## Problem

Implement `single_query_cached_kv_attention` kernel

## Changed Files

- `cacheflow/master/block_manager.py`
- `cacheflow/models/attention.py`
- `cacheflow/worker/cache_engine.py`
- `cacheflow/worker/worker.py`
- `csrc/attention.cpp`
- `csrc/attention_kernels.cu`
- `csrc/attention_utils.h`
- `csrc/cache_kernels.cu`
- `csrc/cuda_primitives.h`
- `setup.py`
- `tests/kernels/attention.py`
- `tests/kernels/cache.py`

