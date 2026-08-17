# Ascend 910B 8 卡验证：证据地图、状态解释与问题分类

**更新时间：** 2026-08-12

**验证目标：** `ascend_910b_8card`
**适用范围：** 本公开快照中的 `sgl-project/sgl-kernel-npu` 281 个 implementation artifact bundle。

本文是查阅指南：source/artifact 用来证明上游改了什么；运行日志用来证明某次特定环境发生了什么。只有 exact SHA、干净构建、repository-local import、正确的 correctness/reference 和完整本地证据同时成立，才能使用严格的 `validated` 结论。

## 1. 当前结果应怎样理解

| 台账状态 | 数量 | 表示什么 | 不表示什么 |
| --- | ---: | --- | --- |
| `validated` | 42 | runner 记录了 build、repository-local import 和所选 correctness 命令成功 | 不自动等于原始环境、无补丁的上游复现 |
| `requires_resolution` | 64 | 已留下 import 或 correctness 阶段的具体日志 | 不等于 kernel correctness failed |
| `build_failed` | 60 | 当前环境或 PR-local 组件未完成构建 | 不等于上游代码一定无法构建 |
| `reference_required` | 28 | 没有可直接执行的数值对照入口 | 不等于实现错误 |
| `not_applicable` | 74 | 当前 SHA 缺合适测试入口，或需求超出当前资源/拓扑 | 不等于 kernel failed |
| `test_entrypoint_invalid` | 11 | 当前调用方式无效，例如 pytest 未收集到测试 | 不等于数值错误 |
| `correctness_failed` | 2 | 正确性断言非零，值得进一步最小化复现 | 不能只凭一次失败归因于内核 |
| **合计** | **281** | 每个完整 bundle 已进入台账 | 不产生整体性能结论 |

基础 8 卡条件已经通过，但这不代表每个历史算子都能在当前软件栈、8 卡拓扑和输入规模下运行。

## 2. 文件地图

### 2.1 调度、台账与环境门禁

| 路径 | 内容 | 何时查看 |
| --- | --- | --- |
| [`../ledger-8card.json`](../ledger-8card.json) | 281 条当前汇总：PR、SHA、状态、`stop_reason`、错误分类 | 首先回答“这条能否跑、卡在哪” |
| [`../framework/execution-manifest-8card.json`](../framework/execution-manifest-8card.json) | exact SHA、execution kind、测试文件与 launcher | 检查测试入口是否正确 |
| [`../framework/run_8card_batch.py`](../framework/run_8card_batch.py) | 实际批量执行器和状态写回逻辑 | 理解状态如何生成 |
| [`../multicard-gate/20260803T081037Z/`](../multicard-gate/20260803T081037Z/) | 8 卡连接、`npu-smi`、环境和 HCCL smoke 原始日志 | 复核多卡基础条件 |

门禁目录中的 `environment.log`、`torchrun-probe.log` 和 `hccl-smoke.log` 记录了：加载 CANN 环境、清理错误的可见卡变量后，`torch.npu.device_count()==8`，以及 `torchrun --nproc_per_node=8` 的 HCCL all-reduce 成功。该 smoke 不等同于具体 DeepEP/自定义 OPP 路径都已适配。

### 2.2 单 PR 目录

单个正式多卡尝试一般保存在：

```text
ascend-910b/validation/PR-<N>/multicard/<timestamp>/
```

| 文件 | 记录内容 | 用于判断 |
| --- | --- | --- |
| `result.json` / `summary.json` | PR、SHA、状态、各阶段退出码和停止原因 | 最快的机器可读摘要 |
| `git-checkout.log` | 实际 checkout 的提交 | 是否与台账/manifest 的 merge SHA 一致 |
| `configure.log`、`build.log` | CMake 与编译输出 | 头文件、编译器、链接问题 |
| `deepep-build.log` | DeepEP/custom OPP 构建 | 是否生成 PR-local `deep_ep_cpp` / OPP |
| `import.log` | repository-local import 检查 | 是否误导入系统安装包 |
| `correctness.log` | pytest、Python spawn 或 torchrun 输出 | 数值、HCCL、OPP、AICORE 失败的主要证据 |
| `environment.log` | 该次 CANN、Python、NPU 与环境变量快照 | 版本/环境差异 |
| `compat.patch` | 仅针对该 PR 的兼容迁移 | 存在时必须标为兼容迁移后运行 |
| `dispatch.*.log`、`state.json` | launcher 和调度过程 | 排查远端运行阶段中断 |

`diagnostics/` 目录保留定位性复跑。比如 `PR-201/diagnostics/` 与 `PR-209/diagnostics/` 的小规模结果明确为 `diagnostic_only`；它们不提升 8 卡默认形状的验证状态。

### 2.3 从 PR 编号下钻的固定顺序

以 PR-201 为例：

1. [`corpus source page`](../../corpus/sources/prs/sgl-kernel-npu/PR-201.md)：确认上游 PR、merge SHA 和描述。
2. [`artifact bundle`](../../corpus/artifacts/prs/sgl-kernel-npu/PR-201/)：阅读 `diff.patch`、`key-files/`、`PROVENANCE.yaml`。
3. [`ledger-8card.json`](../ledger-8card.json)：搜索键名 `"201"`，读取状态、停止原因和分类。
4. [`PR-201`](../PR-201/)：由 `result.json` 与实际日志复核 build/import/correctness 阶段。

在仓库根目录运行：

```bash
jq '.rows["201"] | {pr, merge_sha, status, stop_reason}' \
  ascend-910b/validation/ledger-8card.json
rg -n -i 'error|exception|assert|tiling|aicore|hccl' \
  ascend-910b/validation/PR-201
```

## 3. 执行流程和状态边界

```text
exact SHA checkout
  -> CMake configure/build
  -> repository-local import
  -> DeepEP PR-local OPP/deep_ep_cpp 检查（如适用）
  -> pytest / Python spawn / torchrun / reference precheck
  -> result.json 与 ledger 写回
```

`execution-manifest-8card.json` 将条目分为 `python_spawn`、`pytest`、`build_import_only`、`reference` 等入口。pytest exit code 5 或 “no tests collected” 会归类为 `test_entrypoint_invalid`；只有参考实现和断言条件已确认的 mismatch 才能归类为 `correctness_failed`。因此，台账首先是执行阶段分类，而不是对上游实现优劣的最后裁决。

## 4. 已见问题的分类与处理原则

| 类别 | 典型症状 | 可从日志直接确认的事实 | 处理原则 |
| --- | --- | --- | --- |
| 构建/工具链 | `acl_rt.h` 类型缺失、旧 CANN 头路径、SHMEM 头缺失、AscendC/CMake flags | 当前环境缺类型/头文件或构建不兼容 | 先做隔离软件栈或 PR-local compat patch；不能把 patch 后成功写成上游原样复现 |
| Python/Triton/GE | `extract_slice` 缺失、`No module named 'tbe'`、GE 初始化失败 | 当前解析到的包或 CANN Python 组件不满足入口要求 | 核对 wheel、`site-packages`、`PYTHONPATH` 和 `set_env.sh`，再做版本对照 |
| 测试入口/reference | pytest 未收集、CLI 参数/函数签名不匹配、无数值 reference | 当前调用方式不能有效比较输出 | 读取 exact SHA 的测试文件，确定真实 launcher；无 reference 时保留 `reference_required` |
| HCCL/DeepEP/OPP | world size 不匹配、`HCCL_BUFFSIZE`、custom op API/tiling 失败 | 某次 group、buffer、OPP 或形状不满足 | 先确认 world size、PR-local OPP、`deep_ep_cpp`、输入形状，再归因到内核 |
| AICORE/设备侧 | 507015 非法指令、507014 timeout | 该次设备执行异常，测试未完成 | 固定输入并跨卡/跨环境二分，不能单次就认定为 kernel bug 或硬件故障 |

代表性日志路径可按 PR 目录查找。例如 PR-157 的 Triton 报错在 `PR-157` 下的 `correctness.log`，PR-214 的 GE/TBE 记录在其早期 `correctness.log`，PR-201/209 的 AICORE 与诊断性小规模尝试分别保存在相应目录。请始终先以实际 `result.json` 和日志的时间戳为准。

## 5. 推荐的复核顺序

1. 固定环境快照：CANN、torch_npu、Triton、GCC、CMake、`npu-smi` 和 HCCL 变量。
2. clean checkout 下先解决 build 和 repository-local import；构建通过不等于 correctness 通过。
3. 对 `test_entrypoint_invalid` 按 exact SHA 找真实 CLI/torchrun/spawn 入口。
4. 对 DeepEP 先解决 custom OPP、world size、HCCL 与输入形状；要求 16 rank 的路径不能在 8 卡下伪装成已验证。
5. 对 reference-required 项补可信 I/O 对照；对 assertion mismatch 固定随机种子、dtype、输入和 reference 输出。
6. 仅在 correctness 与原始证据齐全后再做 benchmark/profile；时间打印不能单独支持性能结论。

## 6. 常见误读

- `artifact complete` 不等于已经运行或验证。
- `validated` 的当前 runner 标签不保证“无补丁、原样复现”；请检查 `compat.patch`、import 路径和 correctness 入口。
- `build_failed`、`requires_resolution`、`not_applicable` 都不等于上游 kernel correctness failed。
- 小规模 `diagnostic_only` 成功不等于正式 8 卡默认测试成功。
- 这里没有跨 PR、跨算子或全仓库的性能结论。

本仓库不包含 Cookie、HAR、token、远端 clone/build cache 或大型原始 trace；这些不属于可公开的验证证据。
