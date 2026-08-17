# Kernel Design Agent Knowledge Base

这是一个面向 kernel design 的**证据知识库**，保存上游 PR 的出处、实现摘录、实验记录和 Ascend 910B 验证证据。它不是一个可直接安装的算子包，也不把“已抓取”“runner 返回成功”和“严格正确性验证”混为一谈。

仓库当前以 H200 与 Ascend 910B 两条证据链并列组织；两者的语料、验证环境和结论彼此独立。

## 先从哪里读

| 想了解的问题 | 建议先看 | 内容 |
| --- | --- | --- |
| 910B 工作的简短概览 | [ASCEND_910B_KERNELWIKI_BRIEF_20260811.md](ascend-910b/validation/reports/ASCEND_910B_KERNELWIKI_BRIEF_20260811.md) | 采集范围、8 卡门禁和验证边界 |
| 910B 验证的阶段进度 | [ASCEND_910B_KERNELWIKI_PROGRESS_20260811.md](ascend-910b/validation/reports/ASCEND_910B_KERNELWIKI_PROGRESS_20260811.md) | 已完成工作、主要阻塞和后续方向 |
| 某个 PR 的日志、状态和错误在哪里 | [ASCEND_910B_VALIDATION_EVIDENCE_AND_ISSUES_GUIDE_20260812.md](ascend-910b/validation/reports/ASCEND_910B_VALIDATION_EVIDENCE_AND_ISSUES_GUIDE_20260812.md) | 文件地图、状态解释、问题分类和查阅方法 |
| 281 条 SGL PR 当前停在哪一阶段 | [ledger-8card.json](ascend-910b/validation/ledger-8card.json) | 机器可读总台账：SHA、状态、停止原因和错误分类 |
| 各状态的严格含义 | [status-definitions.md](docs/status-definitions.md) | `captured`、`validated`、`reference_required` 等定义 |

## 目录与文件说明

```text
.
├── h200/                         # H200 独立证据链
│   ├── kernelwiki/                # 上游 PR source、artifact 与检索材料
│   └── derived-validations/       # 基于上游证据开展的独立实验
├── ascend-910b/                   # Ascend 910B 独立证据链
│   ├── corpus/                    # 已公开的 sgl-kernel-npu PR 语料
│   ├── runs/                      # 爬取状态、未完成项与人工复核决定
│   └── validation/                # 8 卡运行框架、台账、单 PR 证据和报告
└── docs/                          # 跨目录状态定义
```

### `ascend-910b/corpus/`：上游 PR 证据，不是运行结论

当前公开快照包含 `sgl-project/sgl-kernel-npu`：384 个已合并 PR source page、281 个 implementation artifact bundle，以及 103 个仅 source/context 条目。

| 路径 | 每条记录包含什么 | 用途 |
| --- | --- | --- |
| `ascend-910b/corpus/sources/prs/sgl-kernel-npu/PR-<N>.md` | PR 标题、merge SHA、描述、上游链接等 | 确认上游来源、目标提交和改动背景 |
| `ascend-910b/corpus/artifacts/prs/sgl-kernel-npu/PR-<N>/diff.patch` | 合并 PR 的差异 | 查看完整改动 |
| `.../key-files/` | 从 diff 选出的关键源码/测试文件 | 快速阅读实现与入口 |
| `.../PROVENANCE.yaml` | PR、仓库、SHA、来源与采集信息 | 追溯 artifact 的来源 |
| `ascend-910b/runs/full-crawl-status.json` | 爬取状态汇总 | 判断采集是否完整 |
| `ascend-910b/runs/unfinished-prs-2026-07-28.txt` | source-only/context 或未形成 bundle 的条目 | 不能把这类 PR 当成待验证 kernel |
| `ascend-910b/validation/full-crawl-structure-validation.json` | corpus 结构校验结果 | 本快照中 `errors=[]` |

`source page` 只说明 PR 元数据已经归档；`artifact bundle` 只说明 diff、关键文件和 provenance 已归档。二者都不代表已经在 NPU 上运行成功。

### `ascend-910b/validation/`：8 卡验证证据

| 路径 | 内容 | 如何使用 |
| --- | --- | --- |
| `ledger-8card.json` | 281 条 bundle 的状态、merge SHA、阶段退出信息、`stop_reason`、错误分类 | 查某个 PR 当前结论时的首选入口 |
| `framework/execution-manifest-8card.json` | 每条的 exact SHA、execution kind、测试/launcher 选择 | 判断测试入口是否与目标提交匹配 |
| `framework/run_8card_batch.py` | 8 卡批量 runner | 理解构建、导入、测试和台账写回逻辑 |
| `framework/README.md` | framework 使用说明 | 需要复现 runner 流程时先读 |
| `multicard-gate/20260803T081037Z/` | 8 卡连通性、`npu-smi`、环境、torchrun/HCCL all-reduce 日志 | 确认基础多卡条件已通过 |
| `PR-<N>/` | 单个 PR 的 result、证据、部分条目的 multicard/diagnostics 运行目录 | 追到某一次实际运行 |
| `reports/` | 面向阅读者的汇总、分类和审计报告 | 先理解整体，再下钻日志 |

单 PR 的正式多卡尝试通常位于：

```text
ascend-910b/validation/PR-<N>/multicard/<timestamp>/
```

常见文件包括：`result.json`（机器可读阶段摘要）、`git-checkout.log`（实际 SHA）、`configure.log`、`build.log`、`import.log`、`correctness.log`、`environment.log` 和 `compat.patch`。并非每次尝试都会生成所有文件；文件是否存在本身也反映运行实际走到了哪个阶段。`diagnostics/` 下的小规模或最小复现只用于定位，不能替代正式 8 卡 correctness 结论。

### `h200/`：与 Ascend 分离的历史/实验材料

`h200/kernelwiki/` 保存 H200 使用的 PR 证据与检索资料；`h200/derived-validations/` 是受上游证据启发的独立实验。它们不应被解释为 Ascend 结果，也不应反推某个上游 PR 已被原样复现。

## 如何追查一条 SGL PR

以 PR-201 为例，建议按下面顺序阅读：

1. 打开 [`PR-201.md`](ascend-910b/corpus/sources/prs/sgl-kernel-npu/PR-201.md)，确认合并 SHA、描述和上游链接。
2. 打开 [`PR-201 artifact`](ascend-910b/corpus/artifacts/prs/sgl-kernel-npu/PR-201/)，查看 `diff.patch`、`key-files/` 和 `PROVENANCE.yaml`。
3. 在 [`ledger-8card.json`](ascend-910b/validation/ledger-8card.json) 搜索键名 `"201"`，读取 `status`、`stop_reason` 和错误分类。
4. 再进入 [`PR-201`](ascend-910b/validation/PR-201/)，按 `result.json` 指向的尝试查看 build/import/correctness 日志。该条还含 `diagnostics/`，其结果必须与正式多卡尝试区分。

在本地 clone 中可以这样查询：

```bash
jq '.rows["201"] | {pr, merge_sha, status, stop_reason}' \
  ascend-910b/validation/ledger-8card.json
rg -n -i 'error|exception|assert|hccl|aicore' \
  ascend-910b/validation/PR-201
```

## 当前公开快照的验证范围

对 `sgl-kernel-npu` 的 281 个 implementation bundle，8 卡 runner 台账目前记录：42 `validated`、64 `requires_resolution`、60 `build_failed`、28 `reference_required`、74 `not_applicable`、11 `test_entrypoint_invalid`、2 `correctness_failed`。8 卡门禁曾确认 `torch.npu.device_count()==8`，并以 8 rank HCCL all-reduce 通过。

这些状态首先是某次执行流程的阶段分类，不能脱离日志直接解读为上游实现质量：

- `validated` 仍需检查同目录是否存在 `compat.patch`，并核对测试入口与 reference；兼容迁移后的通过不等同于“历史 PR 原样复现”。
- `build_failed`、`requires_resolution` 和 `not_applicable` 不等同于 kernel correctness failed。
- benchmark/profile 不能替代 correctness；本仓库没有可用于跨 PR、跨算子或全仓库性能比较的结论。

更完整的状态机和环境/构建/测试/DeepEP/AICORE 问题解释见新增的[详细指南](ascend-910b/validation/reports/ASCEND_910B_VALIDATION_EVIDENCE_AND_ISSUES_GUIDE_20260812.md)。

## 发布边界

本次公开快照只同步了上述 SGL Ascend corpus 和相应的 910B 验证证据。工作区内后来采集的其他 Ascend 仓库尚未推送到本仓库，不能因为其他汇总材料提到它们，就误认为其 source/artifact 已在这里发布或已做 runtime validation。

仓库不包含 Cookie、HAR、token、远端 clone/build cache 或原始大型 trace。上游 provenance 与许可证信息保留在 artifact bundle 中；再分发前请核对相关上游许可证。
