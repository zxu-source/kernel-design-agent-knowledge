# 本周 Kernel Design Agent 知识库扩充与验证进展

生成时间：2026-07-29  
目标：在既有 KernelWiki 基础上，扩充 vLLM、SGLang 与 Ascend 算子证据，并逐步建立“可检索的设计知识 + 可追溯的运行验证”的 kernel design agent 知识库。  

本报告严格区分四层内容：

1. **source page**：上游 PR 的元数据、合并 SHA、描述和链接；
2. **artifact bundle**：可复查的 diff、关键实现文件与 provenance；
3. **派生算子验证**：基于上游证据独立实现的 H200 实验；
4. **原始 PR 复现**：在目标硬件按精确 SHA 构建、导入、correctness、benchmark/profile 的验证。

因此，“已爬取”不等于“已验证”，“派生算子跑通”也不等于“对应上游 PR 原样跑通”。

## 1. H200：vLLM / SGLang 爬取情况

本周 H200 侧的主要工作是扩充 KernelWiki 的上游代码证据库。知识库仍位于 `skills/KernelWiki/`，面向 NVIDIA/H200 路径，重点收集 attention、MLA、MoE、fused GEMM、量化和 Triton/CUDA 调度相关 PR。

| 上游项目 | 已归档 source page | 具备完整 artifact bundle | 未形成完整 bundle | 主要覆盖内容 |
|---|---:|---:|---:|---|
| vLLM | 1,364 | 1,298 | 66 | Triton、attention/MLA、MoE、NVFP4/FP8、调度与推理集成 |
| SGLang | 1,289 | 1,130 | 159 | CUDA/Triton、MLA、MoE、DeepGEMM、SM90/SM100 路径、kernel benchmark/test 组织 |
| **合计** | **2,653** | **2,428** | **225** | 用于后续按算子、硬件特性、实现语言和优化模式检索 |

完整 artifact bundle 至少包含 `diff.patch`、非空 `key-files/` 和 `PROVENANCE.yaml`。未形成完整 bundle 的 PR 保留 source page，不把它们伪装成已经抽取到完整实现；常见原因是超大 diff、历史内容获取超时或改动本身不含可独立抽取的实现文件。

### H200 语料对 kernel design agent 的作用

- 为 agent 提供真实上游实现与精确提交来源，而不只依赖抽象知识描述；
- 将“算子类别—实现文件—优化策略—硬件条件—来源 PR”关联起来，支持设计前检索；
- 可从 vLLM/SGLang 的 MoE、attention、量化和调度路径中提取可迁移的思路，同时保留 SM90、SM100 等硬件边界，避免跨架构直接照搬。

## 2. H200：验证情况

本周 H200 并没有开展 vLLM/SGLang 上游 PR 的大规模原样 checkout 复现，工作重心已转向 Ascend。已完成的是两条**基于 KernelWiki 上游证据的独立派生算子实验**：它们验证了设计思路在 H200 上的效果，但不等同于验证对应上游 PR。

| 派生实验 | 关联的上游证据 | H200 已完成验证 | 结论与限制 |
|---|---|---|---|
| FP16 row-wise softmax，warp-shuffle reduction | SGLang PR-8130 与层级 reduction 证据 | 64/64 correctness 通过；8 个 shape benchmark；相对 PyTorch `F.softmax` 为 1.75–2.13x | 自定义 CUDA 实现已完成 correctness/benchmark；不是 PR-8130 的原样复现 |
| BF16 ragged grouped GEMM，persistent queue | SGLang PR-9199、vLLM PR-25990，以及 FlashInfer/CUTLASS/DeepGEMM 证据 | 多种 expert/token 分布 correctness 通过；相对自定义 baseline 在 uniform/skewed/small 为 1.45x/2.07x/5.59x | mixed large case 为 0.47x，存在回退；没有 PyTorch/cuBLAS/CUTLASS 对照，且 H200 环境无 NCU，不能作通用性能结论 |

H200 直接证据：

- `rowwise_softmax_warp_reduce_h200/docs/final_report.md`
- `rowwise_softmax_warp_reduce_h200/outputs/validation.json`
- `rowwise_softmax_warp_reduce_h200/outputs/benchmark.csv`
- `ragged_grouped_gemm_h200/docs/final_report.md`
- `ragged_grouped_gemm_h200/outputs/validation.txt`
- `ragged_grouped_gemm_h200/benchmark.csv`

**H200 当前判断**：知识库爬取规模较大，已有少量“检索—独立设计—H200 correctness/benchmark”的闭环样本；但尚未对 vLLM/SGLang 的 2,428 个完整 PR bundle 做批量原样运行验证，也不应作此类表述。

## 3. Ascend：算子爬取情况

Ascend 语料独立存放在 `npu-kernelwiki/`，不混入 `skills/KernelWiki/` 的 NVIDIA/H200 标签、选择逻辑或验证记录。选择 `sgl-project/sgl-kernel-npu` 作为首个 Ascend 语料，是因为它是 SGLang 的官方 Ascend NPU kernel 库，覆盖 attention/MLA/GQA/decode attention、RMSNorm、SwiGLU、LoRA、KV cache、MoE/DeepEP 和量化路径。

| 项目 | 数量 | 状态 |
|---|---:|---|
| merged PR source page | 384 / 384 | 已归档 |
| 完整实现 artifact bundle | 281 | 已包含 diff、key-files、provenance |
| source-only/context PR | 103 | 已语义复核为文档、CI、测试、release、依赖、构建或配置等非直接算子实现 |
| 结构校验 | errors = [] | 通过 |

这 281 条是后续 Ascend 运行验证的候选池；103 条 source-only/context 不应被当作待验证 kernel。该分层将让后续 agent 能区分“可提取设计实现”“仅上下文变更”“已有运行证据”和“尚未验证”。

## 4. Ascend：910B 验证情况

### 4.1 环境与框架

真实环境为 Ascend 910B2C，CANN 9.0.0，Python 3.13.13，torch 2.11.0+cpu，torch_npu 2.11.0.rc4，GCC 13.3.0。NPU 可用性门禁已通过。验证框架支持每条 PR 独立 checkout、精确 merge SHA、构建、repository-local import、correctness、benchmark/profile、兼容 patch 隔离、stdout/stderr 保存、结果回传和断点续跑。

### 4.2 已验证与已执行分流

| 类别 | 数量 / PR | 当前可得结论 |
|---|---:|---|
| 已有严格真实性 correctness 基线 | PR-572 | 已在真实 910B 构建、导入并运行 upstream correctness；没有性能结论 |
| 兼容适配后的运行记录 | PR-431 | 经兼容适配完成构建、导入、correctness 及有限 benchmark/profile；不是原始环境无修改复现 |
| 流程阶段通过、证据仍待补齐 | PR-29、PR-35、PR-43 | build/import/correctness/benchmark/profile 流程通过；原始 profile 归档不完整，暂不升级为严格 `validated` |
| correctness-candidate 首轮 | 93 | 24 `FULL_PASS`、13 构建未完成、23 mismatch、4 test failed、26 需多卡、3 历史工程布局不兼容 |
| heuristic 覆盖探测 | 160 | 8 probe passed、49 probe failed、103 多卡或无可直接运行测试；probe 通过不等于 correctness 已验证 |
| reference-required 预检查 | 28 | 9 build smoke 通过、11 reference 预检查通过、8 失败；尚未形成完整 reference correctness 结论 |

### 4.3 失败和未验证原因

| 原因分类 | 影响范围 | 解释 | 正确处理方式 |
|---|---:|---|---|
| 历史 API / 工具链差异 | 构建或导入失败项 | 历史 Triton-Ascend、AscendC、CANN/ACL 接口与当前 CANN 9.0 栈不完全匹配 | 保存完整日志；按共同依赖聚类做兼容实验，不直接判为 kernel 错误 |
| 测试入口错误 | 部分 27 条 correctness 非通过 | 通用 pytest 命令不一定适用于历史脚本；PR-31/41 为无测试收集，PR-38 需专用分布式 launcher | 先恢复该 SHA 的正确 test entrypoint，再判断数值正确性 |
| 单卡环境限制 | 26 条候选及 probe 中的部分条目 | DeepEP/intranode/低时延通信路径需要多 NPU 或指定拓扑 | 标记为“单卡不适用/待多卡”，不是 correctness failed |
| 历史工程布局差异 | 3 条 | 历史提交早于当前 CMake 工程组织 | 使用该 revision 对应构建路径，或保留 pre-bootstrap 状态 |
| 无直接 correctness reference | 28 条 reference-required 及部分 probe | 版本、配置、license、通信骨架等不天然提供独立输入输出对照 | 仅做 build/import smoke，或补建 reference；不得强行套用通用测试 |
| 证据回传不完整 | 多数 `FULL_PASS` | runner 返回成功不等于本地已拥有完整 stdout/stderr、benchmark 和 raw profile | 补齐远端原始证据后才可升级验证状态 |

### 4.4 910B 耗时判断

准备好的单次自动尝试中，`FULL_PASS` 的中位墙钟约 91 秒，构建失败约 72 秒，correctness mismatch 约 117 秒。这只是“执行到当前阶段”的时间；不包括历史环境适配、找正确入口、补 reference、多卡排队、重跑和证据整理。

因此：

- 仅完成全量可审计分类和日志归档：预计仍需约 **30–70 人工小时**；
- 将单卡可验证 PR 尽量推进到严格 correctness 结论：保守估算约 **250–790 人工小时**；
- 多卡路径必须获取多卡资源后单独排队，单卡环境下最严谨的结果是“环境限制，未执行分布式 correctness”。

## 5. 面向 Kernel Design Agent 的后续计划

### 阶段 A：整理知识库的统一证据模型

1. 保持 H200 `skills/KernelWiki/` 与 Ascend `npu-kernelwiki/` 两个架构命名空间隔离。
2. 为每条知识增加统一状态：`captured`、`artifact_complete`、`runnable`、`validated`、`failed`、`not_applicable`、`evidence_incomplete`。
3. 在检索结果中同时返回：算子类型、硬件/软件前提、关键文件、上游 SHA、可用测试入口、reference、已知失败原因和证据路径。

### 阶段 B：把爬取结果变成可用于设计的检索层

1. 对 H200 语料按 GEMM、MoE、attention/MLA、norm、KV cache、量化、调度等建立索引和设计模式摘要。
2. 将 H200 的派生实验回写为“证据驱动的设计案例”：输入约束、采用的上游思路、实现选择、correctness、benchmark、失败模式和适用边界。
3. 对 Ascend 281 条 bundle 做同样的算子/依赖/测试入口分类，但不引入 NVIDIA 架构标签。

### 阶段 C：分层推进 Ascend 验证

1. 先补齐 PR-29、PR-35、PR-43 和其余 `FULL_PASS` 的原始日志/profile 证据。
2. 优先处理单卡、有明确 reference、可复用兼容策略的候选；先校正测试入口，再处理数值问题。
3. 将构建失败按 CANN、ACL、AscendC、Triton-Ascend 和工程布局聚类，避免逐 PR 重复检修。
4. 将多卡和无 reference 项形成独立待办；在有多卡资源或补好 reference 前，保留“未适用/待验证”，不伪造失败或成功。

### 阶段 D：形成 agent 闭环

未来 kernel design agent 的一次工作流应为：**按算子和硬件约束检索证据 → 选择可迁移设计模式 → 生成候选实现与测试计划 → 在目标硬件跑 correctness → 仅对 correctness 通过的候选跑 benchmark/profile → 将结果和失败原因回写知识库**。这样知识库不是静态 PR 仓库，而是能持续积累“什么设计在什么硬件/软件栈下可行、为什么失败”的经验库。

## 6. 本周结论

本周优先完成了 Ascend 的大规模 PR 爬取和 910B 批量验证框架建设，因此 H200 的工作以 vLLM/SGLang 证据扩充和少量派生算子闭环为主，并未开展 H200 上游 PR 的全量复现。当前最重要的成果是：H200 和 Ascend 已分别拥有可追溯的上游代码证据；Ascend 已具备真实 910B 的批量运行、失败保留和断点续跑能力；下一步应把两侧证据统一成面向设计决策的检索与反馈闭环，而不是把“抓到 PR”直接等同于“算子已验证”。
