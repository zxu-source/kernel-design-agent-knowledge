# Ascend 910B 多卡算子知识库简报

**日期：** 2026-08-11  
**验证目标：** `ascend_910b_8card`（独立 8 卡配置）

## 当前总体情况

| 项目 | 当前结果 | 说明 |
| --- | ---: | --- |
| 已采集 Ascend 相关仓库 | 6 个 | SGL、vLLM-Ascend、PTO kernels、Triton-Ascend、ops-transformer、ops-nn |
| merged PR/MR source page | 18,033 | GitHub PR 与 GitCode MR 的上游元数据、链接已归档 |
| 完整 implementation artifact bundle | 12,291 | 含 diff、关键代码文件与 provenance；不等于已验证 |
| source-only / context | 5,742 | 文档、CI、配置、测试或删除型变更，不作为待验证 kernel |
| 结构校验 | 6/6 通过 | 所有仓库 validator 均为 `errors=[]` |
| 已进入真实 910B 运行验证 | 281 条 | 目前仅 `sgl-project/sgl-kernel-npu` 全量进入 8 卡执行台账 |

## SGL 281 条 PR 的 8 卡执行结果

8 卡门禁已通过：`torch.npu.device_count()==8`，并完成 8-rank HCCL all-reduce（sum=36）。SGL 条目按精确 merge SHA 与对应测试入口运行，当前自动台账如下。

| 状态 | 数量 | 含义 |
| --- | ---: | --- |
| `validated` | 42 | 自动流程记录 build/import/correctness 成功；仍需审计是否用了兼容补丁，不能统一表述为原样上游复现 |
| `requires_resolution` | 64 | 已运行到测试，但受环境、通信、OPP 或运行时问题阻断 |
| `build_failed` | 60 | 当前工具链或 PR-local 组件未构建成功 |
| `reference_required` | 28 | 缺少直接数值 reference，需另设验证方法 |
| `not_applicable` | 74 | 当前 SHA 缺入口，或需要超过 8 卡的拓扑/资源 |
| `test_entrypoint_invalid` | 11 | 测试脚本不能按当前 launcher 直接调用 |
| `correctness_failed` | 2 | 正确入口下出现非零结果，仍待区分内核与测试原因 |
| **合计** | **281** | 全量执行覆盖完成 |

## 验证中发现的主要问题

| 问题类别 | 表现 | 当前判断与处理方向 |
| --- | --- | --- |
| 构建头文件兼容 | `acl_rt.h` 缺类型，部分 PR 无法构建 | 兼容迁移后部分 build/import 通过；完整版本因果仍需隔离对照确认，不能把迁移成功写成原样复现 |
| Triton Ascend 扩展接口 | `extract_slice` 等 API 不存在 | 历史日志支持扩展包不一致的诊断；需在当前环境核对 wheel、RECORD 与磁盘文件 |
| DeepEP 自定义 OPP 与 HCCL | OPP 缺失、tiling 或 `HCCL_BUFFSIZE` 报错 | 已改为每 PR 隔离构建 OPP，并按 HCCL/tiling/运行时故障分别记录；尚未证明单一变量可解决所有 8 卡失败 |
| AICORE / 设备侧错误 | 507014、tiling、参数非法等 | 已避免误归类为 world-size；需以最小规模诊断与 8 卡复现区分内核、规模和环境原因 |
| 资源与测试入口限制 | 部分 DeepEP 路径需要 16-rank 或特定拓扑；部分 pytest 未收集 | 不计为 kernel correctness failed；保留为资源受限或测试入口待修正 |

PR-201 与 PR-209 已在 2-rank、小输入的诊断条件下运行成功，但这仅说明最小合法分布式规模可运行，**不等于** 8-rank 默认规模 correctness 通过，也没有性能结论。

表中的台账计数、8 卡门禁和上述 2-rank 退出码已直接核对。环境问题的“根因”是基于日志的诊断结论，仍需隔离环境与对照重跑后才能作为确定因果汇报。

## 下一步

1. 固定并隔离匹配的 CANN、torch_npu、Triton-Ascend 环境；
2. 按上述根因分组回归 64 条 `requires_resolution` 和 60 条 `build_failed`；
3. 修正测试入口并为 reference-required 条目补充正确对照；
4. 仅在“精确 SHA、clean build、repository-local import、正确 correctness/reference、完整证据”都满足后升级为严格 `validated`；
5. correctness 稳定后再进行 benchmark/profile。当前**无跨 PR、跨算子或全仓库性能结论**。

**详细报告：** `reports/ASCEND_910B_KERNELWIKI_PROGRESS_20260811.md`  
**核心台账：** `validation/ascend-910b/ledger-8card.json`
