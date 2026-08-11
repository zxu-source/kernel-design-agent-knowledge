# Ascend 910B 多卡算子知识库与验证进展

**汇报日期：** 2026-08-11  
**目标环境：** Ascend 910B 8 卡目标 `ascend_910b_8card`  
**工作目标：** 建立可追溯的 Ascend 算子 PR 知识库，并在真实 910B 环境中逐步验证其可构建性、导入能力和数值正确性。本文将“代码归档”和“运行验证”严格分开。

## 技术摘要

目前已完成 6 个 Ascend 相关仓库的 PR/MR 归档，共 **18,033** 条 merged PR/MR source page 和 **12,291** 个完整 implementation artifact bundle；各仓库结构校验均为 `errors=[]`。其中仅 `sgl-project/sgl-kernel-npu` 的 **281** 个 bundle 已进入真实 910B 8 卡执行台账。该台账已覆盖全部 281 条，但结果主要反映当前 CANN/torch_npu/测试入口/分布式拓扑组合下的可运行性，不等于所有 PR 都已经得到严格正确性认证。

8 卡基础门禁已通过：`torch.npu.device_count()==8`，并在 `torchrun --nproc_per_node=8` 下完成 HCCL all-reduce（结果为 36）。当前的主要瓶颈不是“没有卡”，而是软件栈版本兼容、DeepEP 自定义 OPP 构建、测试入口和部分设备侧 AICORE 失败。下一阶段应优先建立隔离的兼容软件栈并按失败类别回归，而不是继续以同一环境盲目重复所有 PR。

## 已完成的 PR 采集与知识库建设

采集流程为：merged PR/MR source page → merge diff 与关键代码文件的 artifact bundle → provenance 记录 → 结构校验。`artifact complete` 仅表示源码证据完整，不表示算子已在 910B 上运行；其中 `ops-transformer` 与 `ops-nn` 为 GitCode merged MR，其余为 GitHub merged PR。

| 上游仓库 | PR/MR source page | 完整 artifact bundle | source-only / context | 运行验证状态 |
| --- | ---: | ---: | ---: | --- |
| `sgl-project/sgl-kernel-npu` | 384 | 281 | 103 | 已完成 8 卡执行台账 |
| `vllm-project/vllm-ascend` | 5,700 | 3,487 | 2,213 | 尚未启动运行验证 |
| `huawei-csl/pto-kernels` | 111 | 80 | 31 | 尚未启动运行验证 |
| `triton-lang/triton-ascend` | 645 | 473 | 172 | 尚未启动运行验证 |
| `cann/ops-transformer` | 6,030 | 4,301 | 1,729 | 尚未启动运行验证 |
| `cann/ops-nn` | 5,163 | 3,669 | 1,494 | 尚未启动运行验证 |
| **合计** | **18,033** | **12,291** | **5,742** | 仅 SGL 已进入运行阶段 |

六个仓库的结构校验均通过（`errors=[]`）。source-only/context 项主要是文档、CI、配置、测试、构建改动或在合并版本中已删除的代码，不应被误称为待验证 kernel。

## 910B 8 卡基线与执行覆盖

验证使用独立的 `ascend_910b_8card` 配置，而非历史单卡目标。门禁记录显示：加载 CANN 环境并清除 `ASCEND_VISIBLE_DEVICES` 与 `ASCEND_RT_VISIBLE_DEVICES` 后，`torch.npu.is_available()` 为真、逻辑设备数为 8；设置进程级 `HCCL_INTRA_ROCE_ENABLE=1` 后，8-rank HCCL all-reduce 成功。该设置反映当前 8 卡跨 plane 拓扑的实际要求。

对 SGL 的 281 个完整 bundle，台账按真实入口执行：94 条 `python_spawn`、79 条 `pytest`、80 条 build/import-only、28 条 reference 路径。当前自动台账状态如下：

| 自动状态 | 数量 | 当前含义 |
| --- | ---: | --- |
| `validated` | 42 | 自动流程记录 build/import/correctness 成功；仍须逐条审计兼容补丁、精确入口和证据完整性，不能一概等同于上游原样复现 |
| `requires_resolution` | 64 | 已进入测试，但受环境、通信、OPP 或运行时问题阻断 |
| `build_failed` | 60 | 当前工具链或 PR-local 自定义组件构建未完成 |
| `reference_required` | 28 | 没有可直接套用的数值 reference，需独立设计验证方式 |
| `not_applicable` | 74 | 精确 SHA 下缺合适测试/运行入口，或需要超出当前资源的拓扑 |
| `test_entrypoint_invalid` | 11 | 现有调用方式无效，例如 pytest 未收集到参数化脚本 |
| `correctness_failed` | 2 | 在当前正确入口下得到非零 correctness 结果，仍需区分测试与内核原因 |

这些状态相加为 281。特别需要注意：台账中的 `validated` 是当前自动 runner 的状态标签；部分运行使用了 PR 隔离的兼容迁移补丁，因此后续会按“精确 SHA → clean build → repository-local import → 正确 reference/correctness → 完整证据”的标准复核后，才作为严格 validated 对外表述。

## 已观察到的关键问题（根因仍需分项复核）

1. **构建头文件兼容问题。** 构建日志可见 `acl_rt.h` 中 `aclmdlRITask` 类型缺失。PR-142 的记录显示兼容迁移后 build/import 均为 0；这支持“当前头文件组合不兼容”的判断，但尚不足以单独证明完整版本因果链。该迁移不是原始上游环境复现。

2. **Triton Ascend 扩展接口缺失。** 历史运行日志反复出现 `extract_slice` 等接口不存在；这支持“已安装包与所需 Ascend 扩展不一致”的诊断。仍需在当前远端环境重新核对 wheel、`RECORD` 与磁盘文件后，才能将其确认为版本覆盖根因。

3. **DeepEP 自定义 OPP、HCCL 配置与分布式规模共同影响。** 原始日志包含 `HCCL_BUFFSIZE too SMALL`，runner 已将 HCCL buffer 配置和 PR-local OPP 构建隔离处理。PR-201 与 PR-209 在 2-rank、小输入诊断下均以 `rc=0` 完成，但这只说明最小合法分布式条件下可运行，**不证明** HCCL buffer 已单独解决 8-rank 默认规模失败，也不构成性能结论。

4. **设备侧 AICORE/tiling 故障需要与 world-size 问题分离。** 先前 PR-201/209 的 8-rank 日志同时包含 launcher 文本和 507014/AICORE 信息，旧分类器可能将其宽泛标为 world-size。现已调整分类优先级，使 AICORE、`execute kernel param invalid`、`fftsplus`、507014 等设备侧签名独立归类；历史记录不直接改写，需保留原日志后再复核。

5. **部分工作本身超出 8 卡资源边界。** DeepEP 的若干 C++/MoE 路径以 16-rank 或特定拓扑为前提。对这些条目，8 卡不能验证不等于 kernel correctness failed；应保留为资源/拓扑受限，并在具备相应资源时再验证。

## 当前证据边界

- 已有的 source page、diff、关键文件和 provenance 可支持回溯“哪个 PR 改了什么”，但不能证明性能或正确性。
- PR-572 存在独立的 repository-local import 与 upstream correctness 历史基线；其后一次 8 卡批量 pytest 调用未收集到测试，不能覆盖或否定该历史证据，后续需要统一测试入口记录。
- 部分 PR 留有 benchmark/profile 尝试记录，但原始 profile 或可比较基线并不齐全。**当前不对任何仓库、算子或 PR 作性能提升结论。**
- 本文的台账数字、8 卡门禁、PR-201/209 诊断退出码和 PR-572 历史 correctness 文件已直接核对；环境问题章节中的“根因”仍属于日志支持的诊断假设，需通过隔离环境与对照重跑才能升级为确定因果。

## 下一阶段计划

1. **固定并审计环境基线。** 将 CANN、torch_npu、Triton Ascend、GCC/CMake、HCCL 变量写成可复现快照；优先在隔离环境中获得匹配 Triton/CANN 组件，避免污染共享系统环境。

2. **按根因分组回归，而不是逐 PR 手工修补。** 先处理可验证的构建头文件兼容、HCCL buffer 和 PR-local OPP 构建；再分别处理 Triton API 缺失、GE 初始化、dtype 不匹配和 AICORE 异常。

3. **纠正测试入口与状态分类。** 对 `python_spawn`、`torchrun`、pytest、reference 四类入口逐条确认；把“pytest 未收集”“16-rank 资源不足”“设备 AICORE 错误”和“数值 mismatch”分开记录。

4. **建立严格 validated 队列。** 对具备正确 reference 的候选，执行 clean build、repository-local import、正确 correctness，然后归档 stdout/stderr、退出码、环境快照和兼容补丁。兼容迁移成功将明确标为“迁移后运行”，不与原样复现混淆。

5. **最后才做性能工作。** 只有 correctness 通过且 benchmark/profile 原始证据完整时，才进行统一基线下的性能与 profile 对比；在此之前不发布性能结论。

## 可审计证据位置

- 多仓库 inventory：`docs/ASCEND_CORPUS_INVENTORY.md`
- SGL 8 卡台账：`validation/ascend-910b/ledger-8card.json`
- 8 卡门禁：`validation/ascend-910b/multicard-gate/20260803T081037Z/`
- 单 PR 运行日志：`validation/ascend-910b/PR-<N>/multicard/<timestamp>/`
- 环境问题分析：`validation/ascend-910b/reports/ENVIRONMENT_ISSUES_ANALYSIS_20260806.md`
- PR-201/209 2-rank 诊断：`validation/ascend-910b/PR-{201,209}/diagnostics/20260811T*/`
