# 已沉淀仓库清单

更新时间：2026-07-17  
统计范围：`artifacts/prs/<repo>/PR-*/PROVENANCE.yaml` 中已完成原子提交、并可被 `scripts/validate.py` 验证的 PR 代码证据包。

## 当前已验证入库的上游仓库

| 知识库标识 | 上游 GitHub 仓库 | 主要沉淀内容 | 已验证 PR 证据包 |
|---|---|---|---:|
| `cutlass` | `NVIDIA/cutlass` | CUDA C++、CuTe、SM90/SM100 GEMM、FMHA、TMA、PDL、FP8/FP4 | 31 |
| `deepgemm` | `deepseek-ai/DeepGEMM` | FP8/FP4 GEMM、MoE 与 DeepGEMM 实现 | 1 |
| `flashinfer` | `flashinfer-ai/flashinfer` | CUDA/CuTeDSL、attention、MLA、GDN、GEMM、fused MoE、TRTLLM-Gen | 131 |
| `pytorch` | `pytorch/pytorch` | PyTorch CUDA/Inductor 相关的 Triton、TMA、WGMMA 接入证据 | 10 |
| `sglang` | `sgl-project/sglang` | CUDA/Triton、MLA、MoE、DeepGEMM、SM90/SM100/SM120 内核 | 66 |
| `vllm` | `vllm-project/vllm` | Triton、attention、MLA、MoE、NVFP4/FP8 算子路径 | 88 |
| **合计** | — | — | **327** |

## 全库快照

- 知识页：2,271
- 来源 ID：2,222
- 全部资产包：365
  - 上游原文（verbatim）：340
  - 提取代码（extracted）：13
  - 派生资产（derived）：12
- 候选账本：6 个（对应上表六个 GitHub 仓库）

这些数字由以下命令验证：

```bash
cd skills/KernelWiki
python3 scripts/validate.py
```

## 每条证据如何追溯

每一个表中的 PR 证据包均位于：

```text
artifacts/prs/<知识库标识>/PR-<编号>/
```

目录内的 `PROVENANCE.yaml` 固定了上游 PR URL、合并 commit SHA、获取日期、关键源码路径和 SHA-256。对应的检索页位于：

```text
sources/prs/<知识库标识>/PR-<编号>.md
```

例如 FlashInfer Router GEMM：

```text
sources/prs/flashinfer/PR-2323.md
artifacts/prs/flashinfer/PR-2323/PROVENANCE.yaml
```

## 当前未作为大规模来源的站点

- **Gitee / GitCode**：有公共仓库低频抓取试验与采集工具，但目前没有作为该正式知识库的大规模入库来源。原因是公开接口、反爬和访问频率限制使其难以提供与 GitHub token/API 相同的稳定性与可复现 provenance。
- **GitHub**：当前正式大规模来源。抓取使用认证 API，但只对脚本预筛和 AI 复核后的高价值 PR 获取关键源码；不是全站无差别爬取。

## 使用入口

```bash
# 结构与 provenance 校验
python3 scripts/validate.py

# 检索知识与对应来源页
python3 scripts/query.py "Blackwell router GEMM CUTLASS fused MoE" --limit 10 --compact

# 读取一个来源页及其证据目录
python3 scripts/get_page.py pr-flashinfer-2323
sed -n '1,120p' artifacts/prs/flashinfer/PR-2323/PROVENANCE.yaml
```
