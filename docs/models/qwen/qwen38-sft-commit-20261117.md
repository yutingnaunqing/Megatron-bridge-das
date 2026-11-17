# Qwen3.8 SFT 提交与迁移说明（2026-11-17）

本文是当前工作区的**待提交说明**，不是已提交记录。准备本文时没有执行 git add、git commit、git push，也没有启动训练。本次不增加运行环境变量开关、不调整训练默认值或默认目录。

## 提交目标和边界

保存当前 Qwen3.8-27B native 文本/VL SFT 实现、独立的单节点减层入口、数据准备与权重转换工具、测试和交接文档。接收服务器需要具有兼容的海光 DTK/PyTorch/Transformer Engine/Triton、MPI/网络插件及模型数据资源。

“环境一样”不包含模型权重、数据、节点名、共享挂载、SSH、网络拓扑和空闲显存必然相同。代码迁移不会自动准备这些外部资源；当前单节点日志仍有 OOM 和 hipErrorNoBinaryForGpu，不能将本次快照描述为所有机器均已训练验收。

## 本次实现包含哪些修改

| 范围 | 文件/目录 | 保存的功能 |
| --- | --- | --- |
| 文本训练入口 | `examples/qwen/run_qwen38_text_sft.sh` | 文本 SFT 独立入口，配置模型/数据/并行/缓存、MPI 和 torchrun，提供 prof 设置；训练不自动执行手工预检 |
| 单节点入口 | `examples/qwen/run_qwen38_text_sft_single_node.sh` | 独立配置、直接本机 torchrun；默认 16 层、4K、TP2/PP4、GBS32/MBS1、不重计算、prof 第 5–6 步 |
| VL 入口 | `examples/qwen/run_qwen38_vl_sft_full.sh` | VL 独立启动和配置，保留 prof；不与文本共用 MPI 包装入口 |
| Worker/观测 | `examples/qwen/qwen38_*worker.py`、`qwen38_mtp_audit.py`、`qwen38_training_health.py` | 缓存按 host/local-rank 隔离、GPU 绑定、模型/加载/退出证据及可选 MTP 监督审计；具体审计能力以各文件实现为准 |
| HCU recipe | `hcu_megatron_bridge/src/hcu_megatron/bridge/recipes/qwen38/`、`qwen38_vl/` 的清单文件 | 新增并导出文本/VL SFT recipe，读取环境配置，保持框架原生 RMSNorm/Adam 路径，不携带替代实现 |
| 数据 | `data_preparation/prepare_qwen38_text_sft.py`、`prepare_ultrachat_sft.py` | 会话格式准备、UltraChat 数据准备、可选 token/监督检查；不是训练时直接读取的离线 token 缓存 |
| 工具 | `tools/checkpoint/`、`tools/verify_ultrachat_parquet.py` 及包初始化文件 | VL→文本/减层权重转换、显存估算、手工预检、数据来源校验 |
| 测试 | `tests/unit_tests/qwen38_text_sft/`、`tests/fixtures/qwen38_text_sft/messages.jsonl` | 保留当前 CPU/静态回归和小型会话样例，不包含实际训练数据 |
| 文档 | 清单中的 `docs/models/qwen/*.md` | 使用说明、方案和历史交接；历史链接指向的 runs/outputs 证据不随 Git 提交 |
| 迁移与提交 | `patches/qwen38-sft/`、`tools/release/`、`hostfile.example` | 固定官方基线的补丁、应用工具、精确提交清单、提交脚本及节点模板 |
| Git 配置 | `.gitignore`、`.gitmodules` | 忽略 outputs；保留已跟踪的官方子模块来源。`.gitmodules` 当前没有内容变化，但必须随仓库保留 |

现有 `env/env.sh` 相对 HEAD 仅新增两行被注释的调试变量，不改变行为，建议本次不提交。该文件原先已跟踪，克隆后仍存在，当前启动所需逻辑不会丢失。

## 特别说明：官方 3rdparty 修改

`3rdparty/Megatron-Bridge` 是独立 Git 子模块，主仓库记录它的 commit ID，并不记录子模块内未提交的文件内容。只运行主仓库的 `git add .`/`git commit`，不能保存这些未提交内容。

本次选择保留官方 gitlink，不在子模块内创建仅本机可访问的 commit；将现有差异作为补丁提交在主仓库。这样不要求向 NVIDIA 官方仓库推送，也不需要先建立可访问的私有 fork。新服务器初始化子模块后须显式应用补丁。

固定基线：

| 官方仓库 | Commit |
| --- | --- |
| `3rdparty/Megatron-Bridge` | `51885cf132b2814188b6855c25a8588254274c2a` |
| `3rdparty/Megatron-Bridge/3rdparty/Megatron-LM` | `cd4afffa648426a959dc7cb1e24b5ce7d0c3ff54` |

当前官方目录实际只有两处文件差异：

1. **`3rdparty/Megatron-Bridge/src/megatron/bridge/recipes/__init__.py`**：增加 HCU recipe 导出和说明。它与主仓库已跟踪的 `hcu_megatron_bridge/src/hcu_megatron/bridge/recipes/__init__.py` 内容一致，`env/env.sh` 原有的复制逻辑也会生成这处变化。保存在 `patches/qwen38-sft/bridge-recipes.patch`，完整记录当前差异。
2. **`3rdparty/Megatron-Bridge/src/megatron/bridge/training/gpt_step.py`**：`use_mtp` 为真时，在 `forward_args` 中传入 `loss_mask`，让 MTP 目标移位时保留 SFT assistant-only 监督。共增加三行；MTP 关闭时不执行该分支。保存在 `patches/qwen38-sft/bridge-mtp-loss-mask.patch`。这处不会由 `env/env.sh` 自动恢复，迁移时尤其不能遗漏。

**嵌套的 Megatron-LM 当前没有本地修改**；没有改动其模型、并行调度或优化器源码。以上描述是相对当前子模块 HEAD 的完整工作区差异，不是“官方整个目录都由本项目修改”。

补丁应用工具逐项检查基线、已应用状态和冲突，可以重复运行；冲突时退出，不覆盖未知改动。应用后子模块显示 dirty 是本方案的预期状态，主仓库仍引用上述官方 commit。

## 提交文件和排除内容

**逐文件清单以 `tools/release/qwen38_sft_files.txt` 为准**，提交脚本只添加该清单，不使用 `git add .`。文末附有本次清单快照。

不提交：

- `outputs/`、`runs/`、prof trace、编译缓存、检查点、实际模型和数据。
- `examples/qwen/hostfile`：本机节点配置，目前写的是 `m09r2n05`/`m09r2n07`，不是你此前描述的 n05/n06；请按实际运行节点核对。新增 `hostfile.example` 供迁移使用，不修改现有 hostfile。
- `env/env.sh` 的两行注释变化。
- 任何未列入清单的改动、私钥、访问令牌、容器镜像及运行时安装目录。

默认模型/数据/TMPDIR 含本机绝对路径，按用户要求原样保留。另一服务器路径不同，需要设置已有环境变量或准备同路径挂载；TMPDIR 当前为脚本固定路径，需要保证该路径可写。缓存迁移不是启动的必要条件，但没有匹配缓存时会重新编译。

## 本机提交命令

先预览（不修改暂存区、不提交）：

```bash
cd /public/home/yugeng/work/Megatron-bridge-das-qwen38-sft-review
bash tools/release/commit_qwen38_sft.sh --preview
cat tools/release/qwen38_sft_files.txt
```

用户审阅后自行执行实际提交：

```bash
bash tools/release/commit_qwen38_sft.sh --commit
git show --stat --oneline HEAD
git status --short
```

脚本发现已有暂存内容会退出，避免把其他改动一并提交；执行提交前检查补丁快照是否仍与本机官方目录差异一致。检查失败不会自动覆盖文件。`--commit` 会执行 git add 和 git commit，不执行 push，不在子模块创建 commit。提交后 env 注释、本机 hostfile 和已打补丁的子模块仍可能显示为未提交变化，这是明确排除或补丁方案导致的预期结果。

## 另一台服务器部署

将本次根仓库 commit 推送到接收服务器可访问的仓库后：

```bash
git clone --recurse-submodules <你的仓库地址> Megatron-bridge-das-qwen38-sft-review
cd Megatron-bridge-das-qwen38-sft-review
# 若需要特定提交，先 git checkout <本次根仓库commit>，再同步其子模块：
git submodule update --init --recursive
bash tools/release/apply_qwen38_vendor_patches.sh --check
bash tools/release/apply_qwen38_vendor_patches.sh --apply
```

首次获取子模块需要能访问其官方 Git 来源，或提前准备好这两个固定 commit 的本地镜像。不要在迁移时执行 `git submodule update --remote`，它可能偏离本次基线。

确认模型/数据存在、8 张 GPU 可用、DTK 等运行环境一致后，可运行单节点入口；默认不依赖 hostfile：

```bash
bash examples/qwen/run_qwen38_text_sft_single_node.sh dryrun
bash examples/qwen/run_qwen38_text_sft_single_node.sh sft
```

`dryrun` 不是 GPU 训练验收，也不证明显存充足。若模型/数据路径不同，可通过 `LOCAL_HF_PATH`、`DATA_PATH` 覆盖；保持减层模型与 `EXPECTED_NUM_LAYERS=16` 一致。

双节点需在所有节点上部署相同代码并应用补丁，或使用共享代码目录；模型、数据和缓存路径需按当前脚本约定可访问。先复制模板并编辑真实节点名：

```bash
cp examples/qwen/hostfile.example examples/qwen/hostfile
# 编辑 examples/qwen/hostfile，替换 node-a/node-b。
# 核对 MPI_SSH_PORT（默认11463）、免密SSH、网卡/插件和共享路径。
bash examples/qwen/run_qwen38_text_sft.sh sft
# VL 使用自己的模型/数据资源：
# bash examples/qwen/run_qwen38_vl_sft_full.sh sft
```

本次制作提交材料只执行静态验证和补丁校验，不替用户启动上述训练，不把此前的历史测试视为新服务器验收。

## 本次提交清单快照

共 48 个候选路径，其中 `.gitmodules` 已跟踪且没有变化，不会因此产生新的文件差异。清单如下：

```text
.gitignore
.gitmodules
data_preparation/prepare_qwen38_text_sft.py
data_preparation/prepare_ultrachat_sft.py
docs/models/qwen/dapo-math-17k-for-sft.md
docs/models/qwen/parquet-sft-guide.md
docs/models/qwen/qwen38-27b-text-sft-review-20261113.md
docs/models/qwen/qwen38-sft-commit-20261117.md
docs/models/qwen/qwen38-sft-long-sequence-and-vl.md
docs/models/qwen/qwen38-text-sft-full.md
docs/models/qwen/qwen38-text-sft-handoff.md
docs/models/qwen/qwen38-text-sft-mtp-plan.md
docs/models/qwen/qwen38-text-sft-steady-plan.md
docs/models/qwen/qwen38-text-sft-ultrachat.md
examples/qwen/hostfile.example
examples/qwen/qwen38_mtp_audit.py
examples/qwen/qwen38_text_sft_worker.py
examples/qwen/qwen38_training_health.py
examples/qwen/qwen38_vl_sft_worker.py
examples/qwen/run_qwen38_text_sft.sh
examples/qwen/run_qwen38_text_sft_single_node.sh
examples/qwen/run_qwen38_vl_sft_full.sh
hcu_megatron_bridge/src/hcu_megatron/bridge/recipes/qwen38/__init__.py
hcu_megatron_bridge/src/hcu_megatron/bridge/recipes/qwen38/bw1000/__init__.py
hcu_megatron_bridge/src/hcu_megatron/bridge/recipes/qwen38/bw1000/qwen38_sft.py
hcu_megatron_bridge/src/hcu_megatron/bridge/recipes/qwen38/qwen38.py
hcu_megatron_bridge/src/hcu_megatron/bridge/recipes/qwen38_vl/__init__.py
hcu_megatron_bridge/src/hcu_megatron/bridge/recipes/qwen38_vl/bw1000/__init__.py
hcu_megatron_bridge/src/hcu_megatron/bridge/recipes/qwen38_vl/bw1000/qwen38_vl_sft.py
patches/qwen38-sft/bridge-mtp-loss-mask.patch
patches/qwen38-sft/bridge-recipes.patch
tests/fixtures/qwen38_text_sft/messages.jsonl
tests/unit_tests/qwen38_text_sft/test_full_workflow.py
tests/unit_tests/qwen38_text_sft/test_interop.py
tests/unit_tests/qwen38_text_sft/test_mtp.py
tests/unit_tests/qwen38_text_sft/test_recipe.py
tests/unit_tests/qwen38_text_sft/test_tools.py
tests/unit_tests/qwen38_text_sft/test_ultrachat_parquet.py
tests/unit_tests/qwen38_text_sft/test_worker_exit.py
tools/__init__.py
tools/checkpoint/__init__.py
tools/checkpoint/check_qwen38_text_sft.py
tools/checkpoint/convert_qwen38_vl_to_text.py
tools/checkpoint/estimate_qwen38_text_sft_memory.py
tools/release/apply_qwen38_vendor_patches.sh
tools/release/commit_qwen38_sft.sh
tools/release/qwen38_sft_files.txt
tools/verify_ultrachat_parquet.py
```

## 本次制作提交材料的实际改动与验证

新增上述提交说明、两个 release 脚本、精确文件清单、两个 Bridge 补丁和 `hostfile.example`；在 handoff 追加提交迁移说明。两个文本启动脚本各仅清理 PROFILE 赋值行的一个行尾空格，没有修改参数、目录或执行逻辑。官方目录没有新增改动，补丁仅导出现存差异。

验证：release/训练 shell 语法检查、提交脚本只读预览、补丁从固定官方基线在临时目录中正向应用和反向检查，并逐字节比对当前官方文件。没有启动 GPU 训练，没有执行实际暂存、提交或推送。
