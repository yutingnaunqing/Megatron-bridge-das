# Qwen3.8-Flash-Next 在海光 BW1000 上的 Transformers 复现手册

更新日期：2026-09-23。本文只使用 **上游 Transformers 的 `Qwen4ExpForCausalLM` 文本模型**，不依赖本仓库的 Megatron 自定义模型。按“环境 → 单卡参考 → 单卡优化 → 八卡 DDP → 八卡 TP”顺序运行。所有已通过的训练测试使用**缩小配置、随机初始化权重和随机 token**；通过表示前向、loss、反向及参数更新能执行，不表示官方 125B+51B 权重已加载、模型能正确回答问题，或训练性能达标。

## 1. 先认清模型与并行方式

[官方模型卡](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)给出的主文本模型为 48 层、hidden size 2560，层型按 `GDN → GDN → GDN → QSA` 循环；PLE **只附加在第 2 层 GDN**，不会每四层重复。每层还含 MoE 和 Gated Residual。模型卡列出约 125B 主模型参数、51B N-gram embedding 和 4B MTP。当前脚本不覆盖视觉、MTP、真实权重、生成 cache、长上下文质量或真实数据训练。

| 方式 | 本目录脚本 | 每张卡持有什么 | 已验证 |
|---|---|---|---|
| 单卡 | `transformers_smoke.py` | 缩小后的完整模型 | BF16/FP32 参考路径；BF16 `grouped_mm`/FLA |
| 8 卡 DDP | `transformers_ddp8_smoke.py` | 每卡一份完整缩小模型 | 前反向、梯度 All-Reduce、更新 |
| 8 卡 TP | `transformers_tp8_smoke.py` | 官方 TP 规则切分的部分参数，另有部分复制参数 | 单处 PLE、MoE、GDN、QSA 路径的前反向和更新 |
| 2 卡 PP 探针 | `transformers_pp_probe.py` | 通用 PP 能分层加载 | 当前 Qwen4Exp **前向失败**，见第 7 节 |

DDP 不能让大模型“每卡只占八分之一”；TP 能切分部分矩阵，但既不能保证全部权重等分，也不自动实现流水线并行（PP）。

## 2. 环境和源码

已验证机器：8 × **BW1000 64G**；容器名 `yg_qwen3.8`。容器镜像标签为 `10.16.1.152:5000/jenkins/model_test_env/mbridge:0.6.0-ubuntu22.04-dtk2604-py3.12-20260909-1017`。版本：Python 3.12.7、PyTorch 2.10.0、HIP 6.3.26113、Transformers 5.18.0.dev0；本地 Transformers 源码提交为 `8e6083374735b8c95f13b1cf2f80cd9c7486c389`。FLA 0.5.1 与 TileLang `0.1.12+das.opt1.dtk2604.torch2100.2609081633.g25d759` 用于加速路径。换机器时，优先复用相同容器及源码提交，再逐项比较差异；不要把旧 Transformers 5.11.0 的失败误认为 DCU 算子失败，它没有 `qwen4_exp` 模型类。

在**宿主机**进入容器：

```bash
docker exec -it yg_qwen3.8 bash
```

以下命令均在**容器内**的项目根目录执行。这个容器把宿主机 `/public2/home/yugeng/work` 挂载为 `/public/home/yugeng/work`：

```bash
cd /public/home/yugeng/work/Megatron-bridge-das
export PYTHONPATH=/public/home/yugeng/work/transformers_qwen38/src:${PYTHONPATH}

python - <<'PY'
import torch, transformers
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpForCausalLM
print('torch:', torch.__version__, 'HIP:', torch.version.hip)
print('transformers:', transformers.__version__, transformers.__file__)
print('DCU:', torch.cuda.device_count(), torch.cuda.get_device_name(0))
print('model:', Qwen4ExpForCausalLM.__name__)
PY

git -C /public/home/yugeng/work/transformers_qwen38 rev-parse HEAD
```

期望 `transformers.__file__` 指向 `transformers_qwen38/src`，`torch.cuda.device_count()` 至少为计划使用的卡数。ROCm/HIP 构建的 PyTorch 仍使用 `device="cuda"` 和 `torch.cuda` 接口。源码目录应与项目并列；无需移动系统 `site-packages`。如果在新环境安装源码，可从[该官方提交](https://github.com/huggingface/transformers/commit/8e6083374735b8c95f13b1cf2f80cd9c7486c389)检出相同版本，再用 `PYTHONPATH` 指向其 `src`：

```bash
git clone https://github.com/huggingface/transformers.git /public/home/yugeng/work/transformers_qwen38
git -C /public/home/yugeng/work/transformers_qwen38 checkout 8e6083374735b8c95f13b1cf2f80cd9c7486c389
```

这些命令只获取 Transformers 源码；PyTorch/DTK 仍须使用适配 DCU 的构建，不能随意替换成普通 CUDA wheel。已有源码目录无需重复克隆。

## 3. 单卡：先建立可用参考路径

先运行最小 BF16 文本模型。这会构造官方类，并走 `eval → 训练前向/loss → backward → AdamW step`：

```bash
HIP_VISIBLE_DEVICES=0 python -u examples/qwen38/transformers_smoke.py \
  --profile tiny --device cuda --dtype bfloat16 --attention eager \
  --experts batched_mm --gdn torch_reference \
  --seq-len 32 --steps 1 \
  --output runs/qwen38/hf_tiny_bf16_reference_verified
```

`batched_mm` 是 Transformers 的 MoE 参考矩阵乘路径；`torch_reference` 选择上游 GDN 的 PyTorch 实现。它们用于把模型结构问题与可选加速 kernel 问题分开。已实测输出 `PASS: reduced official text model forward/loss/backward/update`，参数数 **1,354,488**，loss 约 6.2493。这个小模型有 GDN、QSA、MoE、PLE 和 Gated Residual，未加载官方权重。

查看结果：

```bash
cat runs/qwen38/hf_tiny_bf16_reference_verified/result.json
```

正常应有 `"status": "PASS"`、`losses` 和峰值显存。`parameters_without_grad` 中 3 个 QSA indexer 参数没有纯 LM loss 梯度，因为离散 Top-K 选块本身不可微；不能把它当成整模型反向失败。

## 4. 单卡：逐条恢复加速路径并扩大尺寸

### 4.1 MoE GroupedGEMM

先隔离底层算子。`offs` 必须是 **int32 累积结束位置**，脚本已处理：

```bash
HIP_VISIBLE_DEVICES=0 python examples/qwen38/grouped_mm_probe.py \
  --m-per-group 16,16,16,16 --k 128 --n 128 \
  --dtype bfloat16 --backward \
  --output runs/qwen38/grouped_mm_balanced_bf16
```

这组形状的 BF16 前反向在 BW1000 **PASS**。改为 `--dtype float32`，同形状实测 `hipblaslt_groupedgemm: no valid algorithm found`。这说明问题与 dtype/可用算法覆盖有关，不能概括成“GroupedGEMM 都不支持”或“Qwen 架构不支持”。

将 BF16 `grouped_mm` 带回模型：

```bash
HIP_VISIBLE_DEVICES=0 python -u examples/qwen38/transformers_smoke.py \
  --profile tiny --device cuda --dtype bfloat16 --attention eager \
  --experts grouped_mm --gdn torch_reference \
  --seq-len 32 --steps 1 \
  --output runs/qwen38/hf_tiny_bf16_grouped_mm_reference_gdn
```

### 4.2 GDN FLA/TileLang

在 MoE BF16 可用后，把 GDN 切到加速路径：

```bash
HIP_VISIBLE_DEVICES=0 python -u examples/qwen38/transformers_smoke.py \
  --profile tiny --device cuda --dtype bfloat16 --attention eager \
  --experts grouped_mm --gdn fla \
  --seq-len 32 --steps 1 \
  --output runs/qwen38/hf_tiny_bf16_grouped_mm_fla
```

当前组合的前向、反向和更新已通过。早期曾在 FLA/TileLang backward 报 `hipModuleLaunchKernel ... invalid argument`；那是特定版本/形状下的优化 kernel 失败，不能据此说 GDN 不支持。单次运行可能首次编译 TileLang kernel，等待较长；`causal_conv1d` 未安装时会提示退回 PyTorch 参考实现，这是**性能提示**，并非失败。

扩大到**官方宽度**，但仍只用 4 层、16 专家、缩小 N-gram 表和词表：

```bash
HIP_VISIBLE_DEVICES=0 python -u examples/qwen38/transformers_smoke.py \
  --profile width --device cuda --dtype bfloat16 --attention eager \
  --experts grouped_mm --gdn torch_reference \
  --seq-len 1024 --steps 2 \
  --output runs/qwen38/hf_width_bf16_grouped_mm
```

此配置已实测 PASS，loss 约 12.540 → 12.422。`--gdn fla --seq-len 128 --steps 1` 也通过。实际参数量约 **1,974,775,200**；`width` 不是完整 125B 模型，配置变更以输出的 `effective_config.json` 为准。

## 5. 八卡 DDP：验证通信与同步更新

```bash
HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=1 \
torchrun --standalone --nproc_per_node=8 \
  examples/qwen38/transformers_ddp8_smoke.py \
  --profile width --layers 8 --dtype bfloat16 \
  --attention eager --experts grouped_mm --gdn fla \
  --seq-len 128 --local-batch-size 1 --steps 2 \
  --output runs/qwen38/ddp8_width_bf16_grouped_mm_fla
```

已实测 `PASS: 8-rank DDP forward/loss/backward/all-reduce/update`，全局平均 loss 约 9.160 → 8.957。`result.json` 记录环境、world size、loss 和峰值显存；失败时检查 `rank<N>_error.txt`。`torchrun` 最后的 `ProcessGroupNCCL ... destroy process group` 日志只表示进程组清理；在 HIP 构建中仍常显示 NCCL 名字。

此 DDP 脚本的 8 层缩小模型在第 2、6 层各放一张 PLE 表，以增加测试覆盖；**官方完整配置只有第 2 层一张**。DDP 成功不等于两张 PLE 表的 TP 反向成功。

## 6. 八卡 TP：真实切分部分模型参数

TP 脚本通过 `DistributedConfig(tp_size=8, tp_plan="auto")` 加载**缩小的随机 checkpoint**。第一次运行时由 rank 0 创建 checkpoint；同一目录若已存在 checkpoint，则不会因更改 `--layers`/`--profile` 自动重建，应为新配置使用新目录。

先做官方宽度 4 层、单处 PLE、短序列：

```bash
HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=1 \
torchrun --standalone --nproc_per_node=8 \
  examples/qwen38/transformers_tp8_smoke.py \
  --profile width --layers 4 --bootstrap-checkpoint \
  --checkpoint runs/qwen38/tp8_width4_checkpoint \
  --dtype bfloat16 --attention eager --experts grouped_mm --gdn fla \
  --seq-len 128 --steps 1 --max-baseline-abs-error 0.5 \
  --output runs/qwen38/tp8_width4_seq128
```

再复用相同 checkpoint，增加序列与步数：

```bash
HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=1 \
torchrun --standalone --nproc_per_node=8 \
  examples/qwen38/transformers_tp8_smoke.py \
  --profile width --layers 4 \
  --checkpoint runs/qwen38/tp8_width4_checkpoint \
  --dtype bfloat16 --attention eager --experts grouped_mm --gdn fla \
  --seq-len 1024 --steps 2 --max-baseline-abs-error 0.5 \
  --output runs/qwen38/tp8_width4_seq1024_steps2
```

两次均已 PASS。第二次 loss 约 9.136 → 9.021，TP rank 之间 logits 最大差 0，峰值已分配显存约 4.05 GiB/卡。`result.json` 的 `tp_plan` 应包含 `model.layers.*.ple.ple_embedding.ngram_embedding: colwise_gather_output`，并且 `dtensor_parameter_count > 0`；这证明不是把 DDP 误称为 TP。该脚本还在 rank 0 与未切分模型比较输出：BF16 下最大绝对差约 0.439、平均绝对差约 0.018；0.5 是本次诊断阈值，不是模型精度验收标准。FP32 + `batched_mm` + GDN 参考路径的相同四层模型，TP 对单卡最大差约 `1.54e-5`。

进一步用 `--profile width --layers 8`、另一个 `--checkpoint runs/qwen38/tp8_width8_checkpoint`、`--seq-len 256 --steps 2` 已通过，仍只有官方第 2 层 PLE。相反，默认 `tiny` 的 8 层合成配置会造两处**不同大小** PLE 表，在 PyTorch DTensor 特征维分片反向中遇到 `EmbeddingBackward0` 梯度形状冲突；`ple_tp_probe.py` 可在 CPU/Gloo 两进程复现。它不是官方单处 PLE 的直接阻断项。调试两处 PLE 时才使用 `--ple-tp replicated`，不可将其当成全量 PLE 表的可扩展方案。

## 7. 流水线并行 PP：通用入口存在，当前 Qwen4Exp 尚未跑通

当前 Transformers 源码有 `DistributedConfig(pp_size=N)` 和通用的按层分段、相邻 rank 发送激活的实现；因此“Transformers 完全没有 PP”不准确。[Qwen4Exp 官方文档](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/qwen4_exp.md)写明没有提供模型 PP plan。实测现有缩小 Qwen4Exp checkpoint：

```bash
HIP_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 \
torchrun --standalone --nproc_per_node=2 \
  examples/qwen38/transformers_pp_probe.py \
  --checkpoint runs/qwen38/tp8_tiny_checkpoint_tp8 \
  --output runs/qwen38/pp2_tiny_probe
```

两个阶段均能加载各自的权重（rank 0 约 104.6 万、rank 1 约 88.9 万参数），但 rank 1 的前向在 `reverse_embedding` 处失败：`PipelineIdentityLayer` 没有 `weight`。原因是通用 PP 在第二阶段删除了 token embedding，同时把输入改为 `inputs_embeds`；Qwen4Exp 的 PLE 路径此时仍要原始 token IDs。错误详见 `runs/qwen38/pp2_tiny_probe/rank1.json`。

即便修复 token ID 传递，还需保持四路 Gated Residual 状态跨段连续，而通用 PP 当前会让每段运行 `hyper_connection_mixer`，语义不等同于完整模型；其 P2P 发送/接收也不是已经验证的 Qwen4Exp 训练反向管线。因此**当前版本不能把 `pp_size=2` 当成“已支持 Qwen3.8 PP 训练”**。需要模型专用分段前向、跨段状态/PLE ID 传输、正确的反向与数值对齐测试。Megatron 自定义参考模型也在 [README.md](README.md) 中明确限制 `PP=1`；两条实现都不能仅靠多给几张卡自动开 PP。

顺带更正：官方 Qwen4Exp 文档称 TP 与 FSDP 不能组合，但当前本地 `DistributedConfig`/`DistributedMixin` 源码会先应用 TP 再应用 FSDP；**本环境未验证这一组合**，不能简单断言它已经可用或必然不可用。当前本地源码明确拒绝的是 `FSDP>1` 与 `PP>1` 的组合。

## 8. 输出、常见报错和下一阶段

| 现象 | 先检查 |
|---|---|
| `ModuleNotFoundError: transformers.models.qwen4_exp` | `PYTHONPATH` 是否指向本地 `transformers_qwen38/src`，以及源码提交/版本 |
| `Offsets have to be int32` | `grouped_mm` 的 `offs` 必须为 int32；使用本目录 probe |
| `hipblaslt_groupedgemm: no valid algorithm found` | 保存 dtype、各组 M/K/N、offsets；此机器 FP32 测试失败而 BF16 对应形状通过 |
| `causal_conv1d ... falling back` | 仅说明退回慢速 PyTorch 路径；继续看最终 `PASS`/`FAIL` |
| `ChildFailedError` 或其他 rank 收到 SIGTERM | 它们通常是汇总/连带现象；先读最早的 `rank<N>_error.txt` 或 `rank<N>.json` |
| TP 提示输出维度不能整除 8 | 检查是否使用 TP 脚本的专用缩小配置；不要拿单卡 `tiny` 配置直接做 TP8 |
| TP BF16 logits 最大差超过默认 0.2 | 官方宽度测试显式使用 0.5 诊断阈值，同时看平均差与 FP32 对照；不要用放宽阈值代替精度评测 |
| PP 的 `PipelineIdentityLayer.weight` | 当前 Qwen4Exp 模型专用 PP 接口缺失，见第 7 节 |

各运行目录中的 `result.json`/`effective_config.json` 是最重要的复现证据。`runs/` 通常被 Git 忽略，迁移服务器时需另行保存结果。此文档记录的损失值来自随机权重与固定输入，不要求新机器逐位一致；应首先看配置、后端、有限数值、梯度、更新和每个 rank 的最终状态。

**下一阶段与当前结论分开：**官方 PLE 单表在 BF16 下约 **95.37 GiB**，TP8 仅其权重理想分摊约 **11.92 GiB/卡**；125B 主模型加 PLE 的 BF16 文本权重约 **328 GiB**。尚未下载/加载官方 checkpoint，也没有全量视觉、MTP、训练优化器状态和 262K 上下文验证。需要先准备真实权重并编写独立的**推理加载/显存探针**，再评估全量训练所需的 EP、PP、FSDP/优化器分片或 CPU 卸载；不要直接把这里的随机权重训练脚本当成全量部署脚本。

## 源码与延伸阅读

- [官方模型卡与配置](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
- [Transformers Qwen4Exp 文档](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/qwen4_exp.md)
- [本地 Transformers Qwen4Exp 配置](../../../transformers_qwen38/src/transformers/models/qwen4_exp/configuration_qwen4_exp.py)
- [本地通用 PP 实现](../../../transformers_qwen38/src/transformers/distributed/pipeline_parallel.py)
- [早期结构排错记录](TRANSFORMERS_DCU_TEST.md)和 [当前 Megatron 参考模型说明](README.md)：两者针对不同阶段和不同实现，早期失败结论已被后续 BF16 测试更新。
