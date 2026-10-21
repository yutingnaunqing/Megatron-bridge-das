# Qwen3.8-Flash-Next 在海光 BW1000 上的 Transformers 复现手册

整理日期：2026-10-08；单卡及单节点八卡记录截至 2026-09-23，多节点 DDP 记录于 2026-10-08。本文是本目录 **唯一需要保留的 Transformers 复现说明**，使用上游 Transformers 的 `Qwen4ExpForCausalLM` 文本模型，不依赖本仓库的 Megatron 自定义模型或 Bridge 子模块。按“代码与环境 → 单卡参考 → 单卡优化 → 八卡 DDP → 八卡 TP”顺序运行。所有已通过的训练测试使用**缩小配置、随机初始化权重和随机 token**；通过表示前向、loss、反向及参数更新能执行，不表示官方 125B+51B 权重已加载、模型能正确回答问题，或训练性能达标。

本文同时包含新服务器准备、已通过的命令、早期失败的修正结论、输出判定与可选诊断。命令均假定项目位于容器内 `/public/home/yugeng/work/Megatron-bridge-das`；路径不同则按实际挂载位置替换。

## 1. 先认清模型与并行方式

[官方模型卡](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)给出的主文本模型为 48 层、hidden size 2560，层型按 `GDN → GDN → GDN → QSA` 循环；PLE **只附加在第 2 层 GDN**，不会每四层重复。每层还含 MoE 和 Gated Residual。模型卡列出约 125B 主模型参数、51B N-gram embedding 和 4B MTP。当前脚本不覆盖视觉、MTP、真实权重、生成 cache、长上下文质量或真实数据训练。

GDN 执行因果卷积和 Gated Delta Rule；QSA 用 indexer 选择注意力块；PLE/N-gram 按历史 token 查表并注入特征；MoE 使用 routed experts 与 shared expert；Gated Residual 混合四路残差状态。单卡 `tiny` 仍保留这些模块的连接，只缩小矩阵、词表、专家和 N-gram 表。各脚本会把实际测试配置写到输出目录，不能仅凭 `tiny` 或 `width` 名称推断参数量。

| 配置 | 层数 | hidden size | 词表 | experts | PLE 层 |
|---|---:|---:|---:|---:|---|
| 官方 `config.official.json` | 48 | 2560 | 248320 | 512 | 第 2 层 |
| 单卡 `tiny` | 4 | 128 | 512 | 4 | 第 2 层 |
| 单卡 `width` | 4 | 2560 | 248320 | 16 | 第 2 层 |
| DDP `width --layers 8` | 8 | 2560 | 8192 | 16 | 第 2、6 层 |
| TP `width --layers 4` | 4 | 2560 | 8192 | 16 | 第 2 层 |

所有缩小配置都把官方每头 2000 万的 N-gram 表基数改小；`width` 保留官方主要 hidden/head 维度和 Top-10 MoE 路由，不等同于完整模型。

| 方式 | 本目录脚本 | 每张卡持有什么 | 已验证 |
|---|---|---|---|
| 单卡 | `transformers_smoke.py` | 缩小后的完整模型 | BF16/FP32 参考路径；BF16 `grouped_mm`/FLA |
| 8 卡 DDP | `transformers_ddp8_smoke.py` | 每卡一份完整缩小模型 | 前反向、梯度 All-Reduce、更新 |
| 多节点 DDP | `run_multinode_ddp.sh` 启动同一个 DDP 脚本 | 每卡一份完整缩小模型 | 3 节点 24 卡 tiny 参考路径已通过 |
| 8 卡 TP | `transformers_tp8_smoke.py` | 官方 TP 规则切分的部分参数，另有部分复制参数 | 单处 PLE、MoE、GDN、QSA 路径的前反向和更新 |
| 多节点 TP | `run_multinode_tp.sh` 启动同一个 TP 脚本 | 默认 2 节点 × 8 卡，TP 分片跨节点 | 缩小随机模型的跨节点加载、前反向和更新 |
| 2 卡 PP 探针 | `transformers_pp_probe.py` | 通用 PP 能分层加载 | 当前 Qwen4Exp **前向失败**，见第 7 节 |

DDP 不能让大模型“每卡只占八分之一”；TP 能切分部分矩阵，但既不能保证全部权重等分，也不自动实现流水线并行（PP）。

## 2. 新服务器代码与环境

### 2.1 获取两份源码

在宿主机准备一个能挂入容器的工作目录。新服务器尚无项目时：

```bash
cd /public/home/yugeng/work
git clone --branch feat/qwen38-flash-next-dcu \
  https://github.com/yutingnaunqing/Megatron-bridge-das.git
git clone https://github.com/huggingface/transformers.git transformers_qwen38
git -C transformers_qwen38 checkout 8e6083374735b8c95f13b1cf2f80cd9c7486c389
```

项目已经检出时，先确认其分支和工作区状态，再只补缺失的 Transformers 源码目录：

```bash
git -C /public/home/yugeng/work/Megatron-bridge-das status --short --branch
git -C /public/home/yugeng/work/Megatron-bridge-das rev-parse HEAD
git -C /public/home/yugeng/work/transformers_qwen38 rev-parse HEAD
```

最后一条若提示目录不存在，就执行上面的 Transformers `git clone` 和 `checkout`。若项目尚未处于 `feat/qwen38-flash-next-dcu`，先按现有工作区状态决定如何切换分支，不要覆盖本地修改。**本手册的 Transformers 脚本不需要初始化 Bridge 子模块**，但需要本仓库的 [`config.official.json`](config.official.json)。克隆本分支不会带来 `runs/` 中的随机 checkpoint、测试结果或官方完整权重。

### 2.2 容器、依赖和导入检查

已验证的基线为 8 × **BW1000 64G**、Python 3.12.7、PyTorch 2.10.0、HIP 6.3.26113、Transformers 5.18.0.dev0；Transformers 源码提交为 `8e6083374735b8c95f13b1cf2f80cd9c7486c389`。原测试容器 `yg_qwen3.8` 使用镜像 `10.16.1.152:5000/jenkins/model_test_env/mbridge:0.6.0-ubuntu22.04-dtk2604-py3.12-20260909-1017`；镜像地址可能因服务器而异。FLA 0.5.1 与 TileLang `0.1.12+das.opt1.dtk2604.torch2100.2609081633.g25d759` 用于加速路径。应使用适配 DCU 的 PyTorch/DTK，不能用普通 CUDA wheel 替换。容器自带的 Transformers 5.11.0 没有 `qwen4_exp` 模型类。

确保项目与 `transformers_qwen38` 都挂到容器内 `/public/home/yugeng/work`。例如在已创建的 `test_m09r2n07` 中运行：

```bash
# 宿主机：先确认该容器的工作目录挂载
docker inspect test_m09r2n07 --format '{{json .Mounts}}'
docker exec -it test_m09r2n07 bash
```

以下命令均在**容器内**执行：

```bash
cd /public/home/yugeng/work/Megatron-bridge-das
export PYTHONPATH="$(cd ../transformers_qwen38 && pwd)/src:${PYTHONPATH:-}"

python - <<'PY'
import importlib.metadata as metadata
import sys
import torch, transformers
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpForCausalLM
print('python:', sys.version.split()[0])
print('torch:', torch.__version__, 'HIP:', torch.version.hip)
print('transformers:', transformers.__version__, transformers.__file__)
print('DCU:', torch.cuda.device_count(), torch.cuda.get_device_name(0))
print('model:', Qwen4ExpForCausalLM.__name__)
for package in ('flash-linear-attention', 'tilelang'):
    try:
        print(package + ':', metadata.version(package))
    except metadata.PackageNotFoundError:
        print(package + ': not installed')
PY

git -C ../transformers_qwen38 rev-parse HEAD
```

`transformers.__file__` 必须指向 `transformers_qwen38/src`，且 `torch.cuda.device_count()` 至少为计划使用的卡数；新 shell 必须重新设置 `PYTHONPATH`。HIP 构建的 PyTorch 仍使用 `device="cuda"` 和 `torch.cuda` 接口。单卡参考路径不依赖 FLA/TileLang；运行加速路径前再核对它们。不同服务器的随机 loss 不要求逐位一致。

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

同一参考后端的 FP32 单卡测试也曾通过，可在 BF16 路径异常时作为诊断：

```bash
HIP_VISIBLE_DEVICES=0 python -u examples/qwen38/transformers_smoke.py \
  --profile tiny --device cuda --dtype float32 --attention eager \
  --experts batched_mm --gdn torch_reference \
  --seq-len 32 --steps 1 \
  --output runs/qwen38/hf_tiny_fp32_reference
```

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

扩大到**官方宽度**，但仍只用 4 层、16 专家并缩小 N-gram 表；这个单卡脚本的 `width` 保留官方词表大小：

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

若多卡加速组合失败，可先用同一个 DDP 脚本的 `--profile tiny --layers 8 --experts batched_mm --gdn torch_reference --seq-len 64 --steps 2` 隔离通信与优化 kernel；为该诊断命令使用新的 `--output` 目录。它也是缩小随机权重测试，不能验证全量模型显存。

### 5.1 三节点 24 卡 DDP 启动脚本

[`run_multinode_ddp.sh`](run_multinode_ddp.sh) 从本目录的 [`hostfile`](hostfile) 逐行读取节点名；默认容器名是 `yg_qwen3.8_$(hostname)`，与 `/public/home/yugeng/work/basedate/create_containers_nodes_train.sh` 的命名一致。脚本在宿主机通过 SSH 登录各节点，再在各自容器内启动一个 `torchrun`。容器须使用 host 网络，且各节点都能看到相同的项目、Transformers 源码和 `/public/home/yugeng/work/basedate/qwen38_runs`。当前默认使用 `width`、4 层、BF16、`grouped_mm`、FLA GDN、每卡 1 个 token batch、序列 32、1 步；可用 `EXPERTS=batched_mm GDN=torch_reference` 选择参考路径。`WORLD_SIZE = 节点数 × GPUS_PER_NODE`；默认使用 hostfile 全部节点，设置 `NNODES` 可只用前 N 个节点，DDP 脚本允许大于 8 的 world size。

在**任意一台可 SSH 登录 hostfile 节点的宿主机**执行，先做启动前环境检查（会在共享的 `basedate/qwen38_runs/` 下创建输出目录）：

```bash
cd /public/home/yugeng/work/Megatron-bridge-das
bash examples/qwen38/run_multinode_ddp.sh examples/qwen38/hostfile --check
```

检查通过后启动测试：

```bash
bash examples/qwen38/run_multinode_ddp.sh examples/qwen38/hostfile
```

脚本打印实际输出目录。`logs/node<N>_<hostname>.log` 保存各节点的 `torchrun` 输出，`results/result.json` 由全局 rank 0 写出；应看到 `PASS: 24-rank DDP forward/loss/backward/all-reduce/update`，且 JSON 中 `distributed.world_size` 为 24。节点数或每节点卡数变化时，成功提示中的数字也会变化。若失败，先查看各节点日志及 `results/rank<N>_error.txt`。默认 rendezvous 使用 hostfile 首节点的 IPv4 地址和端口 `29538`；可设置 `MASTER_ADDR`、`MASTER_PORT`。可通过 `GPUS_PER_NODE`、`PROFILE`、`LAYERS`、`SEQ_LEN`、`STEPS`、`EXPERTS`、`GDN`、`OUTPUT_DIR` 和 `RUN_TIMEOUT` 覆盖默认值，例如：

```bash
PROFILE=width LAYERS=8 SEQ_LEN=128 STEPS=2 \
EXPERTS=grouped_mm GDN=fla \
bash examples/qwen38/run_multinode_ddp.sh examples/qwen38/hostfile
```

2026-10-08 在 `m09r2n05/m09r2n06/m09r2n07` 的 24 张 BW1000 上，默认 tiny/BF16 参考路径 **PASS**：`world_size=24`、全局 batch 24、参数 1,749,752/卡、一步平均 loss 6.2449、峰值已分配显存约 0.087 GiB/卡。记录位于 `/public/home/yugeng/work/basedate/qwen38_runs/multinode_ddp_3nodes_24ranks_20261008_132029_3579157/results/result.json`，三个节点日志在同级 `logs/`。该目录在仓库之外，不会随 Git 迁移。

扩大配置前先确认各节点没有其他任务占用 DCU。此脚本验证的是多节点**数据并行**；它不会把一个模型切分到 24 张卡。每次运行使用独立输出目录；`--check` 会在发起分布式任务前检查指定 Transformers 源码能否在每个容器里导入。曾发现 `m09r2n07` 容器自带的 `huggingface-hub 1.30.0` 和 `tokenizers 0.22.2` 不满足所检出的 Transformers 源码；该容器已对齐为另外两台节点使用的 `huggingface-hub 1.33.0`、`tokenizers 0.23.2`。若重新创建容器，需重新检查这两个包的版本；容器内可用 `python -m pip install --no-deps 'huggingface-hub==1.33.0' 'tokenizers==0.23.2'` 对齐。

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

进一步的 8 层官方宽度配置也已通过，仍只有官方第 2 层 PLE。新配置必须使用单独 checkpoint 目录：

```bash
HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=1 \
torchrun --standalone --nproc_per_node=8 \
  examples/qwen38/transformers_tp8_smoke.py \
  --profile width --layers 8 --bootstrap-checkpoint \
  --checkpoint runs/qwen38/tp8_width8_checkpoint \
  --dtype bfloat16 --attention eager --experts grouped_mm --gdn fla \
  --seq-len 256 --steps 2 --max-baseline-abs-error 0.5 \
  --output runs/qwen38/tp8_width8_seq256_steps2
```

相反，默认 `tiny` 的 8 层合成配置会造两处**不同大小** PLE 表，在 PyTorch DTensor 特征维分片反向中遇到 `EmbeddingBackward0` 梯度形状冲突。此问题也可在 CPU/Gloo 两进程独立复现，因此不是 DCU 专用 kernel 错误：

```bash
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=2 \
  examples/qwen38/ple_tp_probe.py --device cpu
```

它不是官方单处 PLE 的直接阻断项。调试两处 PLE 时才使用 `--ple-tp replicated`，它仍切分其他受 TP 规则覆盖的参数，但会在每卡复制完整 PLE 表，不能将其当成全量 PLE 表的可扩展方案。`--skip-baseline` 仅跳过额外的单卡数值对照，不会减少 TP 训练梯度和优化器状态显存。

### 6.1 跨节点 TP 启动脚本

[`run_multinode_tp.sh`](run_multinode_tp.sh) 默认读取 hostfile 的前两个节点，各启动 8 个进程，构成一个跨节点的 16 rank TP 组。可用 `NNODES` 和 `GPUS_PER_NODE` 自行设置节点数与每节点卡数，`TP_SIZE = NNODES × GPUS_PER_NODE`，启动器会把该值传给测试程序并校验结果。脚本通过 SSH 和 host 网络容器启动 `transformers_tp8_smoke.py`，先由全局 rank 0 创建本次运行专用的随机 checkpoint，然后执行数值对照、TP 前向、反向和参数更新。启动器不再限制 TP_SIZE=8，但模型 TP 规则仍要求受分片的维度可被 TP_SIZE 整除；当前 `width` 的 hidden size 2560、专家数 16，不能直接做 TP24。

```bash
cd /public/home/yugeng/work/Megatron-bridge-das
bash examples/qwen38/run_multinode_tp.sh examples/qwen38/hostfile --check
bash examples/qwen38/run_multinode_tp.sh examples/qwen38/hostfile
```

默认参数是 `width`、4 层、BF16、`eager` 注意力、`grouped_mm`、FLA、序列 32、逻辑 batch 1 和 1 步；`width` 的 BF16 对照阈值默认 0.5。可用环境变量调整，例如指定两节点各 4 卡的 TP8：

```bash
NNODES=2 GPUS_PER_NODE=4 SEQ_LEN=128 \
bash examples/qwen38/run_multinode_tp.sh examples/qwen38/hostfile
```

扩大模型时，先保持序列 32、batch 1、一步，逐项提高 `LAYERS`（4 的倍数）、`NUM_EXPERTS`（至少 10 且能被 TP_SIZE 整除）、`VOCAB_SIZE` 和 `NGRAM_VOCAB_SIZE_BASE`。后三项只对 `PROFILE=width` 生效，默认分别为 16、8192、20000；隐藏宽度及注意力维度维持官方配置。以下示例在 TP16 上增加到 8 层、32 专家，并扩大词表和 PLE 表：

```bash
NNODES=2 GPUS_PER_NODE=8 PROFILE=width LAYERS=8 \
NUM_EXPERTS=32 VOCAB_SIZE=16384 NGRAM_VOCAB_SIZE_BASE=50000 \
SEQ_LEN=32 STEPS=1 SKIP_BASELINE=1 \
bash examples/qwen38/run_multinode_tp.sh examples/qwen38/hostfile
```

`SKIP_BASELINE=1` 跳过额外的单卡未切分模型前向，适合单卡无法容纳完整参考模型的规模；此时 `max_tp_vs_unsharded_logit_abs_error` 为 `null`，仍会检查跨 TP rank 的 logits、一轮训练的 loss/梯度/参数以及实际分片。规模增加后，再单独提高 `SEQ_LEN`、`LOCAL_BATCH_SIZE` 和 `STEPS`。每次启动器使用新的随机 checkpoint 和结果目录；手动复用 checkpoint 时，新脚本会核对层数、层类型、PLE 层、词表、专家数等配置，不匹配则报错。

第二步验证并行方案时，以相同模型规模分别运行 TP8 和 TP16，比较 `result.json` 中的 `model.logical_parameter_numel`、`sharding.local_parameter_numel_by_rank`、`sharding.logical_sharded_parameter_numel`、`sharding.parameter_replication_factor`、`sharding.per_rank`、`checks.max_observed_peak_allocated_gib_per_rank` 和 `max_step_seconds`。逻辑参数量应保持一致；实际每卡参数量和复制因子反映当前 TP plan 的切分效果。`sharding.per_rank` 另记录参数、梯度、AdamW 状态的实际本地字节数，以及模型加载前后和执行阶段的峰值已分配显存。`max_step_seconds` 是每步最慢 rank 的耗时，包含同步，短跑结果只适合发现明显瓶颈；优化器状态是当前已触发梯度的专家的观测值，后续可能增长。此脚本只验证纯 TP，不验证 TP×DP、EP 或 PP 的组合。

运行目录默认位于 `basedate/qwen38_runs/multinode_tp_...`，保存 `launch.txt`、每节点 `logs/`、随机 `checkpoint/` 和 `results/result.json`。检查 `status=PASS`、`parallelism.tp_size=NNODES×GPUS_PER_NODE`、`model.dtensor_parameter_count>0`；如失败，先查节点日志和 `results/rank<N>_error.txt`。`--check` 会创建输出目录，只做环境检查，不验证模型维度是否适合所选 TP_SIZE。容器需以 `yg_qwen3.8_$(hostname)` 命名、使用 host 网络、挂载同一 `/public/home/yugeng/work`，且宿主机之间可免密 SSH。本测试仅验证缩小随机模型，不代表官方全量权重可运行。

2026-10-08 在 `m09r2n05/m09r2n06` 的各 4 张 BW1000 上，默认 tiny/BF16 配置 **PASS**：`world_size=tp_size=8`、`dtensor_parameter_count=47`、跨 rank logits 最大差 0、TP 对未切分模型 logits 最大差约 0.0106、一步 loss 约 6.2584。结果位于 `/public/home/yugeng/work/basedate/qwen38_runs/multinode_tp_2nodes_8ranks_20261008_141309_3892387/results/result.json`，日志在同级 `logs/`。

同日，在 `m09r2n05/m09r2n06` 各 8 张 BW1000 上，`width`、4 层、BF16、序列 32 的 **TP16 PASS**：`world_size=tp_size=16`、`dtensor_parameter_count=47`、跨 rank logits 最大差 0、TP 对未切分模型 logits 最大差约 0.1621、一步 loss 约 9.1244。结果位于 `/public/home/yugeng/work/basedate/qwen38_runs/multinode_tp_2nodes_16ranks_20261008_153744_162381/results/result.json`。

## 7. 流水线并行 PP：通用入口存在，当前 Qwen4Exp 尚未跑通

当前 Transformers 源码有 `DistributedConfig(pp_size=N)` 和通用的按层分段、相邻 rank 发送激活的实现。[Qwen4Exp 官方文档](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/qwen4_exp.md)写明没有提供模型 PP plan。以下是**预期失败的可选诊断**，不属于通过标准。先用 TP 脚本生成它需要的四层 tiny 随机 checkpoint；若该目录已用不同配置生成，换一个新目录并同步修改 PP 命令：

```bash
HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=1 \
torchrun --standalone --nproc_per_node=8 \
  examples/qwen38/transformers_tp8_smoke.py \
  --profile tiny --layers 4 --bootstrap-checkpoint \
  --checkpoint runs/qwen38/tp8_tiny_checkpoint_tp8 \
  --dtype bfloat16 --attention eager --experts grouped_mm --gdn fla \
  --seq-len 32 --steps 1 \
  --output runs/qwen38/tp8_tiny_bf16
```

然后运行 PP 探针：

```bash
HIP_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 \
torchrun --standalone --nproc_per_node=2 \
  examples/qwen38/transformers_pp_probe.py \
  --checkpoint runs/qwen38/tp8_tiny_checkpoint_tp8 \
  --output runs/qwen38/pp2_tiny_probe
```

两个阶段均能加载各自的权重（rank 0 约 104.6 万、rank 1 约 88.9 万参数），但 rank 1 的前向在 `reverse_embedding` 处失败：`PipelineIdentityLayer` 没有 `weight`。原因是通用 PP 在第二阶段删除了 token embedding，同时把输入改为 `inputs_embeds`；Qwen4Exp 的 PLE 路径此时仍要原始 token IDs。错误详见 `runs/qwen38/pp2_tiny_probe/rank1.json`。

即便修复 token ID 传递，还需保持四路 Gated Residual 状态跨段连续，而通用 PP 当前会让每段运行 `hyper_connection_mixer`，语义不等同于完整模型；其 P2P 发送/接收也不是已经验证的 Qwen4Exp 训练反向管线。因此**当前版本不能把 `pp_size=2` 当成“已支持 Qwen3.8 PP 训练”**。需要模型专用分段前向、跨段状态/PLE ID 传输、正确的反向与数值对齐测试。

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

各运行目录中的 `result.json`/`effective_config.json` 是最重要的复现证据；单卡失败时另看 `error.txt`，多卡失败时优先看最早的 `rank<N>_error.txt`、`rank<N>.json`。保存运行命令、容器镜像标签、`torch`/HIP/Transformers 版本和两份源码提交。`runs/` 被 Git 忽略，迁移服务器时需另行保存结果。此文档记录的损失值来自随机权重与固定输入，不要求新机器逐位一致；应首先看配置、后端、有限数值、梯度、更新和每个 rank 的最终状态。

**下一阶段与当前结论分开：**官方 PLE 单表在 BF16 下约 **95.37 GiB**，TP8 仅其权重理想分摊约 **11.92 GiB/卡**；125B 主模型加 PLE 的 BF16 文本权重约 **328 GiB**。尚未下载/加载官方 checkpoint，也没有全量视觉、MTP、训练优化器状态和 262K 上下文验证。需要先准备真实权重并编写独立的**推理加载/显存探针**，再评估全量训练所需的 EP、PP、FSDP/优化器分片或 CPU 卸载；不要直接把这里的随机权重训练脚本当成全量部署脚本。

## 9. 保留文件与来源

独立复现 Transformers 路径需要本文件、[`config.official.json`](config.official.json)、[`transformers_smoke.py`](transformers_smoke.py)、[`transformers_ddp8_smoke.py`](transformers_ddp8_smoke.py) 和 [`transformers_tp8_smoke.py`](transformers_tp8_smoke.py)。多节点 DDP 使用 [`run_multinode_ddp.sh`](run_multinode_ddp.sh)，多节点 TP 使用 [`run_multinode_tp.sh`](run_multinode_tp.sh)，两者都使用本地 [`hostfile`](hostfile)。[`grouped_mm_probe.py`](grouped_mm_probe.py)、[`ple_tp_probe.py`](ple_tp_probe.py) 用于定位算子和 DTensor 问题；[`transformers_pp_probe.py`](transformers_pp_probe.py) 仅用于记录当前 PP 失败。以上脚本均直接使用 PyTorch/Transformers，不导入本仓库的 Megatron 自定义模型。`config.official.json` 是三个主测试脚本的默认输入，不能随其他 Bridge 文件一起删掉。

- [官方模型卡与配置](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
- [Transformers Qwen4Exp 文档](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/qwen4_exp.md)
- [已验证的 Transformers 源码提交](https://github.com/huggingface/transformers/commit/8e6083374735b8c95f13b1cf2f80cd9c7486c389)
