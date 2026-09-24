# Qwen3.8-Flash-Next：BW1000 文本训练参考实现

直接使用官方 Transformers 在 DCU 上复现单卡、八卡 DDP/TP 测试及 PP 现状，请从 [DCU_TRANSFORMERS_REPRODUCE.md](DCU_TRANSFORMERS_REPRODUCE.md) 开始。本文件前半部分的 Megatron 参考实现属于另一条路径。
迁移到另一台服务器的拉取和依赖准备步骤见 [REPRODUCE_ON_NEW_SERVER.md](REPRODUCE_ON_NEW_SERVER.md)。

该示例把 Qwen3.8 的文本结构接入 **Megatron-Bridge 的 pretrain、MCore DDP 和分布式优化器**。
用于验证结构和提取算子需求，随机初始化并使用 mock token。当前不是完整模型权重部署，也不是融合 QSA 性能方案。

## 实现范围

| 模块 | 本示例 |
|---|---|
| GDN | 官方 Torch chunk/recurrent 计算、卷积、sigmoid output gate |
| QSA | 压缩 indexer、块路由、尾部 token、dense-mask GQA、Stage-2 KL 梯度 |
| Gated Residual | 官方四路分组归一化与读写门 |
| N-gram/PLE | 官方 hash、EOS 边界、查表、门控注入、膨胀卷积；表留在设备上 |
| MoE | 官方路由与专家计算，添加可求导负载均衡损失 |
| 并行 | 单节点 DP；TP/PP/EP/ETP/CP 必须为 1 |
| 优化器 | Bridge 分布式 Adam；没有复现官方 Muon 参数分组 |
| 数据/权重 | mock、随机初始化；无 HF 导入导出 |
| 视觉、MTP、KV cache、packing | 当前未实现 |

**八卡使用 DP，不会把模型参数分成八份。**分布式优化器分摊优化器状态，模型参数和梯度仍需按本实现的内存布局分配。

QSA 在本实现中创建 `[B,H,S,S]` 稠密注意力张量，硬性限制 `S <= 4096`。
这实现了稀疏路由的数学语义，但不具备稀疏 kernel 的显存/吞吐收益。
TopK 分数相同时，训练路径按 block ID 升序选择，保障重计算一致性；这可能与上游 `torch.topk` 的任意同分选择不同。

## 容器与路径

本机环境：`yg_mbridge`，8 × BW1000 64GB，PyTorch 2.10.0 / HIP 6.3.26113 / Transformers 5.11.0。
在容器内进入项目根目录，执行后续命令：

```bash
docker exec -it yg_mbridge bash
cd /public/home/yugeng/work/Megatron-bridge-das
```

脚本自动设置项目、Bridge、内嵌 Core 和本地 Energon 的 Python 路径。
不会切换 Core 提交、复制上游 recipe 文件、升级 Transformers 或安装包。
容器未提供新模型 HF 类，所以文本 building blocks 独立保存在适配目录；没有把模型伪装成 Qwen3.5。

在新环境仍需提供兼容的 PyTorch、Transformer Engine、Megatron-Bridge 依赖和 Energon。
本地 `3rdparty/Megatron-Energon` 属于已有但未被主仓库跟踪的目录；迁移时需要额外带上它或安装匹配版本。


## 直接 Transformers 的八卡张量并行测试

`transformers_ddp8_smoke.py` 是 DDP：每个 rank 有完整模型。`transformers_tp8_smoke.py` 则不使用 DDP，采用 Transformers 的原生 `DistributedConfig(tp_size=8, tp_plan="auto")`。它会在加载 checkpoint 前按 Qwen4Exp 官方 TP plan 创建 `DeviceMesh/DTensor`，并切分 MoE 专家、GDN 投影、QSA indexer、PLE embedding 和 `lm_head`。

先运行下面的合成权重测试。`--bootstrap-checkpoint` 仅在 checkpoint 不存在时由 rank 0 创建一个四层随机权重 checkpoint；其目的在于验证 DCU 上的 TP 加载、通信、前反向与更新，不能说明真实权重的模型效果。

```bash
cd /public/home/yugeng/work/Megatron-bridge-das
export PYTHONPATH=/public/home/yugeng/work/transformers_qwen38/src:${PYTHONPATH}

HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=1 \
torchrun --standalone --nproc_per_node=8 \
  examples/qwen38/transformers_tp8_smoke.py \
  --bootstrap-checkpoint \
  --checkpoint runs/qwen38/tp8_tiny_checkpoint_tp8 \
  --dtype bfloat16 --attention eager --experts grouped_mm --gdn fla \
  --seq-len 32 --steps 1 \
  --output runs/qwen38/tp8_tiny_bf16
```

成功时输出 `PASS: 8-rank native Transformers TP forward/loss/backward/update`，并写入 `result.json`。其中 `dtensor_parameter_count > 0` 证明实际有参数被 TP 分片；`max_cross_rank_logit_abs_error` 应为 0 或非常接近 0；默认还会输出 TP 和单卡未切分结果的最大 logits 差异。

本脚本只面向缩小随机 checkpoint 的训练兼容性测试。`--skip-baseline` 仅跳过额外的单卡数值基准，**不会**减少训练梯度与优化器显存；不要直接用它加载全量官方 checkpoint 做训练。真实权重需要另行准备推理加载与显存探针，并核对 checkpoint 是文本 `Qwen4ExpForCausalLM` 格式还是包含视觉前缀的条件生成格式。

### 官方宽度、单处 PLE 的八卡 TP 验证（2026-09-23）

官方 `config.official.json` 的 `ple_layer_ids=[2]`，所以全量文本模型只有第 2 层一张 PLE N-gram 表。当前 Transformers 配置给它的 TP 规则是 `colwise_gather_output`；本地实现对 `nn.Embedding` 的权重第 1 维（特征维）切分，checkpoint 的 `split_ngram_parts=128` 则是按第 0 维（词表维）分文件，两者不能混为一谈。官方文档目前称 PLE “vocabulary-row sharding”，与本地源码实际行为不一致；下述容量按本地源码的特征维切分计算。

从官方配置和源码中的素数扩表规则可算出单张表形状为 `[320001536, 160]`，共 51,200,245,760 个参数。BF16 权重需 102.40 GB（95.37 GiB），TP8 理想均分为每卡 11.92 GiB；这只是 PLE 权重本身。若训练时采用 BF16 权重、BF16 梯度、FP32 AdamW 一阶和二阶状态，约为 12 字节/参数，TP8 的 **PLE 一项** 就约 71.53 GiB/卡。若额外保留 FP32 master 权重，约 95.37 GiB/卡。实际优化器和卸载策略会改变数字。

为验证官方宽度的算子路径，`--profile width` 保留 hidden size 2560、QSA/GDN head 配置及单处 PLE，同时缩为 4 层、16 专家、8192 词表、每头 20000 N-gram 行；checkpoint 为随机权重。以下命令已在 8 张 BW1000 64G 上通过：

```bash
HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=1 \
torchrun --standalone --nproc_per_node=8 \
  examples/qwen38/transformers_tp8_smoke.py \
  --profile width --bootstrap-checkpoint \
  --checkpoint runs/qwen38/tp8_width4_checkpoint \
  --dtype bfloat16 --attention eager --experts grouped_mm --gdn fla \
  --seq-len 1024 --steps 2 --max-baseline-abs-error 0.5 \
  --output runs/qwen38/tp8_width4_seq1024_steps2
```

`result.json` 显示 PLE 官方分片规则存在、跨 rank logits 最大差为 0、loss 为 9.135968 → 9.021369、峰值已分配显存约 4.05 GiB/卡。TP 对比单卡 BF16 参考的最大绝对差为 0.4395，平均绝对差 0.0182，参考 logits 的 RMS 为 0.4970；所以此宽度测试显式使用 0.5 的最大差阈值。序列 128、1 步也通过，峰值约 1.78 GiB/卡。以上仅验证缩小模型的训练路径，不能外推全量显存或模型效果。

另用 `--profile width --layers 8 --seq-len 256 --steps 2` 保持官方单处 PLE，八卡 TP 前向、反向和更新通过，loss 9.149139 → 8.773849；记录在 `runs/qwen38/tp8_width8_seq256_steps2/result.json`。为查明 BF16 数值差，再用同一个四层 checkpoint 测试参考路径：FP32 + `batched_mm` + `torch_reference` 的 TP 对单卡最大 logits 差为 `1.54e-5`；BF16 + 同样的参考路径为 `0.2769`。后者与 BF16 加速路径的 `0.3018` 接近，提示主要差异与 BF16 下 TP 计算次序和舍入有关，但这些 smoke test 不等价于任务精度评测，也不保证逐项数值相同。

按模型卡给出的约 125B 主模型 + 51B PLE 估算，文本 BF16 权重约 328 GiB；加约 4B MTP 则约 336 GiB，视觉编码器及运行缓存另计。8 张 64 GiB 卡在**理想均匀切分、短上下文推理**下有理论容量，16 张更有余量，但尚未在 DCU 上加载全量 checkpoint 验证。含 MTP 的全参数训练，其权重、梯度和 AdamW FP32 状态按 12–16 字节/参数约需 1.97–2.63 TiB（不含视觉与激活），8 卡不可能；64 卡可作为后续多维切分/卸载方案的容量规划起点，不代表现有 Transformers TP 脚本开箱支持 64 卡全量训练；官方 Transformers 文档写明未提供 PP plan；它也称 TP 与 FSDP 不能组合，但本地源码会尝试组合 TP+FSDP，此组合尚未在 DCU 上实测。长上下文激活、通信缓存和视觉编码器还会增加需求。

参考：[Qwen 官方模型卡](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)、[Transformers Qwen4Exp 文档](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/qwen4_exp.md)。

### 8 层、两处 PLE 的反向问题（2026-09-23）

默认 `--profile tiny --layers 8 --seq-len 256` 会生成两处 PLE；`--profile width` 始终只在官方第 2 层放一处 PLE。官方 TP plan 将两张 n-gram embedding 表按特征列切分；当前 PyTorch 2.10.0 / HIP 6.3.26113 在反向时报告 `EmbeddingBackward0` 梯度形状为 `[6400, 8]`，而第一张表预期 `[4864, 8]`。相同错误可由 `ple_tp_probe.py` 在 **CPU/Gloo 两进程**独立复现，因此这不是 DCU 专用 kernel 报错；两张表的大小来自 PLE 各层使用不同的 n-gram 素数词表。

```bash
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=2 \
  examples/qwen38/ple_tp_probe.py --device cpu
```

继续测试缩小模型的其他 TP 路径时，使用 `--ple-tp replicated`：它只从官方 TP plan 中去掉 PLE embedding 规则，MoE、GDN、QSA 投影等仍按 TP 切分。已在 8 张 BW1000 上验证 8 层、序列 256、3 步训练通过：

```bash
HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=1 \
torchrun --standalone --nproc_per_node=8 \
  examples/qwen38/transformers_tp8_smoke.py \
  --checkpoint runs/qwen38/tp8_8layers_checkpoint \
  --ple-tp replicated --dtype bfloat16 --attention eager \
  --experts grouped_mm --gdn fla --seq-len 256 --steps 3 \
  --output runs/qwen38/tp8_8layers_seq256_steps3_ple_replicated
```

`result.json` 记录 `ple_tp_mode: replicated`、3 步 loss、实际 TP plan 和显存。该诊断模式让每张卡保留完整 PLE 表，不能据此推断官方全量模型的 PLE 分片训练已经通过。官方配置只有一处 PLE；上面的单处 PLE 官方分片测试已通过，因此“两张表梯度冲突”并不是当前官方配置的直接阻断项。

## 三种配置

| 字段 | 官方配置 | `smoke`（默认） | `tiny` |
|---|---:|---:|---:|
| 层数 | 48 | 4 | 4 |
| hidden size | 2560 | 2560 | 128 |
| 词表 | 248320 | 248320 | 512 |
| Q/KV heads | 24/2 | 24/2 | 4/2 |
| QSA token budget | 2048 | 2048 | 16 |
| 专家数 / router topK | 512/10 | 16/10 | 4/2 |
| N-gram vocab base（每 head） | 20000000 | 20000 | 257 |
| MTP | 1 | 关闭 | 关闭 |

`smoke` 保留官方宽度、词表、主注意力头、GDN head、专家 FFN 维度，以及 QSA 超参数。
除减层外，还缩小专家数和 N-gram 表，所有变更写入输出配置。
官方 N-gram 表 BF16 权重本身约 95 GiB，不能放入一张 64 GiB 卡；仅减层不能解决这个问题。

`official` 使用官方宽度/专家/表大小，但层数仍由 `NUM_LAYERS` 控制，MTP 与视觉仍未实现。
它可用于 dryrun 检查；64 GiB 卡上的训练会因表大小在分配前明确报错，不会静默缩表。

## 运行

先解析配置：

```bash
bash examples/qwen38/train_qwen38_flash_next_pretrain.sh smoke dryrun
```

单卡微型结构测试：

```bash
GPUS=1 SEQ_LENGTH=32 \
bash examples/qwen38/train_qwen38_flash_next_pretrain.sh tiny train
```

八卡、四层、官方宽度的缩小模型：

```bash
GPUS=8 TRAIN_ITERS=3 SEQ_LENGTH=128 \
bash examples/qwen38/train_qwen38_flash_next_pretrain.sh smoke train
```

开启本实现的逐层重计算：

```bash
GPUS=8 SEQ_LENGTH=128 \
bash examples/qwen38/train_qwen38_flash_next_pretrain.sh smoke train --recompute
```

显式调整模型规模：

```bash
NUM_LAYERS=8 GPUS=8 \
bash examples/qwen38/train_qwen38_flash_next_pretrain.sh smoke dryrun \
  --num-experts 32 --ngram-vocab-size 40000
```

可覆盖环境变量：`GPUS`、`NUM_LAYERS`、`TRAIN_ITERS`、`SEQ_LENGTH`、`GLOBAL_BATCH_SIZE`、`OUTPUT_DIR`、`PYTHON`。
额外参数直接传给 `run_qwen38.py`；用 `--help` 查看。全局 batch 必须是 DP 卡数的整数倍。
首次不建议扩大规模，先核对本配置的实测显存。

默认输出位于 `runs/qwen38/<profile>_<timestamp>/`：

```text
text_config.json         实际文本结构
resolved.yaml            Bridge 训练配置
run_manifest.json        命令参数、版本与限制
train_logs/train.log     训练日志
tb_logs/                TensorBoard
result.rank<N>.json      各 rank 正常返回后的结果与峰值显存
```

默认不保存 checkpoint。正常结束后应看到 3 步日志，`number of nan iterations: 0`、`number of skipped iterations: 0`，并且每个 rank 产生结果文件。
日志中的 LM loss、QSA KL 和 router aux 分开记录；实际优化目标为 `LM + QSA_KL + router_aux_loss_coef * router_aux`。
stock Megatron 的理论 FLOPs/显存估计不适用于该自定义骨干，不能把其显示值作为模型性能结论。

## 正确性检查

容器内：

```bash
PYTHONPATH="$PWD/hcu_megatron_bridge/src" OMP_NUM_THREADS=1 \
python -m pytest tests/qwen38/test_reference.py -q
```

包括官方 QSA 前向/输入梯度对齐（同分策略对齐后）、尾部与短序列、indexer KL 梯度、GDN chunk/recurrent 对比、N-gram EOS 边界、整模型各模块梯度和重计算一致性。

`smoke` 默认 128 token，小于官方 QSA 预算，不能单独证明截断稀疏路由已被覆盖。
`tiny` 的 budget=16、测试长度=32，专门覆盖超预算路由。

## QSA 算子基线

QSA 的固定张量形状、精度要求、稀疏语义和验收办法见 [QSA_OPERATOR_REQUIREMENTS.md](QSA_OPERATOR_REQUIREMENTS.md)。
它附带独立于整模型训练的单卡基准：

```bash
bash examples/qwen38/benchmark_qsa.sh smoke \
  --sequence-lengths 128,2048 --warmup 2 --iterations 5 --profile-ops
```

短于等于 2048 token 时，官方 token budget 会保留所有因果历史；因此必须至少再执行一次 `S=4096`，
才能测到真正的块路由截断。该基准测的是当前稠密参考路径，只有替换成稀疏 kernel 后，才能作为 kernel
性能基线使用。

本机 BW1000 基线已完成：`S=128/2048` 的前向反向分别为 8.13/24.70 ms 每次，`S=4096` 的仅前向为
29.48 ms 每次；完整条件和不能直接比较的原因见 [QSA_OPERATOR_REQUIREMENTS.md](QSA_OPERATOR_REQUIREMENTS.md)。

## 本机实测结果（2026-09-15）

在上述 `yg_mbridge` 容器中完成了以下检查：

| 检查 | 配置 | 结果 |
|---|---|---|
| 正确性测试 | `tests/qwen38/test_reference.py` | 11 passed |
| 单卡训练 | tiny，序列 32，3 步 | 正常完成，无 NaN/跳步 |
| 八卡训练 | smoke，序列 128，3 步 | 正常完成，无 NaN/跳步 |
| 八卡逐层重计算 | 同上，增加 `--recompute` | 8 个 rank 正常返回，无 NaN/跳步 |

八卡 smoke 实际参数量为 **1,974,775,200（约 19.75 亿）**，global batch=8、micro batch=1。
普通训练和重计算训练打印的三步指标一致：

| 步数 | LM loss | QSA KL loss | 梯度范数 |
|---|---:|---:|---:|
| 1 | 12.54204 | 0.6720692 | 13.949 |
| 2 | 12.31348 | 0.6736218 | 14.213 |
| 3 | 12.13255 | 0.6584882 | 12.509 |

重计算训练各卡峰值张量分配约 **17.495–17.497 GiB**，缓存分配器峰值保留约 **20.24–20.54 GiB**；
这些是 PyTorch 记录值，不含所有驱动/通信开销。三步随机数据训练只证明该配置能完成前向、反向和参数更新，不代表收敛或性能基准。

本机证据（位于被忽略的 `runs/` 下，不随 Git 自动迁移）：

- 普通八卡日志：`runs/qwen38/smoke_8gpu/train_logs/train.log`
- 重计算八卡日志：`runs/qwen38/smoke_8gpu_recompute/train_logs/train.log`
- 重计算各卡结果：`runs/qwen38/smoke_8gpu_recompute/result.rank0.json` 至 `result.rank7.json`
- 同目录的 `text_config.json`、`resolved.yaml` 和 `run_manifest.json` 记录本次实际配置。

对“Megatron 能否开箱运行”的结论：当前仓库/容器需要新增模型接入和参考计算路径，已有环境不能直接加载该官方模型进行完整训练；
本示例证明新增适配后，缩小的文本结构可以通过 Bridge 在 BW1000 上训练。

## 下一步算子与工程需求

1. 在 BW1000 上实现 indexer pooling/归一化/RoPE/score 和确定性 TopK。
2. 实现真正的 Sparse GQA 前后向，避免 dense mask 和完整分数矩阵。
3. 将 Stage-2 KL/teacher 聚合融合进稀疏前后向；进一步核对真实训练的损失权重。
4. 验证 GDN 的 FLA/HIP 优化替代路径，保持前后向数值一致。
5. 完成专家并行、宽残差 PP 传输和 N-gram 表分片/卸载；它们是完整模型资源扩展的必要工程。
6. 添加完整 HF 权重映射、保存恢复、真实数据与 tokenizer；然后才评估完整模型和长序列性能。

## 来源

- [官方模型与配置](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
- [Transformers 参考源码](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py)
- [Qwen 技术报告](https://arxiv.org/html/2608.30320v1)

`reference.py` 保留 Qwen/HuggingFace 版权，附 Apache-2.0 LICENSE；文件头记录下载日、原始文件 SHA256 和本地修改。
固定源码快照可以在当前 Transformers 版本上执行，不依赖在线下载模型代码。
