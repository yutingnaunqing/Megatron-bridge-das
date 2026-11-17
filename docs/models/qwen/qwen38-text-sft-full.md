# Qwen3.8-27B 全64层纯语言全参 SFT

## 当前范围

最新入口统一为`run_qwen38_text_sft.sh`，默认native；旧多层包装及共用MPI/runtime已删除。下文保留历史验收说明，最新操作与默认值以[handoff](qwen38-text-sft-handoff.md)为准。sft不再自动预检或生成恢复manifest，显式preflight仅为手工工具。

2026-11-16：旧runs测试checkpoint与参数比较副本已按用户要求清理，原始模型和审计结论保留；以下恢复步骤适用于后续新实验产生的checkpoint。当前启动与结论以[handoff](qwen38-text-sft-handoff.md)为准。

使用本地官方权重的全部64个语言层，以及 embedding、final norm、LM head；约26.896B参数全部参与训练，非LoRA。不加载、不构造视觉塔；默认关闭MTP，显式模式支持原始MTP1。新入口 `examples/qwen/run_qwen38_text_sft.sh` 默认64层，减层实验用EXPECTED_NUM_LAYERS和LOCAL_HF_PATH显式配置；无需单独的单节点入口。

2026-11-14最新要求为IB报错换节点、禁止Socket完成验收。n05+n07已完成完整64层IB不中断8步训练、验证、step5/8保存及第8步checkpoint审计；16rank日志确认外置SHCA IB插件。恢复曾因n07新增显存占用OOM；用户释放节点后，14:39仅补跑同节点IB恢复，14:47通过：第6–8步日志及最终验证loss一致、545个抽样语言参数tensor逐元素相同，optimizer step=8/scheduler num_steps=256，未重新导入HF。全部27B参数未逐位比较。此前n05+n10的IB/Socket混合对照仅保留历史数值证据。详见handoff文末。

2026-11-15 02:20：完整64层MTP1的n05+n07、TP4/PP2/DP2四阶段IB短测已通过，匹配无MTP基线、初训保存、第5步恢复到8步及连续8步对照通过。全部424,699,392个MTP参数及545个抽样语言tensor逐元素一致，最终validation loss均0.8456035；设备采样峰值57,930MiB。原TP4/PP4/DP1非对等embedding配对仍未通过，不沿用其结论。详见[handoff最终验收结果](qwen38-text-sft-handoff.md)与[MTP方案](qwen38-text-sft-mtp-plan.md)。下文原首测默认参数保留，启用MTP请显式选取已验收的TP4/PP2/DP2配置。

## 显存评估与首测配置

节点每卡实际容量65520MiB，即63.984GiB。当前配置为BF16权重、FP32梯度/master参数/Adam双矩、distributed optimizer、CP1、无PEFT。依据固定版本MCore `docs/user-guide/features/dist_optimizer.md`，每个本地参数的常驻状态估算为 `6 + 12/DP` 字节；DP2从18降至12字节，不是减半。

工具 `tools/checkpoint/estimate_qwen38_text_sft_memory.py` 从真实tensor形状计数，按均匀PP划分，计入首尾embedding/head和TP复制norm权重；未计激活、通信、padding、临时分配、allocator reserve和checkpoint staging。结果见 `runs/qwen38_text_sft_full_20261113/memory_estimate.json`。

| 设备与拓扑 | DP | 最高单卡常驻状态 | 距63.984GiB余量 | 判断 |
| --- | --- | --- | --- | --- |
| 单节点8卡，TP2/PP4 | 1 | 61.69GiB | 2.29GiB | 余量过小，不作默认 |
| 单节点8卡，TP4/PP2 | 1 | 56.36GiB | 7.62GiB | 有尝试空间，未证明能跑通 |
| 双节点16卡，TP2/PP4 | 2 | 41.13GiB | 22.86GiB | 默认首测方案 |
| 双节点16卡，TP4/PP4 | 1 | 30.85GiB | 33.14GiB | 额外显存余量方案，吞吐待测 |

双节点默认SEQ1024、MBS1、GBS32、TP2/PP4/DP2；单节点默认SEQ1024、MBS1、GBS8、TP4/PP2/DP1。需要每节点完整8卡可用；若只空闲4卡，不可沿用这些估算。GPU总显存够不意味着每个PP stage都能容纳。

**首测开启 full/uniform/1 重计算**，关闭CUDA graph与激活CPU offload。该模型48层是GDN、16层是full attention，仅重计算 `core_attn` 覆盖不足。重计算降低激活占用，不减少上述参数/梯度/Adam常驻状态；增加计算量。双节点稳定后再比较关闭/选择性重计算的实测峰值和吞吐，单节点不建议直接关闭。

单节点OOM时先降低SEQ（如512并重新prepare到独立数据目录），不要只降低GBS期待参数显存下降。可实验 `OPTIMIZER_OFFLOAD_FRACTION=0.5` 将部分optimizer state转至CPU；默认0，未做该路径运行验收，需要足够宿主内存并衡量传输开销。这是optimizer offload，与PP>1下关闭的activation CPU offload不同。

必须检查至少两次optimizer step后的峰值（Adam可能懒分配），并完成保存；SEQ、样本长度、GBS和拓扑变化均需重新实测。已有16层峰值不能按层数简单乘4。

## 准备与配置检查

训练相关命令在正确挂载此工作树的既有 `yg_qwen38` 容器中执行。项目脚本沿用 `/opt/dtk/env.sh` 和 `env/env.sh`，无需重装环境。本轮已验证n05+n10的共享挂载与双节点训练；后续启动前仍需核对即时显存占用与网络。

```bash
cd /public/home/yugeng/work/Megatron-bridge-das-qwen38-sft-review
export SAVE_DIR="$PWD/outputs/qwen38-text-64layers-first"

# 权重已在本机完成，无需再执行convert，目标目录存在时工具会拒绝覆盖。
# 新机器首次导出：bash examples/qwen/run_qwen38_text_sft.sh convert
# 已有目录：/public/home/yugeng/work/models/Qwen3.8-27B-text-64layers

# 默认准备1024/64条短样本；已有相同配置的准备产物可直接复用，不要覆盖。
# bash examples/qwen/run_qwen38_text_sft.sh prepare

# 标准库元数据检查；不分配模型、不进行分布式通信。
bash examples/qwen/run_qwen38_text_sft.sh preflight
# 训练环境中额外检查真实tokenizer/collator，并解析输出recipe配置。
bash examples/qwen/run_qwen38_text_sft.sh dryrun
```

默认源 `/public/home/yugeng/work/datasets/HuggingfaceH4_ultrachat_200k`；已准备数据 `/public/home/yugeng/work/datasets/HuggingfaceH4_ultrachat_200k-qwen38-sft-seq1024-disabled`。tokenizer和chat template未因减层而改变，因此16层准备产物可复用；转换代码保留相同tokenizer文件。更换SEQ/thinking策略后应重新prepare，不能只改训练变量。

全量准备示例（读全部官方SFT分片，输出独立目录）：

```bash
TRAIN_SAMPLES=0 VALIDATION_SAMPLES=0 \
DATA_PATH=/public/home/yugeng/work/datasets/ultrachat-qwen38-full-seq1024-disabled \
  bash examples/qwen/run_qwen38_text_sft.sh prepare
```

转换后HF权重53,791,996,928字节、851个tensor，排除333个视觉及15个MTP tensor，manifest记录保留tensor的SHA256。源模型不变。checkpoint还包含FP32参数和Adam状态，完整模型的单份权重+优化器理论量级约376.5GB（350.7GiB），实际受格式、padding及额外状态影响；按每份checkpoint至少400GB再加临时空间预留，另需约53.8GB HF目录空间，注意SAVE_INTERVAL会增加总占用。

## 双节点启动

在n05的现有`yg_qwen38`容器执行一次，默认hostfile为`examples/qwen/hostfile`（n05、n07），无需手动启动另一节点。容器SSH端口默认11463，MPI每节点仅启动一个torchrun父进程；共享代码、模型、数据和SAVE_DIR。

```bash
unset NODE_RANK MASTER_ADDR
TP=4 PP=2 GLOBAL_BATCH_SIZE=32 \
HOSTFILE="$PWD/examples/qwen/hostfile" \
SAVE_DIR="$PWD/outputs/qwen38-text-64layers-first" \
EXIT_INTERVAL=5 TRAIN_ITERS=50 SAVE_INTERVAL=5 \
  bash examples/qwen/run_qwen38_text_sft.sh sft
```

保持TRAIN_ITERS50、只用EXIT_INTERVAL5提前退出，可以保留50步学习率调度。显式NODE_RANK仍支持手工逐节点方式，默认不用设置。NNODES可覆盖hostfile节点数，MASTER_ADDR默认取首行；MPI_SSH_PORT可覆盖11463。真实n05+n07启动参数探针通过，未重新启动GPU训练。长序列与VL的独立指令见[操作说明](qwen38-sft-long-sequence-and-vl.md)。

## 单节点与恢复

单节点首测候选（仍未实际训练验证）：

```bash
NNODES=1 TP=4 PP=2 SAVE_DIR="$PWD/outputs/qwen38-text-64layers-single-first" \
EXIT_INTERVAL=5 TRAIN_ITERS=50 SAVE_INTERVAL=5 \
  bash examples/qwen/run_qwen38_text_sft.sh sft
```

保存后恢复必须保持原来的模型、数据、并行拓扑、GBS、SEQ和thinking模式，使用原SAVE_DIR以及其checkpoint目录；双节点时两端都设置相同RESUME_FROM：

```bash
RESUME_FROM="$SAVE_DIR/checkpoints" EXIT_INTERVAL=0 TRAIN_ITERS=50 \
  bash examples/qwen/run_qwen38_text_sft.sh sft
```

首次初始化只读纯文本HF权重；resume关闭HF预加载，恢复optimizer/RNG/scheduler。不可使用16层checkpoint恢复64层。当前脚本不允许通过resume改变拓扑。`launch_manifest*.json`、`resolved_config.yaml`、各rank结构/HF导入/退出JSON、`train_logs/node-N.log`及checkpoint共同构成交接证据；本轮全64层实测证据见handoff文末。

## 原生算子

文本和VL入口直接使用框架原生TE RMSNorm和FusedAdam。当前n05+n06的64层文本SEQ4096/full重计算已完成四组各50步测试，结果见handoff；VL本轮未重跑GPU训练。
