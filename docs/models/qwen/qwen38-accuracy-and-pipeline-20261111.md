# Qwen3.8 视觉优化的数值一致性与流水线分析

最新状态：Linear已按精度优先要求撤回，相关代码/配置/旧审计工具备份于 `/public/home/yugeng/work/checkpoints/qwen38_linear_backup_20261111`。本文Linear结果与命令为历史记录；当前Conv3d+metadata单独验收见handoff第0节，Graph最新方案见 [CUDA/HIP Graph方案](qwen38-vision-cuda-graph-plan.md)。当前审计脚本已不含Linear或bias-followup，重用历史实验须先审阅备份恢复步骤。


2026-11-11。本轮只新增精度审计工具和分析文档，未修改训练算法、后端默认值或通信配置。结论：**所测RoPE融合与metadata准备逐位一致；当前带融合bias的Linear不逐位一致，完整27层ViT真实图片输出relative L2差异约1.062%，不能称为数值无损。**

## 1. 验证方法和适用范围

新增工具：`examples/qwen/audit_vision_accuracy.py`。使用实际Bridge模块，完整27层ViT、hidden1152、heads16、FFN4304、merger输出5120，TP2、BF16；模型只初始化一次，各路径共享完全相同的参数，不进行optimizer更新。原路径重复一次，检查比较过程自身的噪声。

两种场景：

- 真实mantis-birds样本：`train-shard-000000.tar` 的首个 `.jpgs`，两图尺寸500×375、500×491；用项目restricted pickle decoder和本地HF image processor，min_pixels200704、max_pixels1003520。实际grid为 `[[1,26,34],[1,30,32]]`，共1844个patch，冻结视觉，只比较前向；这不是原训练随机采样的第一个microbatch。
- 完整27层的可训练多帧场景：合成40个patch，grid `[[1,4,6],[2,2,4]]`，用同一BF16上游梯度比较输入梯度和每个TP本地参数梯度。用于验证反向，不代表1844-token真实图片的反向已覆盖。

权重是固定seed1234的初始化，**未加载预训练视觉权重，未进行任务验证集评测**。因此本报告量化的是数值误差，不是下游任务accuracy，也不能据此保证长期训练质量相同。

误差定义：`relative L2 = ||candidate-reference||₂ / ||reference||₂`。max abs为张量最大绝对差；bitwise_equal按原dtype原始字节比较。余弦与范数使用float64归约，避免长向量float32 dot的累积误差。参数整体relative L2按所有已计算梯度的误差平方和/参考平方和开根号；两个TP rank分别报告，不将分片或复制参数重复计数后冒称唯一全局值。

## 2. 当前Linear的误差究竟多大

以下为完整主审计，不是之前2层、hidden144的小模型结果。

| 比较对象 | 场景 | relative L2 | max abs |
| --- | --- | ---: | ---: |
| FP32 Linear vs FP32 Conv3d，patch输出 | 同一BF16量化数据/权重转FP32，关闭TF32 | 约0.0000299% | 1.7881e-6 |
| BF16 Linear vs BF16 Conv3d，patch输出 | 1844个真实patch | **0.2824%** | **0.0078125** |
| 完整27层+merger的最终视觉输出 | 同一真实图片、同一权重 | **1.0619%** | **0.0087890625** |
| 完整27层的输入梯度 | 40-token多帧、同一上游梯度 | **1.4778%** | **0.007080078125** |
| 完整27层的参数梯度整体 | 同上，TP rank0 / rank1 | **1.0753% / 1.0997%** | 不以不同参数尺度的统一abs判断 |
| 单个参数中最大的relative L2 | 同上，layer14 QKV前LayerNorm weight | **1.6443%** | 0.001953125 |

真实图片最终输出RMS abs误差约0.0017975，参考输出RMS约0.16928、最大绝对值0.89453，余弦约0.9999436。约80.82%的输出元素不是逐位相同；这是“元素出现差异的比例”，**不代表80.82%准确率下降**。同样，1.0619%的张量relative L2不代表任务accuracy下降1.0619%。

误差随BF16层传播的样例：patch 0.2824% → 第1层0.2112% → 第6层0.3867% → 第13层0.5811% → 第27层0.8908% → merger后1.0619%。

**需要收紧上一轮结论**：108项检查通过证明所覆盖算子与小型集成满足当时门槛，不能外推为完整27层数值无损。此前小模型参数梯度relative L2门槛1%不能直接用于宣称本次完整模型通过；完整模型已经测到约1.1%的整体梯度差异及1.64%的局部参数差异。本次没有提高旧门槛将这些差异包装为“精度一致”。

## 3. 为什么数学等价，BF16仍有差异

输入patch展开顺序和参数没有变化，FP32对照也接近计算舍入量级，排除了明显的通道/位置变换错误。主要区别在计算后端、归约顺序以及bias舍入位置：

- 原Conv3d路径可见逐patch im2col、小GEMM，之后另有bias加法；卷积结果先落为BF16，再做bias相加。
- 当前 `F.linear(x,w,b)` 使用含bias的GEMM，bias可在更高精度累加中融合后一起转BF16。
- 因此bias相加之前是否已经BF16舍入不同。少量GEMM归约差异还会在后续BF16层继续传播。

单算子FP32参考（数据/权重/上游梯度均先固定为同一BF16值再提升计算精度，TF32关闭）：

| patch输出 | 相对FP32 Conv3d参考的relative L2 |
| --- | ---: |
| BF16 Conv3d | 0.2331% |
| BF16 Linear | 0.1659% |

这表示本例Linear的patch输出甚至更接近FP32算术结果，但**更接近FP32不等于与原BF16模型一致，也不保证最终任务质量更高**。

### 3.1 分离bias的实验也做了，仍不能保证全模型一致

仅在审计脚本中替换patch forward为 `F.linear(x,w,None) + b`，训练实现没有修改。

| 相对原Conv3d | 当前融合bias Linear | 实验性分离bias Linear |
| --- | ---: | ---: |
| 同一真实图片的patch输出relative L2 | 0.2824% | 0.002823% |
| 同一真实图片的完整ViT输出relative L2 | 1.0619% | **0.9910%** |

首层差异大幅减少，但剩余微小归约差异仍通过27层BF16计算扩散；不能因为算子误差小就推断完整模型误差同样小。另一个40-token随机多帧场景中，分离bias的完整输出差异约0.7439%、输入梯度1.1063%、参数梯度整体约0.7936%～0.7976%。该补测的合成输入与主审计不同，数值只在同一次配对中比较，不跨两次随机样本拼接比较。

因此尚未将分离bias方案写入训练路径，也未声称其具有B1的性能。它预计会增加一次bias运算，需要独立性能验证。

## 4. RoPE融合前后是否一致

本轮直接比较原 `apply_rotary_pos_emb_absolute` 与 `flash` 视觉路径，后者保持FP32系数、FP32中间乘加、关闭FMA融合、BF16输出。

| 检查 | 所测结果 |
| --- | --- |
| 实际1844-token视觉位置，Q/K，FP32/BF16，各3组随机输入 | 前向与输入梯度全部逐位一致，max abs=0、relative L2=0 |
| 真实双图，完整27层ViT最终输出及逐层输出 | 逐位一致，差值0 |
| 40-token多帧，完整27层输出、输入梯度、每个已计算参数梯度 | 两个TP rank全部逐位一致，差值0 |
| 原路径重复运行控制组 | 输出与所测梯度逐位一致，排除本次比较中的随机噪声 |

之前独立benchmark在T1568/2048/4096上的BF16输出/梯度差值也是0；本次补充了实际视觉坐标、完整深度与TP2。结论仅针对已测软件/设备、形状和固定权重输入，不能宣称所有平台、FP8、任意权重/输入均已形式化证明完全一致。**语言M-RoPE未改，本轮不涉及其融合。**

## 5. metadata是否额外引入误差

本轮额外对照Conv3d+flash以及Linear+flash两条路径，各自只切换reference/prepared metadata：

- 真实图片完整27层逐层与最终输出均逐位一致。
- 40-token多帧的最终输出、输入梯度与已计算参数梯度均逐位一致。

因此在已测样例中，B0→B2的数值变化来自Linear，metadata准备本身没有额外误差。

## 6. 为什么省了大量patch kernel，step却几乎没变

### 6.1 kernel数量、kernel sum与step关键路径不是一回事

B0→B1，rank0 kernel数288587→70755，消失的217832个主要是短小im2col/GEMM。减少75.5%的kernel数量不能解释为减少75.5%的训练时间。

当前是TP2/PP4，每step64个microbatch。只加速rank0/1的视觉前向；视觉冻结，没有相应视觉反向可加速，语言前反向、logits/loss、其他PP stage及通信仍然存在。稳态流水线吞吐由关键阶段和通信依赖共同决定。示意上，若四stage周期为100/120/110/120ms，首stage从100降到80ms，稳态周期仍可能是120ms；这只是解释机制，不是本项目实测的stage时间。

原profile中patch kernel的约1.59s是该rank的GPU duration sum，不能直接从全step中减掉：它可能与其他rank工作、通信、pipeline等待重叠；不同profile还存在外部干扰。

### 6.2 当前配置明确存在较强同步边界

实际训练配置：`batch_p2p_comm=true`、`batch_p2p_sync=true`、`overlap_p2p_comm=false`、`variable_seq_lengths=false`。

MCore `megatron/core/pipeline_parallel/p2p_communication.py` 约415～419行，在上述batch P2P同步配置下调用 `torch.cuda.synchronize()`；源码将其作为race保护。不能为追求速度直接删掉这段同步。本轮未修改MCore或通信参数。

| 同一个第5步profile | B1 rank0 | B1 rank4 | B2 rank0 | B2 rank4 |
| --- | ---: | ---: | ---: | ---: |
| ProfileStep span，ms | 15144.33 | 15144.24 | 14926.75 | 14926.65 |
| hipDeviceSynchronize 次数 | 70 | 134 | 70 | 134 |
| 这些同步调用的CPU duration合计，ms | 3878.33 | 11851.35 | 4554.41 | 11733.64 |
| 非NCCL kernel时间并集，ms | 7642.77 | 6123.85 | 7548.29 | 6096.09 |
| NCCL kernel时间并集，ms | 4280.97 | 8861.99 | 4832.29 | 8748.89 |

这些是同一进程trace的时间并集或CPU调用时间；互相可重叠，**不能相加**，也不能当成全部可消除的开销。NCCL kernel可能包含等其他stage/stream的时间，不等于网络带宽瓶颈。没有这些kernel的区间也不等于整台GPU空闲：可能有拷贝、其他进程工作或host等待。

因此当前证据支持“局部视觉减负后，流水线同步/其他阶段仍限制吞吐”，但只采了rank0/rank4，**不能据此严格断言最慢的是哪一个PP stage**；若要定位具体关键阶段，应补齐各stage的forward/backward/P2P时间与rank2/6等prof。不要把推测写成已证明的LM head或某rank瓶颈。

### 6.3 原测试受干扰，不能用B1均值断言没有加速或发生退化

第10～20步：B0 mean15.1825s、median15.1660s；B1 mean17.2190s、median15.0198s，其中第17步37.63s异常被保留；B2 mean14.9193s、median14.9242s。

- B1相对B0的median约下降0.96%，只是方向性补充，不替换约定mean指标。
- B2对B0的mean约下降1.73%，仍是单轮受干扰粗测。
- B0的第5步profile为19.381s、明显不同于其稳态15.18s；不能把B0/B1 profile的时间差全部归因于Linear。
- 因此能明确证明的是结构开销下降，暂不能给出严格隔离的Linear独立整步收益。

## 7. 当前选择与后续验收

- **数值保真优先**：保留Conv3d；已测RoPE融合和metadata可以分别保持。三者都保留原backend开关，默认值未在本轮改变。
- **Linear继续作为实验选项**：它在数学上等价，patch单算子FP32参考误差也合理，但完整BF16模型不是逐位等价，不能标记为“无损优化”。分离bias只缓解，尚未完全解决。
- 使用真实预训练/恢复权重，在固定验证样本上进一步比较视觉特征、LM logits/loss，随后再做任务评测或训练收敛对照，才能回答最终质量是否相同。当前没有此项结果，不从20步随机初始化loss推断任务accuracy。

已有训练日志只能说明：同seed配方第1步B0/B1 loss13.57141/13.57155，差约0.0010%；第20步3.366511/3.390573，差约0.7147%。这是两条独立优化轨迹，不能视为单次算子误差或验证集accuracy，也不是长期收敛证明。

## 8. 工件和复现

实验根目录：`/public/home/yugeng/work/checkpoints/qwen38_accuracy_20261111`。

- `accuracy_summary.json`：精度结论汇总；`full_vit_tp2/rank-0.json`、`rank-1.json`：主审计完整逐层/逐参数指标。
- `full_vit_tp2_bias/`：分离bias实验，不能混用其不同合成样本与主审计比较。
- `pipeline_analysis.json/log`、`analyze_pipeline.py`：六份现有rank0/rank4 trace的区间并集、runtime和CPU操作汇总。
- `bias_rounding_probe.log`：独立小算子的bias舍入验证，仅用于数值，不作性能计时。
- `full_vit_tp2_initial/` 为初次归约指标记录；其长向量float32 cosine有归约误差，最终使用上面的 `full_vit_tp2/`。重跑只改审计指标精度与比较覆盖，未改模型计算。

在yg_qwen38中source既有DTK兼容环境与worktree环境后：

```bash
uv run --no-project --python /usr/local/bin/python python -m torch.distributed.run \
  --standalone --nproc_per_node=2 examples/qwen/audit_vision_accuracy.py \
  --hf-path /public/home/yugeng/work/models/Qwen3.8-27B \
  --shard /public/home/yugeng/work/datasets/mantis-birds-energon-v1/train-shard-000000.tar \
  --output <新输出目录>
```

加 `--bias-followup` 可单独运行分离bias比较；脚本只临时替换该审计进程中的patch forward，不修改项目训练实现。
