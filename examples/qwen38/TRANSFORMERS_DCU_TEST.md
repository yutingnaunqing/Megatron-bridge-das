# Qwen3.8-Flash-Next：Transformers 在海光 DCU 上的结构兼容性测试

> 此文档是早期测试记录。最新已通过的 BF16 `grouped_mm`/FLA、八卡 DDP/TP 与 PP 探针结果，以 [DCU_TRANSFORMERS_REPRODUCE.md](DCU_TRANSFORMERS_REPRODUCE.md) 为准。

本文记录的是 **Qwen3.8-Flash-Next 官方 Transformers 文本模型结构** 在海光 BW1000 DCU 上的功能验证。目标是先把“模型结构能否前向、反向和更新参数”与“优化 kernel 能否运行”拆开；它不是完整预训练权重部署或性能测试。

## 1. 测试范围和结论

模型仓库的完整架构是 `Qwen4ExpForConditionalGeneration`，其中包含视觉和文本部分。本测试使用上游 Transformers 的 `Qwen4ExpForCausalLM`，即文本主干和语言模型头。使用随机输入与随机初始化权重，因此不验证模型效果、tokenizer、视觉输入、生成 cache、MTP 或完整 checkpoint 加载。

截至本次测试，缩小后的官方文本结构在一张 BW1000 上已经完成了：

```text
eval 前向 → 训练前向与 LM loss → backward → AdamW step
```

已通过的组合：

| dtype | MoE 后端 | GDN 后端 | 结果 |
|---|---|---|---|
| FP32 | `batched_mm` | PyTorch reference | PASS |
| BF16 | `batched_mm` | PyTorch reference | PASS |

因此，当前结论是：**官方 Transformers 的缩小文本模型结构可在 BW1000 上完成 BF16 与 FP32 基础训练计算；当前阻塞是两条可选的加速计算路径。**

## 2. 固定的软件环境

| 项目 | 实际值 |
|---|---|
| DCU | 8 × BW1000 64G |
| PyTorch | 2.10.0 |
| HIP | 6.3.26113 |
| Transformers | 5.18.0.dev0 |
| Transformers commit | `8e6083374735b8c95f13b1cf2f80cd9c7486c389` |
| flash-linear-attention | 0.5.1 |
| TileLang | `0.1.12+das.opt1.dtk2604.torch2100.2609081633.g25d759` |

Transformers 5.11.0 不含 `transformers.models.qwen4_exp`，因此不能识别 Qwen3.8-Flash-Next 配置中的 `model_type: qwen4_exp`。当前使用的源码位于：

```text
/public/home/yugeng/work/transformers_qwen38
```

测试脚本：

```text
examples/qwen38/transformers_smoke.py
examples/qwen38/transformers_ddp8_smoke.py
```

## 3. 需要理解的模型结构

官方配置的文本部分有 48 层，隐藏维度为 2560。层类型按三个 GDN 层、一个 QSA 层循环：

```text
GDN → GDN + PLE → GDN → QSA → ...（共 12 个 QSA 层）
```

每个 decoder layer 外部还使用四路 Gated Residual/Hyper-Connection，将多路残差状态混合。每层都包含稀疏 MoE：512 个 routed experts、每个 token 选择 Top-10 experts，另有一个 shared expert。

| 结构 | 在上游源码中的作用 | 本次测试的含义 |
|---|---|---|
| GDN | `Qwen4ExpTextGatedDeltaNet`；使用因果卷积和 Gated Delta Rule 维护状态 | GDN 的优化 backward kernel 当前失败，但 PyTorch reference 可通过 |
| QSA | 每四层一次；indexer 压缩 KV 块后选择稀疏可见块 | 已走到完整前向与反向；离散选块参数不从单纯 LM loss 获得梯度 |
| MoE | router 选 Top-K experts，再计算 expert MLP 与 shared expert | `grouped_mm` 路径失败，`batched_mm` 通过 |
| PLE/N-gram | 用前文 token n-gram 哈希补充位置信息 | 已包含在缩小配置中；原始 2000 万 N-gram 表被缩小 |
| Gated Residual | 四路残差状态的低秩混合和注入 | 已在完整模型图中执行 |

缩小配置使用 4 层或 8 层，而不是只调用某个独立算子。它保留 GDN、QSA、MoE、PLE 和 Gated Residual 的连接关系，同时缩小 hidden size、专家数量、词表和 N-gram 表，避免把显存问题误判为算子问题。

## 4. 一卡测试结果

### 4.1 默认优化路径：MoE GroupedGEMM 失败

命令默认使用 `grouped_mm`。在第一个 decoder layer 的 MoE expert MLP 前向中失败：

```text
torch.nn.functional.grouped_mm
  → hipblaslt_groupedgemm
  → RuntimeError: hipblaslt_groupedgemm: no valid algorithm found
```

这不是 QSA 报错，也不是显存不足。`grouped_mm` 是 PyTorch API；在 CUDA 上会走 CUDA/cuBLASLt 相关实现，在当前 HIP 栈上调用 `hipblaslt_groupedgemm`。接口存在，但 hipBLASLt 没有为当前小型、动态 expert 分组的 FP32 矩阵形状找到可执行算法。是否为普遍缺失还需要用 BF16、不同 token 数和不同分组 shape 做最小复现确认。

### 4.2 MoE `batched_mm` 成功绕过 GroupedGEMM

Transformers 提供了公开接口：

```python
model.set_experts_implementation("batched_mm")
```

`batched_mm` 会为每个 token-expert 对选取对应 expert 权重并调用 `torch.bmm`，不会调用 `grouped_mm`。它保留 router、Top-K 选择、expert 权重和 MoE 数学含义，只改变执行方式。它适合正确性验证，性能和临时内存都不适合作为正式部署结论。

在脚本中对应：

```bash
--experts batched_mm
```

### 4.3 GDN FLA/TileLang backward 失败

在 MoE 改为 `batched_mm` 后，eval 前向、训练前向和 loss 均通过；反向在 FLA 的 Gated Delta Rule kernel 失败：

```text
fla.ops.gated_delta_rule.chunk
  → tilelang ... chunk_bwd_dqkwg_tilelang
  → tvm runtime rocm_module.cc
  → hipModuleLaunchKernel ... invalid argument
```

这说明 GDN 数学本身没有被证明不支持：失败的是 FLA 选择的 TileLang 优化 backward kernel。堆栈明确已经走到 ROCm/HIP 代码路径，不应简单表述为“只支持 NVIDIA”；更准确的表述是：**当前 TileLang/FLA/DTK 版本组合未能在 BW1000 上成功启动该 kernel。** 可能需要检查生成 kernel 的架构目标、launch 参数、动态共享内存和 TileLang 的 BW1000 支持。

### 4.4 GDN PyTorch reference 通过

上游 `modeling_qwen4_exp.py` 保留了 `torch_chunk_gated_delta_rule` 和 `torch_recurrent_gated_delta_rule` 的完整 PyTorch 实现，FLA 只是其可选装饰器实现。测试脚本的：

```bash
--gdn torch_reference
```

在单次 Python 进程中选择这两个上游 reference 函数，不修改 Transformers 源码和模型数学。使用它配合 `batched_mm` 的结果：

| 输出目录 | dtype | loss | 结果 |
|---|---:|---:|---|
| `runs/qwen38/hf_tiny_fp32_reference` | FP32 | 6.2493014 | PASS |
| `runs/qwen38/hf_tiny_bf16_reference` | BF16 | 6.2493234 | PASS |

两个结果文件均保存了实际环境、配置、loss、显存和失败/成功状态。

`parameters_without_grad` 中的 3 个 QSA indexer 参数不是运行失败：QSA 通过离散 Top-K 决定可见块，纯 LM loss 不会对离散选择本身回传梯度。其他模型参数已完成梯度计算和更新。

## 5. 复现已通过的单卡参考路径

```bash
cd /public/home/yugeng/work/Megatron-bridge-das

HIP_VISIBLE_DEVICES=0 \
python -u examples/qwen38/transformers_smoke.py \
  --profile tiny \
  --device cuda \
  --dtype bfloat16 \
  --attention eager \
  --experts batched_mm \
  --gdn torch_reference \
  --seq-len 32 \
  --steps 1 \
  --output runs/qwen38/hf_tiny_bf16_reference
```

每次运行会产生：

```text
effective_config.json  实际模型结构配置
result.json            环境、loss、峰值显存、状态
error.txt              异常堆栈（失败时）
```

## 6. 八卡 DDP 测试

`transformers_ddp8_smoke.py` 使用 `torchrun` 和 PyTorch DDP，默认使用 8 层而不是单卡脚本的 4 层：

```text
GDN → GDN + PLE → GDN → QSA → GDN → GDN + PLE → GDN → QSA
```

它在 8 个 rank 上各创建一份同样的缩小模型；每个 rank 使用不同随机 token，`loss.backward()` 时通过 NCCL/RCCL All-Reduce 同步梯度，然后各 rank 执行相同的 AdamW 更新。QSA indexer 的离散 Top-K 参数不会获得 LM loss 梯度，因此脚本开启了 DDP 的 `find_unused_parameters=True`。

在容器内运行：

```bash
cd /public/home/yugeng/work/Megatron-bridge-das

torchrun --standalone --nproc_per_node=8 \
  examples/qwen38/transformers_ddp8_smoke.py \
  --profile tiny \
  --layers 8 \
  --dtype bfloat16 \
  --attention eager \
  --experts batched_mm \
  --gdn torch_reference \
  --seq-len 64 \
  --local-batch-size 1 \
  --steps 2 \
  --output runs/qwen38/ddp8_tiny_bf16_reference
```

成功标志为：

```text
GLOBAL_MEAN_LOSS ...
PASS: 8-rank DDP forward/loss/backward/all-reduce/update
```

结果目录中的 `result.json` 会记录 world size、全局 batch size、每卡参数量、平均 loss 和 8 个 rank 中的峰值显存上限。若任何一个 rank 出错，对应的 `rank<N>_error.txt` 会保存该 rank 的 traceback。

这验证的是数据并行，不是模型并行：8 卡不会让单张卡只保存八分之一模型，也不能据此判断完整 125B 权重可以放入 8 张卡。它能确认 RCCL 通信、DDP 未使用参数处理、梯度同步和同步更新是否正常。

宽度更接近官方配置的后续命令是：

```bash
torchrun --standalone --nproc_per_node=8 \
  examples/qwen38/transformers_ddp8_smoke.py \
  --profile width \
  --layers 8 \
  --dtype bfloat16 \
  --attention eager \
  --experts batched_mm \
  --gdn torch_reference \
  --seq-len 32 \
  --local-batch-size 1 \
  --steps 1 \
  --output runs/qwen38/ddp8_width_bf16_reference
```

`width` 保留官方 hidden size 2560、attention/GDN head 维度和 Top-10 路由；为功能测试而缩小词表、N-gram 表和专家总数。先完成 tiny 八卡测试，再运行此命令。

## 7. 接下来的测试顺序

先保持两个参考路径不变，依次增加序列长度、训练步数和模型宽度：

```text
tiny / BF16 / seq 32   已通过
tiny / BF16 / seq 128  下一步
tiny / BF16 / seq 512  下一步
width / BF16 / seq 32  保留主要 hidden/head 维度
8-rank DDP             验证 RCCL 通信与梯度同步
```

之后单独恢复一条优化路径，避免两个错误混在一起：

```text
MoE：--experts grouped_mm --gdn torch_reference
GDN：--experts batched_mm --gdn fla
```

完整 125B checkpoint、多模态视觉路径、长上下文 cache、模型并行和吞吐性能均属于后续独立里程碑。参考路径通过不代表完整模型已经可部署；它说明模型结构与基础 DCU 算子可以计算，从而让优化 kernel 的问题能够被单独、可复现地提出。
