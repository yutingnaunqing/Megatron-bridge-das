# Qwen3.8-Flash-Next Transformers DCU 测试 Handoff

> 此文档是早期测试记录。最新已通过的 BF16 `grouped_mm`/FLA、八卡 DDP/TP 与 PP 探针结果，以 [DCU_TRANSFORMERS_REPRODUCE.md](DCU_TRANSFORMERS_REPRODUCE.md) 为准。

## 目标

不使用 Megatron，直接调用 Hugging Face Transformers 上游的 Qwen3.8-Flash-Next 文本模型结构，在海光 BW1000 DCU 上验证模型前向、反向和参数更新；定位不能运行的优化算子路径，为后续算子适配提供可复现依据。

## 当前结论

- 官方 Transformers 文本模型 `Qwen4ExpForCausalLM` 的缩小配置已在单张 BW1000 上通过 FP32、BF16 的前向、LM loss、反向和 AdamW 更新。
- 测试使用随机初始化权重，只验证结构和计算兼容性，不代表完整 checkpoint 推理效果、预训练、视觉输入或模型部署完成。
- QSA、GDN、MoE、PLE/N-gram、Gated Residual 均已进入测试模型图；QSA indexer 的离散 Top-K 参数未从纯 LM loss 获得梯度属于预期现象。

## 固定环境

| 项目 | 值 |
|---|---|
| 容器 | `yg_qwen3.8` |
| DCU | 8 × BW1000 64G |
| PyTorch / HIP | 2.10.0 / 6.3.26113 |
| Transformers | 5.18.0.dev0 |
| Transformers commit | `8e6083374735b8c95f13b1cf2f80cd9c7486c389` |
| Transformers 源码 | `/public/home/yugeng/work/transformers_qwen38` |
| FLA / TileLang | 0.5.1 / `0.1.12+das.opt1.dtk2604.torch2100.2609081633.g25d759` |

Transformers 5.11.0 不包含 `qwen4_exp` 模型模块，不能直接加载该模型结构。

## 已验证的运行方式

脚本：[transformers_smoke.py](transformers_smoke.py)

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

已通过结果：

| 输出目录 | 配置 | 结果 |
|---|---|---|
| `runs/qwen38/hf_tiny_fp32_reference` | FP32 + `batched_mm` + GDN PyTorch reference | PASS |
| `runs/qwen38/hf_tiny_bf16_reference` | BF16 + `batched_mm` + GDN PyTorch reference | PASS |

每个运行目录保留 `effective_config.json`、`result.json`，失败时另有 `error.txt`。

## 已定位的加速路径问题

| 组件 | 失败路径 | 报错 | 当前替代路径 |
|---|---|---|---|
| MoE | `torch.nn.functional.grouped_mm` → hipBLASLt | `hipblaslt_groupedgemm: no valid algorithm found` | `--experts batched_mm` |
| GDN | FLA → TileLang Gated Delta Rule backward | `hipModuleLaunchKernel ... invalid argument` | `--gdn torch_reference` |

说明：两者都不是模型结构或基础 DCU 算子完全不支持。MoE 问题是 hipBLASLt 无法为当前动态分组矩阵形状选择算法；GDN 问题是当前 FLA/TileLang/DTK 组合无法在 BW1000 上启动优化 backward kernel。普通 PyTorch reference 路径均可执行。

## 重要文件

| 文件 | 用途 |
|---|---|
| [TRANSFORMERS_DCU_TEST.md](TRANSFORMERS_DCU_TEST.md) | 完整测试背景、模型结构、结果和后续计划 |
| [transformers_smoke.py](transformers_smoke.py) | 单卡结构兼容性测试 |
| [transformers_ddp8_smoke.py](transformers_ddp8_smoke.py) | 8 卡 DDP 测试脚本 |
| [config.official.json](config.official.json) | 官方模型配置副本 |

## 八卡状态与下一步

`transformers_ddp8_smoke.py` 已完成语法检查。它使用 8 层缩小模型，通过 `torchrun` 验证 DDP 前向、反向、RCCL All-Reduce 和参数同步；它是数据并行，不会切分模型参数。

尚未完成实际八卡执行：上次启动时宿主机 Docker 客户端异常调用不存在的 `/usr/bin/ZG9ja2Vy`，未进入 `yg_qwen3.8` 容器。恢复 Docker 访问后先执行：

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

成功标志：

```text
PASS: 8-rank DDP forward/loss/backward/all-reduce/update
```

后续顺序建议：

1. 完成 tiny 8 卡参考路径测试。
2. 用 `--profile width` 保留官方 hidden size、head 维度和 Top-10 MoE 路由进行 8 卡测试。
3. 固定 GDN reference，单独复现 `--experts grouped_mm` 的 MoE 问题，提取最小 shape/dtype 复现。
4. 固定 `batched_mm`，单独复现 `--gdn fla` 的 GDN TileLang backward 问题，记录 kernel、shape、DTK 和 TileLang 版本。
5. 最后评估完整权重、模型并行、cache、视觉路径和性能；当前结果不覆盖这些范围。
