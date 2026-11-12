<!-- Copyright (c) 2026 Hygon Information Technology Co., Ltd. -->

# Qwen3.8 视觉 RoPE 性能结果

已按用户最终要求完成单轮粗测：保留已完成的 reference 30 步，flash 运行 20 步，只比较两组第 10～20 步。第 5 步独立采集 prof。旧的三轮完整矩阵已停止，不再自动重跑。

配置：单节点 8 卡，TP2/PP4、GBS64/MBS1、SEQ4096，真实 mantis-birds 数据，16 层语言主干和完整 27 层 ViT，本轮冻结视觉。

| 实现 | 平均 step ms | samples/s |
| --- | ---: | ---: |
| reference | 15315.09 | 4.1789 |
| flash | 14949.50 | 4.2811 |

粗测 step 耗时降低 2.39%，吞吐提升 2.45%。两组都记录到外部 GPU 进程；本结果用于判断优化方向，不能作为严格或长期收益承诺。

融合组训练 prof 中，rank 0 有 1,728 次视觉 FlashAttention 和 3,456 次 rotary kernel，符合 64 microbatches × 27 层 × Q/K 各一次。相对原路径 41,472 次视觉旋转相关 kernel，融合结构实际生效。

[详细单轮结果](qwen38-vision-rope-quick-20261111.md) · [算子与 GPU 验证记录](qwen38-vision-rope-gpu-20261111.md)

本次输出位于仓库相邻的 `checkpoints/qwen38_rope_gpu_20261111_1236/quick_compare/`：`summary.json`、`timing_steps.csv`、`report.md` 和 `flash_20steps/torch_profile/rank-{0,4}.json.gz`。
