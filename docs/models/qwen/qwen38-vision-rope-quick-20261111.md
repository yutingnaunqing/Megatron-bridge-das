<!-- Copyright (c) 2026 Hygon Information Technology Co., Ltd. -->

# Qwen3.8 RoPE 单轮粗测结果

粗测观察到 step 耗时降低 2.39%。

按用户要求：保留当前 reference 30 步，仅追加 flash 20 步，不做三轮重复和其他测试矩阵。当前为冻结视觉，16 层语言主干加完整 27 层 ViT，TP2/PP4、GBS64/MBS1、SEQ4096。

两组均使用第 10～20 步的相同窗口；flash 的 profiler 安排在第 5 步（start=5/end=6），未进入计时窗口。flash 保留 reference 的 30 步 LR decay 配置。

| backend | 平均 step ms | samples/s |
| --- | ---: | ---: |
| reference | 15315.09 | 4.1789 |
| flash | 14949.50 | 4.2811 |

step 耗时降低 2.39%，吞吐提升 2.45%。

资源监控检测到外部 GPU 进程：reference=True，flash=True。
这是单次方向性观测；存在资源竞争，差异可能包含竞争影响，不能作为严格基准。

## Flash 训练 prof

| rank | kernel 总数 | rotary kernel | 视觉 FA 次数 |
| --- | ---: | ---: | ---: |
| rank-0.json.gz | 288587 | 3456 | 1728 |
| rank-4.json.gz | 28143 | 0 | 0 |

原始训练日志、资源监控、prof 和汇总均保存在实验目录的 quick_compare/。没有重新运行完整 reference 训练 prof；可结合已有算子 prof 核对优化结构。

已停止原三轮测试矩阵，没有继续自动重跑。prof 中的 3,456 次 rotary kernel 对应 64 × 27 × 2，融合路径已在完整训练中执行。
