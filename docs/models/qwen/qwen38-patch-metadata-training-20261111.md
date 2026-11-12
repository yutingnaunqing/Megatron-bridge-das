# Patch Linear / metadata 单轮训练结果

最新状态：Linear已按精度优先要求撤回，相关代码/配置/旧审计工具备份于 `/public/home/yugeng/work/checkpoints/qwen38_linear_backup_20261111`。本文Linear结果与命令为历史记录；当前Conv3d+metadata单独验收见handoff第0节，Graph最新方案见 [CUDA/HIP Graph方案](qwen38-vision-cuda-graph-plan.md)。当前审计脚本已不含Linear或bias-followup，重用历史实验须先审阅备份恢复步骤。


精度补充：原108项通过仅覆盖当时的算子/小模型门槛。完整27层TP2新审计发现当前Linear的真实图片输出relative L2约1.062%，不是逐位无损；RoPE融合与metadata所测输出/梯度逐位一致。详见 [最新精度与流水线分析](qwen38-accuracy-and-pipeline-20261111.md)，不以性能结构改善推断任务质量不变。


各组20步，prof5，计时10～20。冻结视觉，16语言层+完整ViT。

三组是逐项叠加的实验：

| 编号 | 视觉RoPE | patch embedding | metadata | 用途 |
| --- | --- | --- | --- | --- |
| B0 | flash（已融合） | 原Conv3d | reference原路径 | 本轮起点，不是未融合RoPE基线 |
| B1 | flash | Linear | reference原路径 | B1/B0比较patch Linear |
| B2 | flash | Linear | prepared一次准备+有界缓存 | B2/B1比较metadata，B2/B0比较两项累计 |

三组均不包含Q/K联合旋转或Graph，语言M-RoPE未改。

| 组 | mean ms | median ms | samples/s | 增量耗时下降 | 累计耗时下降 | 外部干扰 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| B0_flash_conv_reference | 15182.500 | 15166.000 | 4.2154 | 0.00% | 0.00% | True |
| B1_flash_linear_reference | 17218.973 | 15019.800 | 3.7168 | -13.41% | -13.41% | True |
| B2_flash_linear_prepared | 14919.336 | 14924.200 | 4.2897 | 13.36% | 1.73% | True |

| 组/rank0 | kernel总数 | im2col | rotary | scalar读取 | 视觉varlen调用 |
| --- | ---: | ---: | ---: | ---: | ---: |
| B0_flash_conv_reference | 288587 | 108916 | 3456 | 11659 | 1728 |
| B1_flash_linear_reference | 70755 | 0 | 3456 | 11659 | 1728 |
| B2_flash_linear_prepared | 63658 | 0 | 3456 | 2699 | 1728 |

原始命令、配置、日志、监控、prof与源码快照保存在本实验目录；失败或受资源影响的未完成尝试另存invalidated。GPU duration是kernel sum，不能直接当作step收益。

实验目录：`/public/home/yugeng/work/checkpoints/qwen38_patch_metadata_20261111`。

B1控制器在训练已正常结束后遇到hy-smi超时，退出码未保留；使用20步、正常结束时间戳和footer确认完成。B2前补充rotary低精度buffer的FP32系数保护；在本次默认FP32频率下为no-op，B0/B1使用reference metadata不执行此分支。源码前后manifest均已保存。

解读：B2对B0的观测mean下降1.73%、吞吐上升1.76%，仅为单轮粗测。B1第17步37630ms受资源/监控异常影响且完整保留，不能将其mean解释为Linear退化，也不能将B2/B1的13.36%解释为metadata独立加速。三组全程均有干扰标记；B0计时窗口13次、B2计时窗口12次采样未见外部GPU进程，B1窗口发现外部进程。详细状态、组别配置与实现位置见 [handoff](../../../HANDOFF_QWEN38_PERF.md)。
