# Conv3d保留后的metadata单独训练对照

最终决策：Linear已从活动训练代码移除并备份；单节点 `run1_qwen38_vl_single_node.sh` 默认使用 **原Conv3d + flash RoPE + prepared metadata**，Graph保持none。精度优先，prepared在回退后的完整27层TP2固定权重/输入对照中逐层输出和所测梯度逐位一致。

**性能表中的23.71%只是受干扰的原始均值差，不能宣称为metadata独立训练加速。** C0计时窗口18次采样、C1窗口13次采样均发现外部进程；C0还含44001.5ms异常步，样本完整保留。保留prepared的依据是精度零差、独立组件准备耗时下降及结构开销减少，而不是把两组异常负载差异归因于优化。


C0=flash RoPE+Conv3d+reference metadata；C1=flash RoPE+Conv3d+prepared metadata。各20步，prof5（rank0/2/4/6），计时10～20。

| 组 | mean ms | median ms | min/max ms | samples/s | 相对C0 mean下降 | 外部干扰 |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| C0_flash_conv_reference | 19958.291 | 18002.800 | 15241.0/44001.5 | 3.2067 | 0.00% | True |
| C1_flash_conv_prepared | 15226.827 | 15017.000 | 14978.0/17378.2 | 4.2031 | 23.71% | True |

| rank0 | kernel总数 | im2col | rotary | scalar读取 | 视觉varlen |
| --- | ---: | ---: | ---: | ---: | ---: |
| C0_flash_conv_reference | 288587 | 108916 | 3456 | 11659 | 1728 |
| C1_flash_conv_prepared | 281490 | 108916 | 3456 | 2699 | 1728 |

原始命令、配置、源码快照、监控、日志与prof均在实验目录。未重复B0/B1/B2，未启用Linear或Graph。单轮粗测不等于统计显著；如有外部干扰须结合计时窗口的监控解读。

实验目录：`/public/home/yugeng/work/checkpoints/qwen38_metadata_conv_20261111`。

## 独立metadata组件对照

同样使用实际位置插值、RoPE表、packed参数代码，每次等待该函数GPU工作完成。reference/prepared_warm交替顺序配对，各30次；cold每次清缓存，共15次。先核对位置、cos/sin及cu_seqlens逐项相等，再计时。仅为该准备函数的wall time，不代表整个step加速。

| 网格 / token数 | reference中位ms | prepared命中ms | prepared冷缓存ms |
| --- | ---: | ---: | ---: |
| 两张28×28 / 1568 | 3.0241 | 0.1672 | 0.8018 |
| 真实双图网格26×34、30×32 / 1844 | 3.2493 | 0.1891 | 0.8309 |
| 一图加两帧 / 1568 | 2.7436 | 0.1510 | 0.7853 |

默认项在C0/C1完成后调整；两组原训练命令已显式指定flash和对应metadata，默认切换不改变其所测计算。dryrun已核实flash/prepared与Graph none，shell语法检查通过；通用provider/recipe默认仍为reference，避免影响其他入口。通用启动器增加metadata枚举校验，并将rope/metadata环境传给远端MPI；多节点训练本轮未测。

回退metadata：`VISION_METADATA_BACKEND=reference bash examples/qwen/run1_qwen38_vl_single_node.sh train`。当前不再支持 `model.vision_patch_embed_backend=linear`，恢复Linear前必须审阅备份与精度报告。

## 精度与回退验证

- `correctness.xml/log`：94 passed、2 skipped，只有相关metadata/RoPE测试。
- `full_vit_tp2/rank-0.json`、rank-1：实际回退代码的完整27层审计；`accuracy_checks.json` 的16项逐位一致判定全部true。
- Linear备份：`/public/home/yugeng/work/checkpoints/qwen38_linear_backup_20261111`，源码hash、原文件、恢复补丁及dry-run校验日志均保留。活动 `utils.py` 与Bridge基准版本无diff。
- 本轮未实现Graph，详细方案见 [视觉CUDA/HIP Graph](qwen38-vision-cuda-graph-plan.md)。

## 四阶段profile对Graph预期的限制

C1第5步四个PP代表rank（单位秒；并集不能相加）：

| rank / PP阶段 | 非NCCL kernel并集 | NCCL kernel并集 | ProfileStep跨度 |
| --- | ---: | ---: | ---: |
| rank0 / 首阶段含视觉 | 9.198 | 3.231 | 15.105 |
| rank2 / 第二阶段 | 6.112 | 8.946 | 15.104 |
| rank4 / 第三阶段 | 6.117 | 8.966 | 15.105 |
| rank6 / 最后语言阶段 | 13.590 | 1.780 | 15.105 |

该快照中最后语言阶段非通信工作更重，支持先调查语言前反向/LM head及流水线依赖；不能只根据某rank kernel减少推断step同比下降，也不能仅凭并集指标认定唯一瓶颈。Graph可先做小范围试点，若只有视觉局部收益则保持可选。

完整工件另含 `metadata_microbenchmark.json/log`、`pipeline_analysis.json/log`、`graph_shape_histogram.json`、`adoption.json`、源码前后manifest、default_dryrun配置。控制器已complete，无待运行组。
