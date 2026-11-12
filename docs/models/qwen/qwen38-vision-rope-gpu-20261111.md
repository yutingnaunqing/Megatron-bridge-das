<!-- Copyright (c) 2026 Hygon Information Technology Co., Ltd. -->

# Qwen3.8 视觉 RoPE：真实 GPU 验证和诊断 profile

2026-11-11，北京时间。此记录区分 GPU 正确性、算子 kernel 结构和完整训练收益。有效孤立算子测试已补齐；当前尚无无竞争的完整训练计时结果，不能宣称完整 step 已加速。

## 14:06 更新：有效孤立算子结果

前向测试在 13:28 完成，前反向受干扰的一次已归档，14:06 重跑完成。两份有效结果的运行监控均未检出外部 GPU 作业；详细数值见 [性能报告](qwen38-vision-rope-performance-20261111.md)。

- T=1568/2048/4096 时，flash 前向耗时相对 reference 降低 47.5%～73.0%，加速约 1.90～3.71 倍。
- 前反向耗时降低 29.3%～57.0%，加速约 1.41～2.33 倍。
- 输出及输入梯度相对 L2 误差仍为 0；前向 kernel 数 648→56，前反向 1134→110。
- 有效 prof 位于实验目录 `microbenchmark/fwd/`、`microbenchmark/fwd_bwd/`；此前 `diagnostic_contended_*` 继续排除计时。
- 完整训练基线启动前，另一组外部 8 卡作业进入节点，暂未开始正式训练。临时执行脚本已增加有效结果复用、无效结果归档，以及遇到外部作业干扰后重新等待并重试。

## 环境和设备可见性

- bw45，`yg_qwen38`，8 张 BW1000 64G；PyTorch 2.11.0，HIP 7.2.26365。
- 原容器优先搜索 DTK 自带的旧 HSA 库，PyTorch 无法枚举设备。
- 对测试进程设置 `LD_LIBRARY_PATH=/opt/hyhal/lib:/opt/hyhal/lib64:${LD_LIBRARY_PATH}` 后，实际加载 `/opt/hyhal/lib/libhsa-runtime64.so.1.11.0`，可识别 8 卡。
- 训练启动脚本会再次 source DTK 环境，实验目录中的 `dtk_compat_env.sh` 因而在 source 原 DTK 环境后重新设置库优先级，通过现有 `DTK_ENV` 参数传入。全部实验使用同一设置。
- 没有修改主机或容器的全局库配置，也没有重启、终止其他容器。

## 正确性结果

真实 GPU 专项测试：18 项通过，0 失败，0 跳过。

| 范围 | 覆盖 |
| --- | --- |
| 6 项 rotary | FP32/BF16，T=7/65/1568，真实非连续 Q/K，前向和输入梯度 |
| 12 项 ViT | cached/flash × 无重计算/uniform/block × 冻结/解冻 |
| ViT 输入 | 非方形图片及多帧，三段独立 packed 序列 |
| 比较内容 | 相同 state_dict、输入和上游梯度，merger 输出、输入梯度和参数梯度 |

首次真实 ViT 测试因夹具没有显式传入 process group 而失败。已在 `tests/unit_tests/hcu/models/test_qwen38_vision_rope.py` 的 `test_real_vision_encoder_parity` 中使用 `ProcessGroupCollection.use_mpu_process_groups()`，并传给两个模型的 `build_module`。修正后 18 项 GPU 专项测试全部通过。模型优化代码没有为此改变。

## 已抓取的 prof

实验根目录：`../checkpoints/qwen38_rope_gpu_20261111_1236`，相对于仓库根目录。

- `gpu_tests_retry.xml` / `.log`：最终 GPU 专项结果。
- `diagnostic_contended_fwd/`：前向算子 benchmark 和 9 个 Chrome trace JSON。
- `diagnostic_contended_bwd/`：前反向算子 benchmark 和 9 个 Chrome trace JSON。
- `diagnostic_summary.json`：结构计数、误差和每份 trace 的位置，明确标记 `timing_valid=false`。

每份算子 trace 都对应 27 层、Q/K 两次旋转、真实 head dim=72、local heads=8。T 分别为 1568/2048/4096；三个 T 的 kernel 数一致：

| 执行范围 | reference | cached | flash |
| --- | ---: | ---: | ---: |
| 27 层 Q/K 前向，总 kernel 数 | 648 | 488 | 56 |
| 其中 cos/sin kernel | 108 | 2 | 2 |
| 其中 fused rotary kernel | 0 | 0 | 54 |
| 27 层 Q/K 前反向，总 kernel 数 | 1134 | 1028 | 110 |
| 前反向 fused rotary kernel | 0 | 0 | 108 |

融合路径的前向总 kernel 数降低 91.36%，前反向总 kernel 数降低 90.30%。这证明融合和共享表在真实 GPU 路径上实际执行，不等同于相同百分比的耗时降低。

该算子 benchmark 的三个 T 上，cached/flash 相对参考的输出及输入梯度误差记录均为 0。这个结果针对当前随机 BF16 算子输入；完整 ViT 通过设定的容差比较，不将其解释为所有输入均 bitwise 一致。

## 为什么暂不报告加速比例

诊断采集期间，`yujihai_dreamzero` 容器的一组 8 卡 Python 作业在运行，每卡约占 43 GiB，利用率随时间变化。不同 shape/后端在不同时间执行，计时存在资源竞争偏差，因此这些延迟不能作为干净的 A/B 数据，也不能与旧日志的 step 时间直接比较。

当前判断仅限于：真实 GPU 正确性通过，重复 cos/sin 已消除，前反向 kernel 数显著减少。是否提高完整训练吞吐，仍需在相同输入、配置及资源条件下测量。

## 已排队的正式实验

实验目录的 `run_experiment.py` 顺序执行：

1. 等待 8 卡连续四次空闲，采样间隔 10 秒；检查显存、利用率和外部 GPU 进程。
2. 无竞争的前向及前反向算子 benchmark，并另行导出 profile。
3. 冻结/解冻视觉 × reference/cached/flash × 三轮训练，每组 30 步；后端顺序轮换，降低时间漂移影响。
4. 每种视觉训练状态和后端额外运行 8 步 profile，抓取第 5 步、rank 0 和 rank 4；不使用这些运行的 step 时间估算收益。
5. 每个运行前后校验输入及关键源码 SHA256，运行中监控外部 GPU 作业；失败或出现竞争污染时停止，不将其纳入有效计时。

固定 TP2/PP4/CP1、GBS64/MBS1、SEQ4096，真实 mantis-birds 数据，16 层语言主干和完整 27 层 ViT。正式训练稳态窗口计划使用第 11～30 步。

`state.json`、`runner.log` 和 `idle_monitor.jsonl` 给出当前阶段和资源证据。训练正式数据将进入 `timing/`，训练 prof 将进入 `profile/<frozen|trainable>/<backend>/r1/torch_profile/`。在这些运行完成前，本报告没有端到端性能结论。
