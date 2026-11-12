<!-- Copyright (c) 2026 Hygon Information Technology Co., Ltd. -->

# Qwen3.8-27B 视觉 RoPE 优化方案与验证记录

日期：2026-11-11。当前结论：CPU 与真实 GPU 正确性检查通过；按用户最终要求完成单轮端到端粗测，第 10～20 步平均耗时降低约 2.39%，第 5 步训练 prof 已抓取。存在外部 GPU 进程，仅作方向性观测。详见 [单轮粗测结果](qwen38-vision-rope-quick-20261111.md)。原三轮测试矩阵已停止，下面的多轮验收属于原始方案，不再继续执行。

## 工作树与基准

- 主仓库工作树：`../Megatron-bridge-das-perf-rope`，分支 `perf/qwen38-vision-rope`。
- 主仓库基准：`eb51a5e`；Bridge 基准：`51885cf132b2814188b6855c25a8588254274c2a`。
- Bridge 使用独立 worktree；MCore 使用相同提交 `cd4afffa648426a959dc7cb1e24b5ce7d0c3ff54` 的本地共享对象检出，代码未修改。
- 从原目录继承 Bridge 的 `recipes/__init__.py` 注册改动。这是原有 HCU recipe 注册，不属于本次 RoPE 优化。
- 原工作目录的代码和两份未跟踪性能报告均保留。工作树不自动携带未提交更改；继承补丁与原状态已另外保存到本次实验目录。
- 本次未提交 Git commit；主仓库和 Bridge 中的新增文件、修改文件均需分别查看和提交。

## 方案及实施顺序

1. 保留 `reference`，建立可切换基线。
2. 在视觉模型入口，根据真实 packed token 顺序生成一份 FP32 cos/sin。
3. 以 `cached` 模式将这份表传给全部 27 个视觉层，先单独验证复用的正确性。
4. 以 `flash` 模式复用 FlashAttention Triton rotary kernel，Q、K 各一次 out-of-place 调用。
5. 将表显式传入普通 forward 和 uniform/block recompute 的 checkpoint 输入，反向重计算继续复用存储。
6. 对实际 MCore 参考函数比较输出、输入梯度；检查 token 顺序、microbatch 生命周期和分帧 metadata。
7. 在真实设备上执行算子、完整 ViT 和冻结/解冻视觉训练 A/B，排除 warmup/capture/profiler 时间后报告收益。

本轮不改变语言 M-RoPE、attention 的 `cu_seqlens`、图片/帧边界、patch embedding 或 CUDA Graph。优化路径与视觉 Graph 的组合显式报错，避免进入尚未完成适配的捕获路径。

## 三档实现

| backend | 角度表 | 旋转 | 默认行为 |
| --- | --- | --- | --- |
| `reference` | 原有逐层 Q/K 各自算 cos/sin | 原实现 | 默认值，保持基线 |
| `cached` | 每视觉 microbatch 生成一次 | PyTorch FP32 半分旋转 | 单独测表复用 |
| `flash` | 每视觉 microbatch 生成一次 | FA2 Triton fused rotary + 自定义 autograd 适配 | 目标优化路径 |

视觉 head dim 为 72；cos/sin 各为 `[T,36]`。H/W 频率保持原排列；rotary 按前后半段配对，kernel 的 `INTERLEAVED=False`。Q/K 输入可以是非连续 view。表不跨 microbatch 缓存。

目标容器的 FlashAttention 高层 `apply_rotary_emb` 要求 Q/K 和 cos/sin 同 dtype。直接调用会阻止 BF16 Q/K + FP32 表，或迫使每层额外转换整份 Q/K。本实现复用其底层 `rotary_kernel`：BF16 输入与 FP32 表在寄存器中转 FP32 运算，输出转回原 dtype；没有修改 FlashAttention 或新增运行时依赖。

适配层检查 FA2 kernel ABI，不支持的版本明确报错，不静默 fallback。`enable_fp_fusion=False` 保持参考实现分离的 FP32 乘加；backward 使用共轭旋转。表必须不需要梯度。

结构上，视觉旋转从每层 Q/K 合计 24 次相关 launch，变为 2 次 rotary launch；另有每个 microbatch 两次共享表三角函数计算。这是调用结构目标，尚未通过真实 GPU profile 实测，也不代表完整 step 加速 12 倍。

## 修改位置

以下路径均相对于本工作树根目录；Bridge 内文件相对于 `3rdparty/Megatron-Bridge`。

| 文件 / 搜索入口 | 改动 |
| --- | --- |
| `src/megatron/bridge/models/qwen_vl/modelling_qwen3_vl/vision_rope.py` / `prepare_vision_rotary_cos_sin`、`_launch_flash_rotary`、`apply_vision_rotary_pos_emb` | 新增表准备、kernel ABI 检查、混合精度前反向适配和 cached 实现 |
| `src/megatron/bridge/models/qwen_vl/modelling_qwen3_vl/vision_model.py` / `vision_rope_backend` | 构造时校验 backend；每 microbatch 准备表并传给 decoder |
| `src/megatron/bridge/models/qwen_vl/modelling_qwen3_vl/transformer_block.py` / `Qwen3VLVisionTransformerBlock._checkpointed_forward` | 将 cos/sin 传入 checkpoint 与重计算层；普通路径复用已有传参接口 |
| `src/megatron/bridge/models/qwen_vl/modelling_qwen3_vl/attention.py` / `rotary_pos_cos` | 接收共享表并应用旋转，原始角度路径继续调用原 absolute RoPE |
| `src/megatron/bridge/models/qwen_vl/modelling_qwen3_vl/transformer_config.py` / `vision_rope_backend` | 声明视觉配置字段并从 provider 传递 |
| `src/megatron/bridge/models/qwen_vl/qwen35_vl_provider.py` / `vision_rope_backend` | Dense/MoE provider 声明字段；当前 Qwen3.8-27B 使用 Dense provider |
| `hcu_megatron_bridge/src/hcu_megatron/bridge/recipes/qwen38_vl/bw1000/qwen38_vl.py` / `VISION_ROPE_BACKEND` | 读取环境变量，保留 CLI 配置覆盖 |
| `examples/qwen/train_qwen38_vl_pretrain.sh` / `VISION_ROPE_BACKEND` | 导出、校验 backend 环境变量 |
| `tests/unit_tests/hcu/models/test_qwen38_vision_rope.py` | CPU 对照、checkpoint/attention/入口传递、interpreter 及真实 GPU 算子/ViT 测试 |
| `examples/qwen/benchmark_vision_rope.py` | 对照输出/梯度，测 27 层旋转 microbatch 的时间、显存、kernel 数 |
| `examples/qwen/run_qwen38_rope_ab.sh` | 顺序执行冻结/解冻 × 三 backend × 多次重复的训练矩阵 |

## 当前验证与限制

- 最终常规测试：58 项通过、20 项跳过（18 项需要真实 GPU，2 项 interpreter 检查单独运行）；独立 interpreter 检查 2 项通过。目标文件 lint、shell 语法和 Git whitespace 检查通过。
- CPU 测试使用本工作树的实际算子和实际 MCore 参考函数，覆盖 FP32/BF16/FP16、72 维、非连续输入、输出和输入梯度。
- 通过实际 attention/vision/checkpoint 方法配合小型 CPU doubles，检查原始角度路径、packed metadata、27 层表复用、跨 microbatch 更新，以及普通/TE checkpoint 两个调用分支。它们不替代真实 TE 集成测试或 FP8 数值测试。
- 实际 FlashAttention kernel 的 FP32 CPU interpreter 前向、反向检查通过。
- DTK interpreter 的 BF16 转换使用截断，与真实编译路径的 round-to-nearest-even 不一致，因此没有将 BF16 interpreter 当作精度通过依据。
- BF16/FP32 × 前向/反向共四个 kernel 已针对 `hcu:gfx936:64` 离线编译成功，产生 HSACO。BF16 LLVM IR 包含 FP32 分离乘加和 RNE 存储转换；离线编译不证明真实设备执行正确或有加速。
- 已加入真实设备测试：6 项算子前反向，12 项真实 ViT（两个 backend × 三种 recompute × 冻结/解冻）。ViT 测试使用非方形图片和多帧，比较 merger 输出、输入及参数梯度。
- 上一轮 `yg_qwen38` 中 PyTorch 枚举到 0 张设备。2026-11-11 通过测试进程局部的 `LD_LIBRARY_PATH`，优先加载主机的 `libhsa-runtime64.so.1.11.0` 后，已恢复识别 8 张设备；未重启容器、未修改全局环境。
- 后续真实 GPU 测试的 6 项 rotary 和 12 项 ViT 检查全部通过。算子诊断 profile 已核实前向 kernel 数 648/488/56；当前另一容器占用节点，诊断延迟不能作为正式加速数据，完整训练计时和训练 prof 尚未开始。
- 测试工具 uv/ruff 安装到实验目录，不修改容器已安装的训练依赖。

## 复现命令

进入训练容器并切换到本工作树根目录。使用当前容器 Python；这里的 `uv run --no-project` 避免触发上游依赖同步或重建 TE。

```bash
source env/env.sh

# CPU 测试 + 设备可用时的真实 GPU 测试
uv run --no-project --python /usr/local/bin/python python -m pytest \
  tests/unit_tests/hcu/models/test_qwen38_vision_rope.py \
  -q -o 'markers=unit: isolated unit checks'

# 独立 interpreter 检查；不能据此估算 GPU 性能
TRITON_INTERPRET=1 uv run --no-project --python /usr/local/bin/python python -m pytest \
  tests/unit_tests/hcu/models/test_qwen38_vision_rope.py \
  -k cpu_interpreter -q -o 'markers=unit: isolated unit checks'

# GPU 算子前向 benchmark：包含每 microbatch 的表准备
python examples/qwen/benchmark_vision_rope.py --output runs/rope_fwd --profile

# GPU 算子前反向 benchmark
python examples/qwen/benchmark_vision_rope.py --output runs/rope_fwd_bwd --backward --profile

# 单组训练；backend 可改为 reference/cached
VISION_ROPE_BACKEND=flash bash examples/qwen/run1_qwen38_vl_single_node.sh train

# 完整训练矩阵；默认 40 步、每组 3 次，输出目录 runs/qwen38_rope_<timestamp>
TRAIN_ITERS=40 REPEATS=3 bash examples/qwen/run_qwen38_rope_ab.sh
```

也可以用 `model.vision_rope_backend=flash` 显式覆盖 recipe 环境变量。单组脚本保留原默认的冻结视觉；矩阵脚本分别覆盖为冻结和解冻，并在启动前检查设备、导入位置和显存占用。

benchmark 中 `gpu_elapsed_ms` 是 GPU event 包围整段 microbatch 的 elapsed 时间，可能包含 host 供给不足时的空隙，不是 kernel duration 的简单相加；同时记录 wall time。`--profile` 单独采集 trace，不进入计时样本。

## 后续性能验收

1. 先运行真实 GPU 单测，再运行算子 benchmark；错误时停止，不忽略失败或切换 fallback。
2. 训练对照固定 TP2/PP4、GBS64/MBS1、SEQ4096、数据集、seed 和其他 fusion/overlap 设置。
3. 每组运行 30～50 步并重复至少三次；使用未启用 profiler 的稳态窗口比较 step mean、P50/P95、samples/s、显存峰值。
4. 精度比较使用相同权重、输入和上游梯度。真实 GPU 算子/ViT 测试已执行 state_dict 对齐；不能仅凭两次独立训练的最终 loss 接近判定正确。
5. 单独短 profile 核实共享表次数和 rotary kernel 数，保持 attention 分帧边界；结合全 step 判断 rank 0 加速是否减少下游等待。
6. 若双 kernel 路径正确且有收益，再评估 Q/K 合并一次 launch；当前不引入该第二阶段优化。
