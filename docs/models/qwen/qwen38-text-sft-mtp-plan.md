# Qwen3.8-27B 文本 SFT 的 MTP 测试方案

> 2026-11-16存储更新：旧runs内测试checkpoint及参数比较副本已按用户要求清理，历史日志/曲线/审计JSON保留；本文涉及旧checkpoint的步骤仅作历史方案，不再可直接恢复。当前启动与状态见[handoff](qwen38-text-sft-handoff.md)。

更新日期：2026-11-15 02:20。完整64层MTP的n05+n07、TP4/PP2/DP2四阶段IB短测已通过：匹配无MTP基线、MTP初训保存、恢复、连续对照全部通过。TP4/PP4/DP1原embedding配对仍失败，范围限制及证据见[handoff最终结果](qwen38-text-sft-handoff.md)。

## 实施与验收状态

| 阶段 | 当前状态 |
| --- | --- |
| 固定版本源码支持分析、mask缺口定位、显存与拓扑方案 | 已完成分析和方案 |
| 无MTP的TP2/PP4/DP2 IB训练、保存及同节点恢复对照 | 已通过；见[handoff](qwen38-text-sft-handoff.md) |
| 保留15个MTP tensor的独立权重转换与导入验证 | 转换/hash及真实16rank HF导入通过，8个MTP rank抽样匹配 |
| MTP recipe/manifest模式接入、assistant mask传递及shift边界测试 | 已实施；真实forward/梯度/tokenizer定向验证通过 |
| TP4/PP4/DP1的无MTP匹配对照 | 23:18已通过8步训练及checkpoint审计 |
| 启用MTP后的GPU训练、梯度/参数更新、保存恢复及不中断对照 | 完整64层TP4/PP2/DP2四阶段通过；全部MTP参数和545个抽样语言tensor恢复/连续一致 |
| 生成效果、部署导出及speculative decoding评估 | 尚未执行 |

此前验收的TP2/PP4无MTP IB实验两阶段均为`mtp_num_layers: null`。配置中的默认`mtp_loss_scaling_factor: 0.1`不代表启用了MTP，目前已完成显式MTP1的完整64层TP4/PP2/DP2 GPU短测，下文保留原测试设计；实际拓扑调整、通过证据及未测范围见[handoff最终结果](qwen38-text-sft-handoff.md)。

## 为什么默认关闭 MTP

MTP 是语言模型的辅助预测模块，不是视觉模块。默认关闭它是为了先验收标准 next-token、assistant-only 全参 SFT，减少附加损失、流水线尾部显存和权重转换的变量，并非纯文本模型必须删除 MTP。

普通 SFT 不要求开启 MTP。NVIDIA 的 [MTP 文档](https://docs.nvidia.com/nemo/megatron-bridge/latest/training/multi-token-prediction.html)将其主要用途定位于预训练，也说明 SFT packing 暂不支持。若部署需要使用 MTP 做 speculative decoding，联合微调 MTP 是值得单独验证的方向；仅开启训练 MTP 并不能证明推理加速或生成质量提高，仍需部署端支持和实测。

## 固定版本的实际支持与缺口

依据本工作树 Bridge `51885cf`、MCore `cd4afff`，不是假定最新上游行为：

- 原始权重有 15 个 `mtp.*` tensor，共 424,699,392 参数、849,398,784 BF16 字节。转换工具默认过滤它们，并设置 `mtp_num_hidden_layers=0`；新增显式`--mtp-num-layers 1`保留模式和对应recipe开关，默认模式仍关闭。已有 text-64layers 目录不能只改配置后当作含 MTP 的权重。
- `src/megatron/bridge/models/qwen/qwen35_bridge.py::_get_dense_mtp_mappings` 已覆盖 fc、norm、QKV、attention output 和 dense MLP。配置映射支持 `mtp_num_hidden_layers → mtp_num_layers`；这说明有接入基础，不等于已经运行验收。
- **assistant mask 缺口**：`src/megatron/bridge/training/gpt_step.py::_forward_step_common` 普通路径原先没有传递`loss_mask`。本次已在Bridge独立分支为MTP普通forward补齐，并以真实辅助梯度/shift边界测试验证；schedule-plan分支原本已显式传入。MCore 的 `process_mtp_loss()` 在 `loss_mask=None` 时使用全 1，然后 roll labels/mask。只开启 MTP 会使辅助目标不再保证 assistant-only，必须先补齐并测试。
- provider 的 `mtp_block_spec()` 会取本 stage 最后一层的 spec。本模型默认末层是 full attention，应断言实际 MTP 为标准 attention，而不是 GDN；自定义 PP 布局改变后要重新核对。
- MTP 默认位于最后一个 PP stage，并可能需要复制输入 embedding。不能仅用新增约 0.425B 参数的大小估算峰值；还要计 embedding 副本、梯度、Adam、额外 logits、通信及保存的临时分配。

上述源码路径相对于 `3rdparty/Megatron-Bridge/`；MCore 位于其 `3rdparty/Megatron-LM/`。不直接修改 MCore。

## 实施位置

| 位置 | 所需改动 |
| --- | --- |
| `tools/checkpoint/convert_qwen38_vl_to_text.py` | 新增显式可选的保留 MTP 模式，默认仍关闭；严格校验源 15 个 MTP key/shape/dtype，保留顶层 `mtp.*` 前缀、记录 SHA256，输出配置为 1 层 MTP；写入新的 `Qwen3.8-27B-text-64layers-mtp1` 目录 |
| `tools/checkpoint/check_qwen38_text_sft.py` | 根据显式 MTP 模式核对语言和 MTP 的完整 key/shape；manifest 与 resume 兼容检查增加 MTP 层数和辅助 loss 系数，禁止跨模式恢复 |
| `hcu_megatron_bridge/src/hcu_megatron/bridge/recipes/qwen38/bw1000/qwen38_sft.py` | 显式启用 1 层 MTP、系数 0.1，保持 BF16 全参、CP1/MBS1、无 packing；校验含 MTP 权重，不随机补建缺失参数 |
| Bridge `src/megatron/bridge/training/gpt_step.py` | 在该依赖建立分支后，最小修改普通 forward 路径，将现有 batch 的 `loss_mask` 传给 GPTModel；不重新 shift labels；不得因 MTP 开启改变主 LM loss 的 mask |
| `examples/qwen/qwen38_text_sft_worker.py` | 记录 MTP 类型、参数与加载证据；检查全部语言和 MTP 参数可训练、无视觉参数；抽样保留 MTP 非零有限梯度与更新证据 |
| `tests/unit_tests/qwen38_text_sft/` | 补权重转换、模式隔离、mask 透传及 shift 边界的定向测试 |

## 验收顺序

1. **CPU/真实 tokenizer**：单轮、多轮、中文、assistant 结束符、padding、窗口末尾。以实际 collator 的 `labels/loss_mask` 为起点，按本版本 MCore 的一次 roll 构造 MTP 期望监督位置；断言目标 user/system/padding 和越界位置为 0，assistant 有有效监督。用 spy 捕获模型收到的 mask，防止只测配置而漏掉调用链。
2. **权重导入**：独立新目录转换 64 层加 1 个 MTP block。应有 866 个 tensor，54,641,395,712 字节，27,320,697,856 参数；语言 tensor hash 与无 MTP 目录一致，MTP tensor hash 与源一致。核对 HF 导入无 missing/unexpected 参数及实际 MTP 数值，不能把“调用返回成功”当作完整映射证明。
3. **双节点 5 步**：无MTP的TP2/PP4/DP2 IB基线已完成，先按以下新拓扑补齐匹配对照，再启用MTP。使用再次确认空闲的 2×8 卡，通过IB通信（IB插件报错则换节点，禁止Socket回退）、TP4/PP4/DP1、SEQ1024、GBS32、MBS1、相同 UltraChat 数据和 disabled thinking。开启 1 层 MTP、loss 系数 0.1、full/uniform/1 重计算，独立 SAVE_DIR。先以同一TP4/PP4/DP1拓扑跑无MTP对照，再启用MTP；两组沿用相同的RMSNorm/Adam后端开关。当前TP2/PP4/DP2的8步训练及恢复结果可作为接入依据，效果和吞吐比较使用新匹配拓扑对照。检查 MTP 为 full attention、主 loss 和 `mtp_1 loss` 有限、MTP 梯度/参数更新、所有 rank 正常退出、至少两步后和保存时的设备显存。
4. **保存与恢复**：总调度固定为 8 步，5 步提前退出并保存，恢复 optimizer/RNG/scheduler/样本消费后跑到 8 步。checkpoint 应包含全部语言和 MTP 参数及优化器状态；与同配置不中断 8 步结果比较，检查 step 6–8 loss、LR、grad norm、consumed samples 及抽样参数。
5. **效果和部署**：在相同验证集比较无 MTP 与有 MTP 的主 LM loss 和生成结果；若目标是 speculative decoding，再核对部署端要求的 MTP key、导出完整性、接受率和实际 tokens/s。训练日志里的 teacher-forced acceptance 不能替代真实 speculative decoding 接受率。

## 显存与失败处理

无MTP的完整64层TP2/PP4/DP2实测设备采样峰值为56261MiB（约54.94GiB），每卡容量65520MiB（约63.98GiB），采样余量约9.04GiB，且10秒采样不能证明瞬时峰值。

TP2/PP4/DP2 下，MTP 独有参数粗估额外 `0.425B / TP2 × 12 bytes ≈ 2.37 GiB` 常驻状态，集中于尾 stage。另一个可能的尾 stage embedding 副本约 `1.271B / TP2 × 12 bytes ≈ 7.10 GiB`，是否分配、共享、优化器去重以实际参数/buffer 为准；两项合计约9.47GiB，已超过上述采样余量，还未加入新激活和临时分配，因此MTP首测选择TP4/PP4/DP1。该拓扑原无MTP最高常驻状态估算约30.85GiB，MTP独有参数约额外1.78GiB，可能的embedding副本约5.33GiB，仍需真实测峰值。两项不能直接当作峰值或保证可运行。预留每份 checkpoint 至少 400GB，并计入更多状态和临时空间。

如 OOM，先保留日志和每 rank 峰值，先试 SEQ512（重新准备数据），再按实际stage峰值评估其他TP/PP布局，不能只减 GBS 推断常驻显存会下降。如 MTP mask、导入或布局测试失败，先修复对应问题，再启动 GPU 实验；不可把不含 MTP 的基线 checkpoint 当成有 MTP 实验的 resume 来源。

本方案不修改既有基线权重，也不承诺 MTP 会提升微调质量。完成上述接入后，才能给出已验证可执行的 MTP 启动命令；当前入口要求显式MTP模式与checkpoint匹配，默认模式仍拒绝含MTP权重。

2026-11-15 00:53补充：n05+n10探针仍在n10 NCCL barrier失败，完整64层双节点MTP等待健康空闲节点。已启动独立4层语言模型+1层MTP的n05两卡TP2/PP1功能测试，初训5步→恢复8步→连续8步对照，证据`runs/qwen38_text_sft_mtp_smoke4l_20261115/`；该裁剪测试不能替代完整27B验收。

2026-11-15 01:14：用户指定切换n06+n07，已启动强制IB embedding探针；n10停止使用。n06已知SGLang约3.4GiB/卡静态占用保留，后续不宣称独占性能。独立4层两卡初训/恢复已通过，连续8步对照运行中，尚不替代全层验收。

2026-11-15 01:20：独立4层+MTP1两卡功能三阶段全部通过；恢复/连续日志指标、全部424,699,392个MTP参数和35个抽样语言tensor逐元素一致。`runs/qwen38_text_sft_mtp_smoke4l_20261115/result.json`明确full_64_layer_verified=false，完整64层IB训练验收仍待完成。

2026-11-15 01:33：TP4/PP4原embedding配对在n06+n07、n05+n06也超时后，已换n05+n07验证TP4/PP2/DP2对等GPU配对，16rank全部606.25MiB BF16 IB通信PASS。已启动该拓扑完整64层四阶段新验收，基线及MTP结果独立保存，不复用TP4/PP4基线。默认rank order tp-cp-ep-dp-pp使PP跨节点、DP在节点内，不能误称为同节点embedding。

最终结果：2026-11-15 02:20，`runs/qwen38_text_sft_mtp_tp4_pp2_dp2_ib_20261115/result.json` success=true、full_64_layer_verified=true。全部424,699,392个MTP参数及545个抽样语言tensor逐元素一致，恢复/连续最终validation loss均0.8456035；全部本轮进程已退出。生成效果/推理加速/长程稳定性未测试，TP4/PP4原配对失败记录保留。

2026-11-16：完整64层TP4/PP2/DP2长测已完成，先无MTP500步通过，再开启MTP500步通过，各在250保存恢复。数据16384/256，LR2e-5，训练窗口下降约3.37%、验证持续下降，无NaN/跳步/IB异常或显存漂移。首轮LR1e-5最终验证更低，不据固定降幅门槛宣称高LR或MTP效果更优。详细方案、两轮结果和范围见handoff长步数最终结果。
