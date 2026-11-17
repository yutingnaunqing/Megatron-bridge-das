# Qwen3.8-27B 纯文本全参 SFT 方案核查

核查日期：2026-11-13。依据用户提供的 2026-11-12 方案、本地模型配置、safetensors header 和仓库源码。本文是实施前评审，未实现训练入口、未转换权重、未执行 GPU 训练。

## 1. 结论

方案主线合理：先在单节点减层验证真实预训练语言权重的全参 SFT，再在双节点运行完整 64 层。建议保留“离线生成独立纯文本 HF checkpoint → Qwen35Bridge → GPTModel → GPT SFT dataset → gpt_step → finetune”的路线。

纯文本训练应直接不构建视觉模块，因此不需要逐项删除现有 VL forward。冻结视觉或仅不提供图片都不能保证不加载视觉权重。“全参”在此指保留的全部语言参数参与更新；视觉和 MTP 不属于训练范围。

原方案需要修正/补强的主要内容：权重已存在；实际工作树基线不同；distributed optimizer 的显存估算应使用本地实现的公式；thinking 参数没有现成的通用透传；恢复训练要明确 finetune 状态；完整模型要有独立的重计算配置。

## 2. 本次工作树

| 项目 | 实际值 |
| --- | --- |
| 新工作树 | `/public/home/yugeng/work/Megatron-bridge-das-qwen38-sft-review` |
| 分支 | `feat/qwen38-27b-text-sft-review` |
| 主仓库起点 | `d815971`，来自原仓库当前 HEAD；不是原方案的 `eb51a5e` |
| Megatron-Bridge | 独立嵌套 detached worktree，`51885cf132b2814188b6855c25a8588254274c2a` |
| Megatron-LM | 独立嵌套 detached worktree，`cd4afffa648426a959dc7cb1e24b5ce7d0c3ff54` |

原仓库存在未提交修改和未跟踪文件，本次未复制。原仓库性能报告仅作为只读参考；后续不能假定这些启动脚本调整已进入新工作树。两个依赖目录均为独立 checkout，避免 `env/env.sh` 复制 recipe 注册文件时污染原依赖工作树。后续若修改依赖代码，应先在对应依赖工作树建立分支。

## 3. 权重现状：原方案的缺失结论已过时

模型目录：`/public/home/yugeng/work/models/Qwen3.8-27B`。

本次只读取索引、18 个 safetensors 的 header 及文件长度，未加载 tensor 数据体：索引引用的 18 个分片均存在；1199 个索引 key 均出现在指定分片 header；文件长度均与 header 声明的数据结束位置一致。尚未执行完整文件哈希或 tensor 数值校验，因此这不是数值完整性的证明。

| 权重组 | tensor 数 | 参数数 | tensor 字节数 |
| --- | ---: | ---: | ---: |
| `model.language_model.*` | 850 | 25,624,600,064 | 51,249,200,128 |
| `lm_head.weight` | 1 | 1,271,398,400 | 2,542,796,800 |
| `model.visual.*` | 333 | 460,730,096 | 921,460,192 |
| `mtp.*` | 15 | 424,699,392 | 849,398,784 |

纯语言完整模型共 26,895,998,464 参数，BF16 tensor 约 50.10 GiB。视觉仅约 4.61 亿参数，MTP 约 4.25 亿参数；两者合计权重约 1.65 GiB。去除它们可以减少额外参数、状态和相关计算，但不会让 27B 语言模型变成小模型。

模型虽然命名 Qwen3.8，实际配置的架构是 `Qwen3_5ForConditionalGeneration`，语言部分是 dense `qwen3_5_text`。64 层按 3 GDN + 1 full attention 循环，hidden=5120、FFN=17408、词表=248320，embedding/head 不共享。原方案据此选 Qwen35Bridge 是正确的，不能改用纯 attention 的 Qwen3 recipe，也不能继承现有 Qwen3.8-2.4T MoE 配置。

## 4. 无视觉权重的实施路径

保留原方案的独立转换工具，并增加以下明确约束：

1. 从 `text_config` 生成顶层 `Qwen3_5ForCausalLM` / `qwen3_5_text` 配置，保留宽度、GDN、RoPE、词表和 embedding 共享设置。关闭 `mtp_num_hidden_layers`，provider 再设置 `mtp_num_layers=None`。
2. 首先只生成前 16 层版本，需要定位故障时再生成前 4 层版本，最后生成 64 层版本。不必一开始就导出三份大权重。
3. 将实际 tensor key 的 `model.language_model.` 前缀改为 `model.`，保留 `lm_head.weight`；过滤视觉、MTP 和层号超出范围的 tensor。改配置或只改索引不能替代 tensor key 转换。
4. 依据索引逐 tensor 读取，受控分片写出。应在临时目录完成后再发布最终输出目录；拒绝覆盖源目录、拒绝复用非空输出目录，未知 key 报错。避免留下看似完整的半成品。
5. 同步截断 `num_hidden_layers`、`layer_types`；核对 `full_attention_interval` 和 provider 的 `linear_attention_freq`。仅改 provider 层数不足以保证加载和导出一致。
6. 完整保留 tokenizer、词表、special token ID、chat template。不复制图像/视频 processor；数据检查拒绝图片、视频内容和依赖视觉上下文的样本。
7. 输出转换 manifest，记录输入版本/校验信息、保留层号、key/shape/dtype、过滤原因；输出 key 集合应是完整保留语言参数集合。对 head、embedding、GDN、attention、MLP、norm 做抽样数值校验。

`Qwen35Bridge` 已注册 CausalLM，dense 映射默认读取 `model.*`，provider 明确设置普通 `rope`。纯文本前向应使用文本位置 IDs，保留 theta=10000000 和 rotary fraction=0.25；不能因去掉视觉而删除语言旋转位置编码。转换后的 HF 配置和训练 provider 都应检查。

首次 HF 初始化会在 `training/checkpointing.py::_load_hf_pretrained_checkpoint` 中重新构建 AutoBridge。因此，原方案强调“provider 纯文本化”和“初始化权重目录纯文本化”同时完成是必要的。构建 provider 时 `load_weights=False`，不影响稍后由 `checkpoint.pretrained_checkpoint` 导入真实权重。

直接从原 VL 分片选择语言 tensor 的自定义 Bridge 可以作为后续节省磁盘的选项；第一轮没有必要引入注册、恢复和导出三个方向的额外改动。

## 5. 单节点先减层：建议配置与显存边界

| 阶段 | 语言层数 / 卡数 | TP / PP / CP / DP | SEQ / MBS / GBS | 验收 |
| --- | --- | --- | --- | --- |
| 配置、权重及数据检查 | 16 / 无训练 | 检查目标拓扑 | 1024 / 1 / 8 | key、布局、token/mask、最终配置 |
| 故障诊断，按需 | 4 / 8 | 2 / 4 / 1 / 1 | 1024 / 1 / 8 | 各 stage 类型、权重导入和前反向 |
| 单节点主验收 | 16 / 8 | 2 / 4 / 1 / 1 | 1024 / 1 / 8 | 20–50 步、小数据下降、保存及恢复 |
| 单节点序列扩展 | 16 / 8 | 2 / 4 / 1 / 1 | 2048→4096 / 1 / 8→32 | 单独测峰值，逐项扩大 |
| 双节点完整模型 | 64 / 16 | 2 / 4 / 1 / 2 | 1024 起 / 1 / 32 | 真实 64 层初始化，独立训练目录 |
| 双节点备选 | 64 / 16 | 2 / 8 / 1 / 1 | 1024 起 / 1 / 16或32 | 实测 stage 峰值及流水线效率 |

16 层保留四个完整 hybrid block；PP4 每 stage 四层，均包含 GDN 和 attention，适合作为主要验证对象。4 层 PP4 的每 stage 仅一层，不同 stage 的计算类型不同，适合定位问题，但不是更均衡的验收拓扑。无论多少层都保留原始宽度、head 和 embedding。

依据 header，4/16/64 层（均含 embedding、final norm、head）的参数数分别为 4,064,876,704 / 8,631,101,056 / 26,895,998,464。16 层仍是约 8.63B，减层不会缩小词表矩阵；embedding 和 head 各约 1.27B。

本地 Megatron-LM 的 `docs/user-guide/features/dist_optimizer.md` 明确：常规 BF16 参数、FP32 main gradients、FP32 master parameters 和 Adam 状态，distributed optimizer 理论占用是 **`6 + 12/DP` bytes/parameter**。DP1 是 18 bytes，DP2 是 12 bytes。归约后的梯度供 optimizer 分片使用，不代表完整 main gradient buffer 的驻留也按 DP 等比分摊；原方案关于梯度显存分摊的表述需要修正。

以下按普通均匀 PP4、TP2 粗估每卡模型/梯度/优化器状态，忽略 TP 复制参数、buffer 对齐等细节，**不包含 activation、logits、通信、加载和 checkpoint 临时分配**：

| 配置 | 首尾 stage 每卡 | 中间 stage 每卡 |
| --- | ---: | ---: |
| 16 层，TP2/PP4/DP1 | 约 23.42 GiB | 约 12.76 GiB |
| 64 层，TP2/PP4/DP1 | 约 61.69 GiB | 约 51.03 GiB |
| 64 层，TP2/PP4/DP2 | 约 41.13 GiB | 约 34.02 GiB |

原仓库单节点历史报告记录每卡约 64 GiB。因此 64 层在 TP2/PP4/DP1 下首尾 stage 基本没有 activation 余量，优先双节点有明确理由。去视觉并不能解决主要语言状态占用。不同拓扑、offload 等可能使单节点完整训练成为可能，但不是本轮首要路径。

16 层短序列可先不重计算；64 层提供独立的 `recompute_granularity=full`、`recompute_method=uniform`、`recompute_num_layers=1` 起点并实测。不要沿用减层配置的关闭重计算设置。先关闭 FP8、graph、通信 overlap 和 offload，保留 SP 及已验证的 ROCm fusion 设置，之后每次只改变一个因素。

## 6. 数据链路需要补齐的细节

原方案选择 `GPTSFTDatasetConfig` + `ChatSFTPreprocessingConfig(loss_mode="assistant")` 正确，当前源码接受 messages/conversation/conversations。不要再通过 `--dataset local-jsonl` 将它覆盖为 prompt-completion preset。替换 `_sft_common()` 的默认 SQuAD 数据配置，同时明确关闭 offline packing 和 in-batch packing，并设固定 padding、CP1、MBS1。

本地 chat template **没有 `{% generation %}` 标记**，不能依赖 HF 返回的 assistant mask。当前 `data/conversation_processing.py` 已有基于角色 token 边界的 fallback，需用真实 tokenizer 验证，不能仅断言配置中 `loss_mode=assistant` 就认为监督正确。验收至少涵盖单轮、多轮、空回答、长 prompt 被截断、真实 reasoning 和空 think wrapper；截断后无监督 token 的样本应过滤/重采样。

另一个具体缺口是 thinking 参数：模板默认 `enable_thinking=true`、`reasoning_effort=xhigh`，可能自动插入 reasoning 指令；当前 `chat_template_kwargs_from_example()` 仅提取 `tools`，**在 JSONL 增加 `enable_thinking=false` 并不会由这个函数透传**。实施前应确定数据和推理模板策略。第一轮可保留模板默认语义并检查实际渲染；若要非 thinking SFT，需要显式实现模板参数注入并测试，不能仅在方案里写“开关保持一致”。不要直接添加一个数据配置类不存在的字段。

EOS=`<|im_end|>`、PAD=`<|endoftext|>`。不要删除视觉 token，也不要为了清理词表重排 ID。检查 collator 的 next-token labels/mask 对齐；不需要手工再次 shift。抽样解码后应看到 system/user/padding 不计 loss，assistant 内容及预期结束符计 loss，并记录每条有效监督 token 数。

GDN 源码已有 packed metadata 分支，但这不足以证明当前 ROCm kernel、conv 和状态边界都能隔离。原方案先关闭 packing 是正确的；后续需要“样本 B 单独运行”与“A+B packed 时 B 的输出”一致性检查，再开启 packing/CP。

## 7. 初始化和恢复必须分开

| 项目 | 第一次 HF 初始化 SFT | 同模型继续训练 |
| --- | --- | --- |
| `checkpoint.pretrained_checkpoint` | 对应层数纯文本 HF 目录 | 可清空，避免意外回退 HF 初始化 |
| `checkpoint.load` | `None` | 存在且完整的 Megatron checkpoint 路径 |
| `checkpoint.finetune` | HF 预训练分支会设置为 true | **显式 false** |
| `checkpoint.load_optim/load_rng` | HF 本身不提供这些状态 | **显式 true** |
| `checkpoint.save_optim/save_rng` | true | true |
| `checkpoint.save` | 独立输出目录 | 对应实验继续保存目录 |

训练 API 的 `finetune()` 和 `checkpoint.finetune` 布尔值不是同一层概念。后者在 checkpoint 加载代码中控制是否重置 step、跳过 optimizer/RNG 恢复；从首次 HF 运行导出的 YAML 恢复时尤其要检查。缺失恢复路径要直接失败，不应悄悄退回 pretrained checkpoint 重启。

保存/恢复测试除 step、scheduler、optimizer、RNG 和 consumed samples 外，还应与不中断的短跑比较恢复后数步的 loss/参数差异。若要求逐样本复现，需另核对该 dataloader 的采样与 worker 状态，不能仅用 consumed samples 相等证明数据完全一致。

16 层产物只能用于16层实验恢复，不能给64层训练 resume。迁移双节点64层时应从64层纯文本预训练权重重新初始化；单节点减层验证只能证明链路，不能证明完整模型的 loss/质量。

## 8. 实施范围与顺序

第一步新增 `tools/checkpoint/convert_qwen38_vl_to_text.py` 及微型 safetensors 测试，生成16层目录，验证 key、shape、tensor 数值和配置。不修改源模型目录。

第二步新增 `hcu_megatron_bridge/src/hcu_megatron/bridge/recipes/qwen38/bw1000/qwen38_sft.py`，按照现有 recipe 组织方式导出。核对 `qwen38/{__init__.py,qwen38.py,bw1000/__init__.py}` 和 HCU 顶层 recipe 注册；新增测试核查纯语言 provider、全参、hybrid 布局、无 packing 和 checkpoint 行为。

第三步新增 `examples/qwen/train_qwen38_text_sft.sh` 和单节点16层包装脚本。分别支持 HF 初始化和 resume；导出最终 YAML；显式选择 `--mode sft --step_func gpt_step`；校验拓扑和 batch 整除、远端路径及导入来源。沿用现有环境，不因上游 skill 的 NVIDIA 示例重建或升级 ROCm 环境。

第四步验证单节点16层20–50步和保存恢复，再转换64层并跑双节点。短跑显式设置 warmup/decay，避免 `_sft_common()` 默认50步warmup超过总步数。先检验数值和显存，再做吞吐调优。

第一轮不需要改 VL forward、现有 VL recipe 或全局 Bridge 注册。只有真实语言权重导入/前向失败并定位架构差异时，再做最小修复。

## 9. 源码证据和本次变更

相对于新工作树的主要核查位置：

- `3rdparty/Megatron-Bridge/src/megatron/bridge/models/qwen/qwen35_bridge.py`：CausalLM 注册、dense mappings、纯文本 rope provider。
- `3rdparty/Megatron-Bridge/src/megatron/bridge/recipes/common.py::_sft_common`：默认 SQuAD、packing、warmup、checkpoint 配置。
- `3rdparty/Megatron-Bridge/src/megatron/bridge/training/checkpointing.py::load_checkpoint/_load_hf_pretrained_checkpoint`：预训练回退、重新构建 Bridge、恢复逻辑。
- `3rdparty/Megatron-Bridge/src/megatron/bridge/data/conversation_processing.py`：模板参数、assistant 边界 fallback。
- `3rdparty/Megatron-Bridge/src/megatron/bridge/data/datasets/gpt_sft.py`：chat dataset、collator 的 labels/mask shift。
- `3rdparty/Megatron-Bridge/3rdparty/Megatron-LM/docs/user-guide/features/dist_optimizer.md`：`6+12/DP` 公式。
- `hcu_megatron_bridge/src/hcu_megatron/bridge/recipes/qwen38_vl/bw1000/qwen38_vl.py`、`env/env.sh`：既有配置和工作树路径行为。

本次新增文件只有本文档。未修改训练代码、未转换或下载权重、未运行训练、未执行 Git 提交。后续代码修改应逐文件列明位置；Git 提交须先取得用户同意。
