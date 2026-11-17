# Qwen3.8-27B SFT handoff

更新：2026-11-17（北京时间）。本文保留当前操作入口、关键实现修改、阶段测试记录和故障结论，供后续交接。只合并重复状态和过时排期；历史测试证据与当前产物是否仍存在分别说明。

## 当前状态

- 当前有三个独立SFT入口：文本`run_qwen38_text_sft.sh`、VL `run_qwen38_vl_sft_full.sh`、16层单节点`run_qwen38_text_sft_single_node.sh`。单节点不再调用文本主脚本，也不使用MPI/hostfile；仅共用Python worker/recipe。均使用原生算子；单节点PROFILE默认开启，主文本/VL默认关闭。旧full/long/native/train包装已删除。
- 完整64层纯文本全参SFT：无MTP/MTP1均完成双节点500步，250步保存并恢复至500步，IB数据通信，无NaN/跳步和持续显存增长。
- 已验证主拓扑：n05+n07，16GPU，TP4/PP2/DP2、CP1，BF16参数、FP32梯度/master/Adam状态，distributed optimizer，full/uniform/1重计算。SEQ1024、MBS1、GBS32。
- 文本4096已在n05+n06实际运行：run01至step16；取消自动预检并复用缓存后的run02至step22，首步49.082秒、step2–22平均15.214秒，无NaN/跳步。run01的step11后退化在run02未复现，原因为历史未定位项；4K保存/恢复及长测尚未验收。run02的10个残留worker已清理；最新启动器已参照旧入口改为MPI外层tee、节点exec agent，双节点CPU正常/中断探针通过；真实GPU正常结束退出已在下述四组50步实验验证，Ctrl+C/异常退出兜底问题仍保留。8192/VL未进行GPU验收。
- 2026-11-17性能对照已完成：n05+n06、SEQ4096、native、64层无MTP、GBS64/MBS1、DP2、full重计算，TP4/PP2与TP2/PP4各比较TP overlap关闭/开启，每组50步。关闭/开启平均步耗时分别29.034/34.976秒、25.211/29.666秒，开启后吞吐分别下降16.99%/15.02%。16份prof均已核验GPU kernel及step20/21，四组正常退出无残留；详见本文末尾性能对照节。
- run03有两次启动：22:00无重计算在训练前向OOM；22:16 full重计算在模型构建时被前次n06 GPU6/7残留占用阻塞。残留PID3216623/3216624已TERM清理。新八子进程CPU探针复现MPI提前终止agent、子进程信号未转发完整的竞态；旧快速CPU退出测试不能覆盖此场景，尚未实施生产退出兜底修复。
- `text1024-TP2PP4-run01`在模型构建前JIT预热遇到Inductor缓存读取/发布错误，非OOM。文本/VL worker的Inductor缓存已按host/local-rank隔离并保留旧缓存；后续4K native/full的TP2/PP4关闭/开启两组各50步均通过，未再现该缓存错误。
- 启动现默认读取`examples/qwen/hostfile`，只需在一个节点的既有`yg_qwen38`容器中执行一次；MPI通过容器SSH端口11463在每节点启动一个torchrun父进程，各派生8个worker。无需手动启动第二节点。
- MPI日志现统一写入`SAVE_DIR/train_logs/mpi_YYYY-MM-DD-HHMM.log`，包含两节点输出；MPI模式不再写node-0/1.log，历史run01/run02的旧日志保留。单节点/显式NODE_RANK手工模式仍写node-N.log。
- 按用户最新要求，文本`dryrun/sft`已移除全部自动训练前检查，信任输入数据与参数，不再生成launch manifest；显式`preflight`仅保留作手工工具。下文历史检查保护和缓存检查方案不再代表当前自动启动行为，详见最新变更节。
- 历史测试checkpoint、optimizer状态和参数比较副本已按用户要求清理，不能再从旧runs恢复。原始HF模型、数据、日志、JSON/CSV审计、配置、曲线和编译缓存保留。

## 工作树、模型和数据

工作树：`/public/home/yugeng/work/Megatron-bridge-das-qwen38-sft-review`。使用现有ROCm容器，不在宿主机安装训练依赖。Bridge为嵌套工作树；不修改Megatron-LM源码。当前未提交Git；提交需用户授权。

| 用途 | 路径 |
| --- | --- |
| 原始完整多模态HF权重 | `/public/home/yugeng/work/models/Qwen3.8-27B` |
| 64层纯文本，无MTP | `/public/home/yugeng/work/models/Qwen3.8-27B-text-64layers` |
| 64层纯文本，MTP1 | `/public/home/yugeng/work/models/Qwen3.8-27B-text-64layers-mtp1` |
| 原始UltraChat Parquet | `/public/home/yugeng/work/datasets/HuggingfaceH4_ultrachat_200k` |
| 500步测试文本数据 | `/public/home/yugeng/work/datasets/ultrachat-qwen38-steady-train16384-val256-seq1024-disabled` |
| VL图文数据 | `/public/home/yugeng/work/datasets/mantis-birds-energon-v1` |

源模型配置为dense Qwen3.5 hybrid：64层，3 GDN + 1 full attention循环。纯文本转换去除视觉，语言参数约26.896B；MTP1额外424,699,392参数。两个文本HF目录语言tensor字节相同。模型目录及其转换manifest未清理，无需再次转换。

文本数据保持官方train_sft/test_sft边界，16384条train、256条validation，thinking disabled、assistant-only、无packing。CPU全量统计的shift后、padding前train长度86～1024，平均714.46、P50=741、P95=996，平均监督503.12token；validation平均725.58。详见[长度统计](../../../runs/qwen38_long_vl_launch_review_cpu_20261116/current_dataset_lengths.json)。

4096/8192应从原始Parquet分别prepare到新目录，以纳入长对话；旧短JSONL可复用但会大量padding。当前HF最大位置262144，4K/8K无需修改RoPE或转换权重。Birds现有Energon无需重新打包，但只有train，VL入口关闭独立validation；历史核验最长有效输入702，不能把padding到8192当成真实8K图文数据。

## 工作树来源与实现决策

| 项目 | 来源/当前状态 |
| --- | --- |
| 主仓库 | 起点`d815971`，当前分支`feat/qwen38-27b-text-sft-review` |
| Megatron-Bridge | 基于`51885cf132b2814188b6855c25a8588254274c2a`的独立嵌套工作树；当前分支`feat/qwen38-mtp-assistant-mask-20261114`，修改未提交 |
| Megatron-LM | 独立嵌套工作树，`cd4afffa648426a959dc7cb1e24b5ce7d0c3ff54`，不修改其源码 |
| 原项目 | `/public/home/yugeng/work/Megatron-bridge-das`仅作参考，未把原工作树全部未提交修改复制进来 |

实施前评审见[2026-11-13方案核查](qwen38-27b-text-sft-review-20261113.md)。纯文本采用“独立纯文本HF权重→Qwen35Bridge→GPTModel→GPTSFTDataset→gpt_step”，直接不构建视觉塔。仅冻结视觉或不提供图片不能替代纯文本化。完整VL则保留原始ConditionalGeneration配置和视觉processor，使用单独的图文入口，不与文本入口混用。

`env/env.sh`按项目既有方式把HCU recipe注册文件复制到本工作树的Bridge目录，故该注册文件可能显示修改；不要把依赖软链接回原工作树。用户在env脚本中的已有注释保留。以下记录中的模型加载、数值恢复、IB验收、效果和性能分别判断，不能互相替代。

## 关键实现修改及回归依据

### 权重转换、数据和全参recipe

| 修改位置 | 解决的问题与当前行为 | 验证/交接注意 |
| --- | --- | --- |
| [转换工具](../../../tools/checkpoint/convert_qwen38_vl_to_text.py)的`plan_conversion/convert` | 从原VL的`text_config`生成`Qwen3_5ForCausalLM`/`qwen3_5_text`，将`model.language_model.*`改为`model.*`；支持4/16/64层完整hybrid block；去掉视觉，MTP按模式保留 | 按真实key/shape/dtype核验，流式复制tensor字节并记录SHA256，临时目录完成后发布，拒绝覆盖源/既有目录；不是只改配置或索引 |
| 同一转换工具及MTP模式 | 默认去掉MTP；显式MTP1保留全部15个源tensor，不随机重建MTP | 64层无MTP为851 tensor、53,791,996,928字节；MTP1为866 tensor、54,641,395,712字节；语言tensor字节一致。[MTP转换审计](../../../runs/qwen38_text_sft_mtp_20261114/converted_weight_audit.json)仍保留 |
| [通用数据准备](../../../data_preparation/prepare_qwen38_text_sft.py)的`messages_row/make_token_checker` | 保留messages结构，拒绝依赖视觉的样本和把原始RL reward当SFT答案；显式绑定thinking模板 | 使用训练同一tokenizer、assistant mask和collator检查完整对话长度及有效监督；DAPO等数据须先明确真实解答字段，不能直接套UltraChat流程 |
| [UltraChat准备](../../../data_preparation/prepare_ultrachat_sft.py)的`split_files/prepare` | 读取官方完整Parquet分片和原生messages list，保持train_sft/test_sft边界，按首个user文本去重及排除跨split重叠 | 不随机重分官方测试集；超长完整对话过滤；manifest记录源分片、过滤数和token设置。[官方数据核验](../../../runs/qwen38_ultrachat_parquet_20261113/official_verification.json)保留 |
| [文本recipe](../../../hcu_megatron_bridge/src/hcu_megatron/bridge/recipes/qwen38/bw1000/qwen38_sft.py)及同目录导出 | 从转换后的文本HF构建dense provider，所有保留语言参数可训练，无视觉/PEFT；BF16参数、FP32梯度及Adam状态；64层默认full/uniform/1重计算 | CP1/MBS1，无packing，assistant-only；不能拿16层结果声明完整27B通过。`checkpoint.pretrained_checkpoint`负责首次真实HF导入，provider的`load_weights=False`不表示随机初始化 |
| [手工检查器](../../../tools/checkpoint/check_qwen38_text_sft.py)的`check_environment` | 显式preflight时核对模型key/shape、层数、拓扑、数据schema、MTP模式/系数及恢复manifest；文本dryrun/sft已按用户要求取消自动调用 | 新运行不再生成该manifest，手工检查器不能凭其恢复检查验证新运行；训练本身按RESUME_FROM交给框架恢复optimizer/RNG/scheduler |
| [显存估算](../../../tools/checkpoint/estimate_qwen38_text_sft_memory.py) | 按实际tensor及TP复制参数估算状态占用，distributed optimizer常驻状态采用`6+12/DP`字节/参数 | DP2不是把全部常驻状态减半；未计激活、通信和临时内存，不能替代实测。完整模型单节点无offload仍未验收 |

16层转换保留215 tensor、17,262,202,112字节，约8.63B参数；宽度、词表、embedding/head均保留。减层只用于链路验证，不能补层后直接当64层checkpoint恢复。4层模型仍约4.06B，也不是轻量随机小模型。

### ROCm算子与恢复状态修复

| 故障/修改位置 | 修复要点 | 保留的验证及适用范围 |
| --- | --- | --- |
| worker先导入TE导致设备绑定问题；[文本worker](../../../examples/qwen/qwen38_text_sft_worker.py) | 先按LOCAL_RANK绑定GPU，再加载TE/compat及runner；缓存按host/rank隔离，保存结构、HF导入完成和退出/峰值证据 | 各rank实际GPTModel、64层覆盖、无视觉/冻结参数均审计；worker接受`SystemExit(0)`为正常退出，避免成功提前退出却写失败标记。[退出回归](../../../tests/unit_tests/qwen38_text_sft/test_worker_exit.py) |

### MTP接入与长测审计

| 修改位置 | 关键原因/行为 | 验证边界 |
| --- | --- | --- |
| Bridge [gpt_step.py普通forward](../../../3rdparty/Megatron-Bridge/src/megatron/bridge/training/gpt_step.py:438) | 仅启用MTP时向模型传入原始assistant `loss_mask`。原普通路径不传mask，MCore会退化为全1，辅助目标可能监督user/system/padding | MCore内部负责一次labels/mask roll，Bridge不预先移位，也不改变主LM mask；schedule-plan路径原本已有mask。不得重复roll |
| [MTP测试](../../../tests/unit_tests/qwen38_text_sft/test_mtp.py) | 真实forward参数捕获、tokenizer/collator、辅助梯度和shift边界核验，覆盖单轮/多轮、中文、结束符、padding和窗口末尾 | 保证辅助监督与assistant-only一致，不能仅测试配置开关。MTP必须为full attention，不能沿最后GDN层误构造DeltaNet |
| [MTPAudit](../../../examples/qwen/qwen38_mtp_audit.py) | 核验实际HF抽样导入、MTP梯度有限非零、eh_proj参数更新；长测增加真实completed step与稀疏采样 | 默认间隔1保持原行为；长测显式`MTP_AUDIT_INTERVAL=100`并保留保存/结束边界，恢复记录从251起而非重新记1。[8项回归结果](../../../runs/qwen38_text_sft_steady_20261116/mtp_audit_tests.log) |
| 吞吐及长SEQ/VL入口 | 文本默认`LOG_THROUGHPUT=true`；4096/8192统一模型/数据SEQ；VL独立recipe显式加载原始多模态HF，语言/视觉/projection均不冻结，关闭MTP | 已完成数据兼容及dryrun，不能当作实际长SEQ/VL训练通过。VL旧性能recipe的随机初始化配置不能直接替代全参SFT |
| 独立[文本入口](../../../examples/qwen/run_qwen38_text_sft.sh)/[VL入口](../../../examples/qwen/run_qwen38_vl_sft_full.sh) | 从hostfile自动选节点，每节点一个torchrun父进程，统一透传配置；修复远端vendor脚本在nounset下引用未定义CMAKE_PREFIX_PATH的问题 | 9项单测及n05+n07实际SSH/MPI CPU参数探针通过；未由此重复GPU/IB集合通信验收 |

## 阶段测试记录与失败结论

以下均为当时实际完成的测试。JSON中的checkpoint路径是历史位置；权重和参数比较副本现已删除，日志/配置/审计JSON仍在。

| 时间/阶段 | 实际范围及主要结果 | 证据 |
| --- | --- | --- |
| 2026-11-13，UltraChat减层实测 | 16语言层、4GPU（4–7）、TP2/PP2、GBS8，5步，4rank真实HF导入；loss 8.825417→8.385675、无NaN/跳步；保存含optimizer/RNG。**未测试恢复，不代表64层通过** | [result.json](../../../runs/qwen38_ultrachat_parquet_20261113/result.json)；原checkpoint约113GiB |
| 2026-11-14，64层首次恢复链路 | TP2/PP4/DP2，5步保存→恢复至8步、连续8步及同节点匹配恢复；早期混合IB/Socket恢复仅作历史数值证据 | [result.json](../../../runs/qwen38_text_sft_dual_20261114/result.json)、[对照结果](../../../runs/qwen38_text_sft_dual_20261114/resume_comparison.json)；不作为当前IB性能验收 |
| 2026-11-14 14:47，无MTP全IB验收 | n05+n07、64层、TP2/PP4/DP2，连续8步及step5恢复至8步；日志指标、545个抽样语言tensor一致，最终validation 0.8453099，optimizer step8/scheduler samples256，恢复不重新导入HF | [result.json](../../../runs/qwen38_text_sft_ib_n05_n07_20261114/result.json)、[resume_comparison.json](../../../runs/qwen38_text_sft_ib_n05_n07_20261114/resume_comparison.json) |
| 2026-11-14夜，原MTP拓扑试验 | TP4/PP4/DP1的无MTP基线8步通过；开启MTP后embedding同步IB失败，后续多节点组合仍未通过。该失败没有被PP2成功结论覆盖 | [原MTP试验目录](../../../runs/qwen38_text_sft_mtp_20261114/)、[n06+n07探针](../../../runs/qwen38_text_sft_mtp_n06_n07_20261115/) |
| 2026-11-15 01:20，MTP独立功能 | n05两卡、4语言层+完整MTP1、TP2/PP1/DP1、SEQ128/GBS2；5→8恢复与连续8步通过，全部424,699,392个MTP参数和35个语言tensor一致，最终validation 7.135346。**不是全64层** | [result.json](../../../runs/qwen38_text_sft_mtp_smoke4l_20261115/result.json)，明确`full_64_layer_verified=false` |
| 2026-11-15 02:20，完整64层MTP全IB | n05+n07、TP4/PP2/DP2；匹配无MTP基线、MTP初训保存、恢复和连续对照四阶段通过。全部MTP参数（Megatron 12个融合tensor）及545个抽样语言tensor逐元素一致，最终validation 0.8456035，设备采样峰值57930MiB | [result.json](../../../runs/qwen38_text_sft_mtp_tp4_pp2_dp2_ib_20261115/result.json)；各phase checkpoint审计JSON保留 |
| 2026-11-16 00:28–02:55，首轮长测 | 无MTP/LR1e-5，500步及250恢复稳定；训练窗口下降2.05%未达附加3%门槛，validation持续下降，因此未启动该轮MTP | [首轮统计](../../../runs/qwen38_text_sft_steady_20261116/nomtp_assessment.json) |
| 2026-11-16 02:57–07:50，第二轮长测 | 仅将LR调为2e-5，其余数据/步数/拓扑/门槛不变；先无MTP500步通过，再独立MTP500步通过，各在250保存恢复。具体loss、性能和边界见下节 | [第二轮汇总](../../../runs/qwen38_text_sft_steady_lr2e5_20261116/summary.json)、[完成复核](../../../runs/qwen38_text_sft_steady_lr2e5_20261116/completion_reaudit.json) |
| 2026-11-16，配置与启动扩展 | 全量现有短数据在4096/8192下token检查通过；最终文本两种长度及VL4096 dryrun通过；随后hostfile两入口真实双节点CPU启动探针通过 | [配置检查](../../../runs/qwen38_long_vl_launch_review_cpu_20261116/validation.json)、[MPI与清理验证](../../../runs/maintenance_20261116/validation.json)；未训练长SEQ/VL |

### 需要保留的排障结论

- **IB与Socket边界**：早期n05/n10恢复尝试曾用Socket完成数值检查；用户随后明确IB故障换节点、禁止Socket数据面，后续n05+n07全IB验收替代它。不能把早期结果用于IB吞吐比较，也不要照抄历史Socket环境。
- **embedding配对与拓扑**：原TP4/PP4跨节点配对如rank0↔12，对应物理GPU序号0↔4；TP4/PP2的rank0↔8为物理GPU同序号。606.25MiB BF16同步探针在PP2配对通过后，完整训练才通过。证据支持改变拓扑绕过该路径，尚未证明插件内部根因或所有PP4配置均失败。
- **n10本地算子问题独立于通信**：后来原生`torch.zeros(1, device='cuda')`亦可复现invalid device function。n05/n10的Torch/HIP及RCCL库版本核验不能证明所有环境等价；驱动版本差异只是线索，未证明因果。
- **占用与本轮进程区分**：n06曾有其它SGLang任务使前4卡各占约3.4GiB，本轮未清理它；n07曾新增约58GiB/卡占用导致恢复OOM，用户释放后重跑通过。不能将这些历史占用归因于本轮残留，也不能把当时空闲状态当作当前空闲。
- **运行目录与临时路径**：容器overlay满曾导致vendor `/var/log/hylog`写入失败，使用`HYLOG_DISABLE_ERROR_LOG=1`；checkpoint manager的AF_UNIX路径过长通过短的`/public/home/yugeng/work/.q38tmp/<hostname>`解决，不清理系统目录。共享文件系统token检查临时目录析构曾报告Directory not empty，检查器仍exit0；这不是训练数值失败。
- **证据中的失败与最终修正**：历史result可能保留早期pending字段或退出标记问题，最终phase审计、launcher退出码、对照JSON及当前handoff共同判断。最初长SEQ dryrun在运行中修改shell导致中断，已固定脚本后重跑；最初MPI探针暴露vendor nounset问题，修复后重跑。未把失败尝试删改成成功记录。

## 单节点发起双节点训练

在n05宿主机进入容器：`docker exec -it yg_qwen38 bash`。当前[hostfile](../../../examples/qwen/hostfile)为（以文件实际内容为准）：

```text
m09r2n05
m09r2n06
```

hostfile每行一个节点，可修改或用`HOSTFILE=/绝对路径/hostfile`覆盖。节点数默认按有效行数计算，首行为MASTER_ADDR；可显式设置NNODES/MASTER_ADDR。MPI自动生成每节点slots=1的文件，原hostfile即使带slots=8也不会重复启动父进程。`MPI_SSH_PORT`可覆盖11463。节点共享代码/数据/输出路径且容器SSH免密互通。

```bash
cd /public/home/yugeng/work/Megatron-bridge-das-qwen38-sft-review
unset NODE_RANK MASTER_ADDR

# 只在一个容器中执行；长数据需先prepare，已有目录不覆盖。
SEQ_LENGTH=4096 bash examples/qwen/run_qwen38_text_sft.sh prepare
SEQ_LENGTH=4096 SAVE_DIR="$PWD/outputs/text4096-run01" \
  bash examples/qwen/run_qwen38_text_sft.sh sft

# 独立VL全参SFT：真实图文，完整语言/视觉/projection全部不冻结，MTP关闭。
SAVE_DIR="$PWD/outputs/vl4096-run01" \
  bash examples/qwen/run_qwen38_vl_sft_full.sh sft
```

上述两种训练分别运行，不要同时占用相同节点。默认500步/warmup20/LR1e-5，首轮建议用独立目录先测`TRAIN_ITERS=30 WARMUP_ITERS=2 SAVE_INTERVAL=30`，文本短测再设`EVAL_ITERS=0`。8192时同步改SEQ和SAVE_DIR。详细准备/启动指令见[长序列与VL说明](qwen38-sft-long-sequence-and-vl.md)。

`NNODES=1`保留单节点运行；显式`NODE_RANK=0/1`保留手工逐节点模式，日常hostfile启动不要设置NODE_RANK。16层单节点脚本仍固定NNODES=1。`preflight`和`dryrun`仅本地检查，不远程启动GPU训练。

## 已完成测试与性能

三个独立500步运行均为全64层、SEQ1024、GBS32，从各自HF初始化，各在250步保存恢复；每模式消费16000样本。

| 运行 | 训练loss均值21–70→451–500 | 降幅 | validation loss100→500 |
| --- | --- | --- | --- |
| 无MTP，LR1e-5 | 0.8593868→0.8417789 | 2.05% | 0.8849659→0.8578868 |
| 无MTP，LR2e-5 | 0.8996924→0.8693686 | 3.37% | 0.9325169→0.8857405 |
| MTP1，LR2e-5，系数0.1 | 0.8998075→0.8695273 | 3.37% | 0.9326072→0.8860344 |

固定验证集各次结果如下，未采集step0验证基线：

| step | LR1e-5无MTP | LR2e-5无MTP | LR2e-5 MTP1 |
| --- | ---: | ---: | ---: |
| 100 | 0.8849659 | 0.9325169 | 0.9326072 |
| 200 | 0.8790989 | 0.9247986 | 0.9250911 |
| 300 | 0.8679221 | 0.9056366 | 0.9055673 |
| 400 | 0.8597372 | 0.8895817 | 0.8897842 |
| 500 | 0.8578868 | 0.8857405 | 0.8860344 |

三组五次validation均下降。3%训练窗口降幅是额外测试门槛，不是通用SFT正确性标准；LR1e-5最终验证loss最低，不能认定2e-5更优。MTP辅助loss均值1.0287686→0.9672309，但未证明MTP改善模型质量。

| LR2e-5稳态指标 | 无MTP | MTP1 |
| --- | ---: | ---: |
| mean / P50步耗时 | 13.323 / 13.316 s | 13.765 / 13.751 s |
| P95，线性插值 | 13.645 s | 14.110 s |
| samples/s | 2.4018 | 2.3248 |
| padding token/s/GPU（TGS） | 153.72 | 148.79 |
| 日志MODEL TFLOP/s/GPU，时间加权近似 | 23.86 | 24.62 |
| 30秒设备采样显存最大 | 51378 MiB | 57931 MiB |

稳态窗口475步，排除1–20、101、201、251、301、401。samples/s=GBS/mean_seconds；TGS=samples/s×SEQ/16。MTP平均步耗时增加3.31%，中位数增加3.27%。TFLOP/s来自模型公式估算，不是硬件计数或MFU，不能与16层VL短测直接比较。设备采样可能漏掉瞬时峰值，卡容量65520MiB。当前文本recipe默认`LOG_THROUGHPUT=true`，主iteration行输出吞吐；rank0原本也有独立MODEL_TFLOP/s/GPU行。

曲线“50-step mean”为当前步及前49步的移动平均，不是中心窗口。validation曲线不做这种平滑。参见[三组loss对照图](../../../runs/qwen38_text_sft_steady_lr2e5_20261116/loss_comparison.png)。

恢复审计：250/500 optimizer step及scheduler样本8000/16000匹配；RNG恢复，恢复不重新导入HF，step251消费8032样本，全程LR符合500步cosine。未做额外500步不中断逐参数对照。此前8步短测的恢复/连续对照通过，全部MTP参数和545个抽样语言tensor一致；不等于全部语言参数逐位证明。checkpoint清理后保留这些历史审计结论，不再能重做原checkpoint数值对比。

## 已知限制和后续测试

- IB数据面固定`NCCL_NET="NCCL RDMA Plugin v8"`、`NCCL_IB_DISABLE=0`；IB报错换节点，不使用Socket完成。Socket bootstrap不代表数据面回退。
- TP4/PP4/DP1的MTP跨节点非对等embedding配对曾失败，不能套用TP4/PP2/DP2的通过结论。n10曾在原生Torch CUDA操作中报invalid device function，优先使用n05/n06/n07；根因未最终证明。
- 长SEQ可能改善小矩阵效率，但未证明短SEQ是主要瓶颈；旧短数据padding到4K/8K的平均输入占比仅17.44%/8.72%。先检查真实长样本、显存和有效token吞吐，再决定长测。
- 完整VL需真实图文验证视觉/projection梯度及参数更新，纯文本不能替代。Birds长跑需单独划分验证集以判断泛化。
- 旧500步终点LR为0；新实验从HF重新初始化。如需接续某个未来checkpoint，必须明确scheduler方案，不能仅增加TRAIN_ITERS。
- 训练启动后按实际稳态耗时、评估和保存开销估计结束时间，每5～10分钟检查进展。健康检查器`examples/qwen/qwen38_training_health.py`已用CPU进程验证；此前对旧workflow的600秒监控增强是在GPU长测结束后添加，不能算历史监控证据。

## 关键修改与证据

- 启动仅保留：[统一文本入口](../../../examples/qwen/run_qwen38_text_sft.sh)、[独立VL入口](../../../examples/qwen/run_qwen38_vl_sft_full.sh)，各自包含环境/缓存/MPI代码，不互相引用。旧包装及共用MPI/runtime已删除；单节点通过NNODES=1配置，减层通过EXPECTED_NUM_LAYERS与LOCAL_HF_PATH配置。
- 回归：[LauncherTests](../../../tests/unit_tests/qwen38_text_sft/test_full_workflow.py)。此前CPU探针及清理记录仍为历史证据；不将旧脚本路径当作当前可执行入口。
- 长测：[方案](qwen38-text-sft-steady-plan.md)、[结果](../../../runs/qwen38_text_sft_steady_lr2e5_20261116/result.json)、[汇总](../../../runs/qwen38_text_sft_steady_lr2e5_20261116/summary.json)、[无MTP统计](../../../runs/qwen38_text_sft_steady_lr2e5_20261116/nomtp_assessment.json)、[MTP统计](../../../runs/qwen38_text_sft_steady_lr2e5_20261116/mtp_assessment.json)。
- MTP短测：[结果](../../../runs/qwen38_text_sft_mtp_tp4_pp2_dp2_ib_20261115/result.json)；无MTP TP2/PP4短测：[结果](../../../runs/qwen38_text_sft_ib_n05_n07_20261114/result.json)。完整数值/IB/退出日志与审计JSON保留。
- 配置检查：[4096/8192文本和VL4096 dryrun](../../../runs/qwen38_long_vl_launch_review_cpu_20261116/validation.json)。这些只证明配置与数据兼容，不证明GPU训练可容纳。
- 本轮启动验证：[validation.json](../../../runs/maintenance_20261116/validation.json)。文本与VL均通过n05+n07真实SSH/MPI启动探针，正确生成两端rank、传递SEQ/LR/视觉参数等；探针不导入Torch、不执行训练，不证明GPU/IB集合通信性能。9项定向测试通过，覆盖默认hostfile、节点数推导、slots=1、两种入口、参数和SSH端口。

## 交接时的检查顺序

1. 先确认使用本工作树和现有ROCm容器，再读本文件的当前状态；详细旧方案只作设计和历史依据，不按其中过时状态重启已完成测试。
2. 对照模型HF目录、MTP模式、数据SEQ/thinking和hostfile；新实验使用独立SAVE_DIR。纯文本与完整VL选择各自入口。
3. 长SEQ或VL先dryrun和30步实际测试，核验全参范围、HF导入、梯度/权重更新、loss/grad norm有限性、IB、峰值及checkpoint保存，再长跑和恢复。
4. 后续修改优化器适配、mask、初始化/恢复路径或MPI时，运行对应定向测试；修改原因及验证结果继续补充到本handoff，保留结果与证据路径，不只更新状态摘要。

## 2026-11-16：4K数据准备退出清理修复

SEQ4096 prepare已发布16384/256条training/validation，随后weakref/tempfile退出清理报`Directory not empty`。本次核对JSONL均可解析，manifest记录SEQ4096、training invalid=3/token_filtered=7，未覆盖数据目录。旧检查器把Dataset文件映射与TemporaryDirectory保留至解释器退出，存在共享文件系统延迟删除的清理顺序问题；检查旧临时目录时已为空，未捕获当时的`.nfs`文件，具体残留文件名未证实。

修改`data_preparation/prepare_qwen38_text_sft.py::make_token_checker`：用上下文管理临时目录，构建Dataset后释放只用于初始化的`indexed_dataset`，再删除临时文件。后续仍调用原Dataset的`_process_example/collate_fn`，不改变模板、分词或监督策略。新增`tests/unit_tests/qwen38_text_sft/test_tools.py::DataTests.test_token_checker_releases_files_before_temporary_cleanup`，核验资源释放及检查器仍能进入预处理；标准库定向测试16项通过。当前容器无可见GPU，未复跑依赖TE导入的真实tokenizer链路。`fix_mistral_regex`警告独立保留待核验，本次未改变tokenizer设置。


## 2026-11-16：n05+n06启动时PRTE lost communication修复

用户启动SEQ4096时，PRTE报告丢失n06远端daemon。实际hostfile为n05+n06，两端容器SSH可达，尚无训练日志。仅执行hostname的MPI复现捕获：`mkdir /tmp/prted.m09r2n06...`报`No space left on device`，随后`prte_session_dir failed`，最终才出现lost communication，退出码213。n06容器根盘可用空间为0；两端prted均PRRTE 3.0.5rc2、二进制SHA256一致。本次确定原因是远端PRTE临时目录创建失败，不是已发生的GPU IB通信故障。

原脚本通过`-x TMPDIR`传给训练进程，但SSH启动的PRTE daemon更早运行，仍使用远端默认`/tmp`。修改历史公共MPI逻辑`qwen38_sft_mpi.sh`（现已删除）：创建短目录`/public/home/yugeng/work/.q38tmp/prte`，向mpirun显式传入`--prtemca prte_tmpdir_base <目录>`，同时作用于本地和远端daemon；可用`PRTE_TMPDIR_BASE`覆盖，文本/VL/原生后端入口均复用此修复。未清理容器系统目录，未改变IB数据面配置。

验证：[validation.json](../../../runs/mpi_n05_n06_diagnosis_20261116/validation.json)。保持n05+n06不变，仅指定home会话目录后，hostname探针exit0；通过实际长序列文本入口的CPU参数探针exit0、两端rank0/1正确，SEQ4096及IB插件环境保持。新增[PRTE会话目录回归](../../../tests/unit_tests/qwen38_text_sft/test_full_workflow.py)，该文件当前11项测试通过。原始错误见[baseline.log](../../../runs/mpi_n05_n06_diagnosis_20261116/baseline.log)。未加载模型或重跑GPU训练，因此不据此宣布4K训练或GPU IB集合通信已通过。用户原SAVE_DIR尚未产生checkpoint，可用原命令重新启动。

## 2026-11-16：text4096-run01启动及步耗时诊断

用户命令：`SEQ_LENGTH=4096 SAVE_DIR="$PWD/outputs/text4096-run01" bash examples/qwen/run_qwen38_text_sft_long.sh sft`。本次仅检查代码、日志、缓存文件时间和容器状态，未重新启动训练、未改变训练代码/配置、未提交Git。汇总及逐步指标见[诊断证据](../../../runs/qwen38_text4096_run01_review_20261116/evidence.json)。

### 实际进展与正确性边界

- 实际节点为n05+n06（不是历史500步的n05+n07），16GPU、64层、TP4/PP2/DP2、CP1、MBS1/GBS32、SEQ4096、无MTP，full/uniform/1重计算，通信重叠关闭；每个DP副本每步16个microbatch。
- 使用新prepare的`ultrachat-qwen38-train16384-val256-seq4096-disabled`，不是旧1024数据目录；16384/256条，manifest记录SEQ4096/thinking disabled。不能将旧1024数据平均长度及padding占比套用到这批数据。
- 16个`model_rank_*.json`与`hf_loaded_rank_*.json`保留模型结构及HF导入完成证据。日志到step16（15:56:53），消费512样本，loss从1.582169至0.9693398，所有已记录步的loss/grad norm有限、NaN/skip计数均为0；不能据此认定效果或收敛已通过。
- 17:11–17:12只读检查时，两端`yg_qwen38`容器均无训练进程，n05 GPU占用/利用率为0。输出无checkpoint、无rank最终退出JSON，日志无明确退出原因，不能宣布正常完成，也不能判断停止原因；4K保存恢复尚未验收。

### 启动耗时拆分（北京时间）

| 区间/阶段 | 实际证据 | 解释 |
| --- | --- | --- |
| MPI派发前后至逐节点检查完成 | `hostfile.slots` mtime 15:19:56；node0/node1 manifest mtime 15:32:58/15:33:14，窗口约13分02秒/13分18秒 | `train_qwen38_text_sft.sh`在每节点torchrun之前无条件执行`check_qwen38_text_sft.py --check-tokens`，各自串行遍历16384+256条，调用真实模板/分词/collate；检查器还构造token/label/mask列表并解码监督文本，即使启动检查只使用通过/失败。缺少最外层console计时，不能把整个窗口精确归为tokenize，且两节点检查并行，不能相加成26分钟 |
| node1检查完成至Megatron初始化完成 | 15:33:14→15:35:59，约165秒；其中Megatron初始化计时40.692秒 | 包含torchrun、worker导入HCU/TE等和进程组初始化；其余时间没有细分计时，不能全归为IB |
| 模型/优化器构建 | 日志计时约0.494秒，15:36:00完成 | 这部分不是主要等待来源 |
| HF权重导入 | `load-checkpoint`各rank 224.824秒；HF完成标记15:39:44–45 | HF权重约53.79GB，Bridge逐tensor转换、分片、通信并复制到Megatron参数；该计时并非纯磁盘读，因此不能据此断言共享存储带宽故障 |
| 训练数据迭代器构建 | 各rank约1.94–1.96秒，15:39:47完成 | 日志中的dataset indexing/loading不是分钟级瓶颈 |
| 首个iteration | 15:39:47→15:50:12，日志624.2601秒 | 强证据指向冷编译/首次运行开销：共享缓存中15:39–15:49有44736个Triton文件mtime更新，Inductor在15:40–15:47亦有更新；包含`chunk_bwd_kernel_dqkwg`、`l2norm_bwd_kernel`及编译融合算子。worker按host/local-rank隔离Triton缓存，换节点和1024→4096不能保证命中历史缓存，16rank可能重复编译并争用CPU/文件系统。mtime是缓存写入证据，不是独占归属或精确编译耗时剖析 |

从本轮`hostfile.slots`到第一步完成共约30分16秒，主要可见窗口是训练前全量检查、HF导入和首步冷编译。`cuda_graph_impl=none`不代表没有Triton/Inductor编译，两者不是同一个开关。下一次同节点/同配置且保留缓存时首步通常有望缩短，但本轮没有热缓存重跑对照，不能保证时间。

### 步耗时波动：两种现象分别判断

| 步数 | 平均耗时 | 范围 |
| --- | ---: | ---: |
| 1 | 624.260秒 | 首次运行，不能作为稳态吞吐 |
| 2–5 | 23.690秒 | 20.112–26.134秒 |
| 6–10 | 16.075秒 | 15.705–16.505秒 |
| 11–16 | 37.753秒 | 32.302–41.405秒 |

step11后的均值约为step6–10的2.35倍，不能用“前几步预热”解释全部波动。所扫描缓存15:50–16:00没有Triton/Inductor文件mtime更新，这削弱了后半段持续冷编译的解释，但不能排除其它库的调优或未落盘工作。当前只有16步，不能把任一短窗口称为已验证稳态。

已核对的排除项/边界：

- recipe的`dataset_kwargs={"pad_to_max_length": True}`经builder传给`GPTSFTChatDataset`，collator固定到4096；assistant mask只改变监督范围，不免除padding计算。有效文本长短会影响CPU分词，不能直接用长短样本解释GPU主干算量翻倍。`NUM_WORKERS=0`使在线预处理在训练进程执行，仍是待采样的停顿来源。
- 本次eval_interval=100、save_interval=250，step11–16没有评估/保存事件。LR warmup20改变学习率，不改变固定形状前反向计算量，不能把LR warmup当作耗时解释。
- 各步的两种日志（node0 `Step Time`和node1 `elapsed time per iteration`）相互吻合。`GPU utilization: ...MODEL_TFLOP/s/GPU`实际是公式估算TFLOP/s，**不是硬件GPU利用率百分比**。
- 通信日志明确选择`NCCL RDMA Plugin v8`/IB；找不到v10/v9符号后加载v8不代表失败。日志同时含`GPU Direct RDMA Disabled`及channel `GDRDMA`字样，不能只截取某一条认定实际GDR模式。未看到解释step11退化的明确传输错误；这不等于排除通信性能抖动。
- 两端容器overlay当前均100%、available=0；这是已确认的环境问题，曾导致PRTE启动失败，但本轮PRTE已迁到home。当前训练没有对应ENOSPC报错，不能直接认定它导致step11变慢。不要直接清理其它任务/系统目录。
- full recompute和关闭通信重叠会影响总体速度，但在本次各步保持一致，无法单独解释第11步突然变慢。历史1024的n05+n07性能也不是本轮n05+n06的严格对照。

缺失的是15:53前后的GPU频率/功耗/温度、全宿主机其它GPU进程、CPU/iowait、网络计数和分阶段timer。本次进程已退出，当前空闲状态不能反推历史无争用；因此step11后的直接根因仍未确定，不应写成已修复。

### 下一轮定位优先级

1. 保留现有缓存，保持同节点/SEQ/拓扑/算子配置，用新SAVE_DIR跑30–50步性能短测（如`TRAIN_ITERS=30 WARMUP_ITERS=2 EVAL_ITERS=0 SAVE_INTERVAL=30`；这会改变短测LR调度，不与500步作数值对照）。从启动前每5–10秒记录两节点GPU利用率/频率/功耗/温度/显存及GPU进程、CPU/iowait、IB错误/吞吐，连同最外层带时间日志保存。先解决/确认容器根盘可写空间，再跑长测。
2. 用分阶段timer或短profiler区间区分batch生成、forward/backward、optimizer、TP/PP/DP通信；若硬件利用率低而CPU/iowait升高，再单独比较预分词或`NUM_WORKERS`。若出现其它GPU任务或降频，先解决资源问题；不得只因猜测改通信插件或降级Socket。
3. 启动优化优先考虑对已prepare数据使用与模型/tokenizer/模板/SEQ/数据hash绑定的token检查缓存，以及检查器的仅验证模式（不构造返回审计文本）；必须维持数据失配时重新全量检查。本轮尚未实现，不直接删除`--check-tokens`。
4. 首步与前若干步从吞吐统计排除，待采样确认稳定后再比较通信重叠或重计算策略；这些均需单独验证数值、显存和恢复，不能在现有不明退化上同时改多个变量。

本次变更仅为本handoff“当前状态”和本节，以及新增诊断`evidence.json`；未修改可执行代码，未运行训练或单元测试。证据JSON已按训练日志逐步解析并核对，共16步。

## 2026-11-16：训练前检查、冷编译及旧Birds入口对照方案

本节为用户要求的详细分析及待实施方案。仅更新文档，不删除检查、不更改缓存布局/编译策略、不启动GPU训练、不提交Git。后续实施按下述顺序做独立对照，不把预期收益写成已实测结果。

### 训练前检查在哪里、检查什么

调用链：`run_qwen38_text_sft_long.sh` → `run_qwen38_text_sft_full.sh` → `train_qwen38_text_sft.sh`。后者先经MPI在每节点启动自身，再在第55行附近运行`check_qwen38_text_sft.py --check-tokens --manifest ...`，成功后才torchrun。实现主体为`tools/checkpoint/check_qwen38_text_sft.py::check_environment`，真实分词检查复用`data_preparation/prepare_qwen38_text_sft.py::make_token_checker`。检查发生在第一步计时之前，与首步624秒是不同开销。

| 检查层次 | 内容 | 去掉的影响 |
| --- | --- | --- |
| 模型/拓扑/运行参数 | 文本模型类型、4/16/64层及hybrid结构、MTP匹配、safetensors header中的key/shape/dtype/字节跨度、tokenizer文件存在、TP/PP/DP/CP、batch可整除、步数/LR/重计算 | 失配可能直到分布式构建后才报错；header检查不等于对全部权重内容做校验 |
| 数据基础检查 | 每行JSON/schema/messages、无视觉依赖、非空split、行数和完整JSONL SHA256 | 可能把格式错误、错误split或变更数据带入训练 |
| 全量token检查（耗时部分） | 对每行应用实际thinking模板和分词、预处理、collator，检查长度及至少一个有效assistant监督token | 失去启动前发现超长/无监督样本的能力；框架可能截断或产生空监督，不能假定必然显式报错 |
| 恢复与输出保护 | 恢复目录/manifest存在，模型配置、数据hash、SEQ、拓扑、GBS、thinking及MTP一致；新训练拒绝覆盖非空checkpoint目录 | 可能错误续跑、错误初始化或覆盖已有训练产物 |

`make_token_checker`当前除校验外，还把padding到SEQ的tokens/labels/loss_mask转为Python列表，并解码监督文本，供prepare审计使用；launcher只需要通过/失败，却也支付这些返回内容的成本。每节点各自串行处理16640条，两节点并行完成；不是每GPU做一遍，也不能把两节点时间相加。prepare已经做过一次token检查，但现有`data_manifest.json`只记model路径/SEQ/thinking等，未完整绑定tokenizer文件和处理代码内容，因此不能仅凭`token_checked=true`直接永久跳过。

**能否去掉：**全量token重复扫描可以优化或在有可靠证明时跳过；不建议删除整个checker。当前shell无现成的`SKIP_PREFLIGHT`/`TOKEN_CHECK_MODE`开关，以下模式均为拟新增设计，直接设置这些名字目前不会生效。从shell去掉`--check-tokens`会保留基础/恢复检查，但无条件这么改会降低数据保障，不是推荐最终实现。

建议方案A（优先，保持训练语义）：

1. `make_token_checker`增加仅验证模式，仍使用同一预处理/collator及长度/监督判断，但不构建审计列表或解码文本；prepare仅对少量展示样本调用完整审计模式。预期减少重复工作，具体占比必须计时。
2. 引入`full|cached|basic`三种显式模式。`full`始终全量；`cached`仅在有效检查凭据命中时跳过分词，失配自动全量；`basic`明确记录跳过token验证，只作为可追踪的手工选择，不能伪装成已验证。建议默认`cached`，无凭据首次自动全量。
3. 凭据绑定实际数据SHA256、tokenizer.json/tokenizer_config及特殊token配置、实际chat template/thinking、SEQ、loss_mode/长度策略、相关预处理/collator代码指纹和依赖版本。路径/文件mtime/旧manifest的布尔值均不足以作为有效凭据。通过后原子写入，失败不发布；缓存记录本身是本项目生成的检查结果，不用于接收不可信证明。
4. 在共享代码/数据场景，只让协调端执行或复用一次完整检查，再启动MPI；每个节点仍执行模型/环境/拓扑/恢复保护和数据hash对照，确保远端确实看到同一数据。不要把16个worker变成检查进程，也不要让两个节点同时覆盖公共凭据。
5. 给基础检查、数据hash、token检查分别记录开始/结束/行数/命中原因。回归覆盖：有效命中、数据/tokenizer/template/SEQ/处理代码变化失效、超长/无监督失败、恢复失配继续拒绝、两节点读取不一致失败、并发写入不产生半成品。

涉及文件：checker、通用数据准备工具、UltraChat准备工具（发布凭据）、文本launcher、MPI配置透传、对应工具/launcher测试。目标是让重复启动不再全量分词，保留轻量扫描/hash和恢复保护；不能承诺精确省13分钟，因为此前窗口由mtime推算。

### 冷编译是什么意思、为什么缓存文件支持这个判断

这里的“冷”指当前程序/输入形状在选定缓存中尚无可复用编译产物。第一次执行时，系统不仅计算训练结果，还需生成GPU机器代码，部分算子还会试跑若干配置以选择较快实现；后续可加载或复用结果。编译缓存与HF权重、Megatron checkpoint是不同产物。换SAVE_DIR不会必然清空当前缓存（当前QWEN38_CACHE是独立固定目录），但换缓存根目录、输入形状/后端、GPU或软件版本可能失去复用条件。

本项目有两条实际链路：

- MCore `megatron/core/jit.py`在满足Torch版本时将`jit_fuser`设为`torch.compile`；`ssm/gated_delta_net.py`的`_apply_gated_norm`、`_prepare_qkv_for_gated_delta_rule`、`_compute_g_and_beta`均用它装饰。这些局部图可通过Inductor生成Triton代码，即使整个模型未显式调用`torch.compile`。
- GDN还直接调用FLA的`chunk_gated_delta_rule`/`l2norm`等Triton算子。因此只关闭PyTorch图编译也不等于消除所有Triton JIT/调优。`cuda_graph_impl=none`控制CUDA graph，不能据此判断上述编译关闭。

首步15:39:47–15:50:12耗时624秒，期间所扫描缓存中15:39–15:49有44736个Triton文件mtime更新，并有Inductor产物，名称包含GDN反向、l2norm反向、融合sigmoid等；此后已有步数可降至16秒左右。代码链路、时间重合及后续加速共同强烈支持冷编译/首次执行开销是首步主要原因。但一个内核会写多种IR/源码/二进制/metadata，多rank也有副本，**44736个文件不等于44736个不同算子**；文件数量/mtime不能证明624秒全是编译或给出精确占比，仍需编译日志/CPU采样及相同配置热缓存重跑对照。

17点后只读核对现有`yg_qwen38`：Torch 2.11.0、Triton 3.6.0，Inductor FX graph cache已启用，`torch._inductor.config.compile_threads=32`。按8worker/节点估算，编译并发预算可能达到8×32=256；这是上限量级，不是实际同时运行256个线程的观测，尚无当时CPU采样证明过度并发。安装源码支持`TORCHINDUCTOR_COMPILE_THREADS`、`TORCHINDUCTOR_FX_GRAPH_CACHE`及`TORCHINDUCTOR_AUTOGRAD_CACHE`，实际收益/默认值需分别检查，不能声称FX cache目前关闭。

官方机制参考：[PyTorch编译缓存](https://docs.pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html)、[缓存目录/配置](https://docs.pytorch.org/tutorials/recipes/torch_compile_caching_configuration_tutorial.html)、[Triton autotune](https://triton-lang.org/main/python-api/generated/triton.autotune.html)。这些文档说明机制，具体可用选项以当前安装源码为准。编译产物缓存与调优结果缓存不是同一层，磁盘有二进制也不保证下次完全免调优。

### 为什么旧run_qwen38birds.sh看起来更快

直接阅读的参考是原工作树`/public/home/yugeng/work/Megatron-bridge-das/examples/qwen/run_qwen38birds.sh`，不是本review树的同名副本。两者PROFILE默认不同，不能混读。此次未取得用户当次Birds运行的resolved config、日志、环境和缓存快照，以下是当前参考文件的静态差异，不能冒充对历史运行的完整重建。

| 方面 | 原工作树Birds入口当前默认 | 当前text4096入口 |
| --- | --- | --- |
| 目的/数据 | VL pretrain性能路径、Energon Birds，无全量JSONL token检查 | 纯文本HF初始化SFT、UltraChat，逐节点全量token检查 |
| HF参数导入 | provider `load_weights=False`，recipe继承的`pretrained_checkpoint=None`，launcher `load=null`且未另设pretrained路径；默认不导入HF参数，属于随机初始化性能路径 | `checkpoint.pretrained_checkpoint`显式指定转换后HF目录，导入真实语言权重 |
| 重计算 | recipe默认关闭full recompute | full/uniform/1；执行路径和可能生成的前反向图不同，不意味着每层一定单独编译 |
| 通信/批量 | 默认TP4/PP2，TP overlap=true，GBS128 | TP4/PP2，TP overlap=false，GBS32；相同DP时microbatch数量不同，耗时不能直接比 |
| 缓存 | 不主动设置本项目专用根目录、不按host/rank追加隔离目录，默认或调用者已有缓存可能被复用；env设TRITON_HOME=/tmp | `outputs/qwen38-sft-cache`，Triton/MIOpen再加host/local-rank子目录；Inductor当前为公共目录。不会自动搜索旧/tmp或原工作树缓存 |
| 首步测量 | 未获得当时日志及启动计时 | 有624秒首步日志及同期缓存写入证据 |

原工作树与当前review树的MCore `gated_delta_net.py`经逐字节比较一致。因此旧Birds也具有上述GDN编译机制，不能解释成“以前完全不编译”。合理候选包括旧缓存已热、原生/不同重计算路径、缓存共享程度及编译资源不同。特别注意：无HF导入可以解释启动前少了约225秒，**不能直接解释training iteration 1中的624秒差异**。旧随机初始化性能入口也不能直接替换成正确SFT入口。

### 冷编译优化方案B（按收益与改动风险排序）

1. **先验证热缓存复用。** 记录两节点版本/GPU/缓存路径及各阶段计时，保留当前有效缓存，在同节点/同SEQ/同后端做独立SAVE_DIR短测，比较首步与稳态。默认缓存根目录已固定，不应每次加时间戳或清理。若想复用旧缓存，先核验同版本/GPU与产物兼容，再复制到独立候选目录做对照；不要覆盖当前已验证缓存。预计主要改善重复启动，首次面对新形状仍需编译，无法保证首次零开销。
2. **限制编译并发做单变量对照。** 拟在runtime设置`TORCHINDUCTOR_COMPILE_THREADS`并加入MPI透传，先对比当前32与每worker 2或4。OMP_NUM_THREADS=1不等于限制Inductor编译池；当前MPI白名单未透传此变量，单在外层export不能保证远端采用。减少并发可能缓解CPU/内存/文件系统争用，也可能使冷编译更慢，须比较冷启动总时长、CPU峰值及稳态，不直接承诺加速。复用热缓存的对照与编译线程的冷缓存对照应分开；冷对照用新目录而非删除原缓存。
3. **减少重复编译及共享文件系统小文件压力。** 现有Triton host/rank隔离优先保证稳定，但同构rank可能各编相同内核。先对缓存key/版本做去重分析，再试“保留每rank写目录、从已验证种子缓存复制/同步”或经并发验证后的每节点共享缓存。不能把Inductor误认为也按rank隔离；也不能未经压力测试直接让所有节点写一个新共享Triton目录。若确认共享存储小文件成为瓶颈，可把缓存工作副本放到容量独立且足够的节点本地盘，结束后原子回存；当前容器根盘已满，不宜迁到/tmp。
4. **预热完整前反向并保留产物。** 用匹配GPU/版本、BF16、TP后的形状、序列长度、后端和重计算路径做预热，覆盖反向而非只做forward。预热先在独立进程/输出中进行，不把修改参数、推进optimizer/RNG的“隐藏训练步”塞到正式step1之前。它将首次成本提前，并非减少从空缓存到结果的总成本；配合种子复用才可能减少各rank重复工作。
5. **选择性关闭局部图编译仅作备选实验。** 可在项目worker加载MCore相关模块前控制jit_fuser，或针对上述局部函数提供eager开关，不修改MCore源码。需保证导入时机正确（装饰完成后改全局变量不会自动解包已有函数），记录后端差异并验证数值/显存/保存恢复。直接FLA Triton仍要编译，且失去融合可能增加每步开销。原生算子仍可能触发GDN编译。

评估指标同时看“启动至首步”“排除预热后的稳态”“完整目标训练总时长”。例如仅作计算示例：若省600秒但500步每步增加2秒，总体反而增加约400秒；不能只以step1变快选择方案。B1/B2优先，B3/B4由采样结果决定，B5最后。先做A避免重复数据检查，再独立优化B；两者改善不同阶段，不解决尚未定位的step11性能退化，也不替代HF权重导入。

实施位置清单：`qwen38_sft_runtime.sh`（缓存/编译池策略）、`qwen38_text_sft_worker.py`（路径布局/可选局部编译控制）、`qwen38_sft_mpi.sh`（新变量双节点透传）、相应launcher/worker测试及本handoff。验收至少包括双节点实际环境值、缓存冷/热首步对照、数十步稳态、有限loss/梯度及关键后端变动的保存恢复回归。此次只核对源码、容器版本与配置并写方案，未实施这些改动或运行性能测试。

## 2026-11-16：按用户要求移除自动训练前检查，明确缓存复用

用户明确默认提供正确数据/参数，要求去掉全部训练前检查以缩短启动。该要求替代上一节“保留基础检查、缓存token检查”的待实施方案A；本次没有实施方案A或改变编译策略。

### 本次实际代码修改

- `examples/qwen/train_qwen38_text_sft.sh`第4阶段：删除dryrun/sft中的整个`check_qwen38_text_sft.py --check-tokens`调用，以及逐节点manifest写入/公共manifest复制。现在直接进入worker或torchrun，不再执行模型header核验、数据schema/全量分词/数据hash、拓扑和参数预检、恢复manifest比对、非空checkpoint目录保护。long/full/native/单节点文本包装都复用该入口，自动获得此行为。
- 保留显式`preflight`分支和独立checker源码，只有主动选择该模式才检查；不在正常训练中隐式触发。prepare是用户单独执行的数据生成操作，其数据准备/过滤检查保持既有行为。VL入口不属于本次文本preflight调用链，未修改。
- 不再生成新的`launch_manifest*.json`。既有文件原样保留，若复用旧输出目录，其旧manifest不是本次运行的新记录。框架仍按recipe/RESUME_FROM正常构建、加载及恢复；recipe内的构建断言和框架运行时错误不属于本次移除的独立预检。新运行没有checker要求的manifest，旧手工checker的恢复检查将不能用于验证这类新输出，训练恢复本身不依赖该sidecar。
- `tests/unit_tests/qwen38_text_sft/test_full_workflow.py::LauncherTests.test_training_and_dryrun_skip_all_automatic_preflight_checks`：CPU进程探针使用不存在的模型/数据/恢复路径、不合法预检拓扑及非空checkpoint目录，确认dryrun/sft直接派发worker，既有manifest不被修改，不生成节点检查记录。此探针不声称真实模型会接受错误输入，只证明launcher不再预检。

验证：`test_full_workflow.py` 12项、`test_tools.py` 16项通过，共28项；后者包括显式手工preflight仍有效的既有测试。`bash -n examples/qwen/train_qwen38_text_sft.sh`通过。未重启GPU训练，未实测缩短后的总启动时间，未提交Git。原约13分钟的预检窗口对应工作已移除，但HF导入和首步编译仍需执行。

### 缓存根目录及用户是否需要设置

`examples/qwen/qwen38_sft_runtime.sh`设置：

```bash
export QWEN38_CACHE=${QWEN38_CACHE:-${QWEN38_ROOT}/outputs/qwen38-sft-cache}
```

因此本项目默认根目录是：

```text
/public/home/yugeng/work/Megatron-bridge-das-qwen38-sft-review/outputs/qwen38-sft-cache
```

用户此前命令只指定SEQ_LENGTH和SAVE_DIR，无需手工设置QWEN38_CACHE，脚本会自动选择上述默认值；无法仅凭命令证明用户shell此前没有继承其它变量，但本次run01缓存产物位于上述默认目录。若显式`export QWEN38_CACHE=/其他路径`则脚本使用该路径。当前runtime会据此设置TRITON_CACHE_DIR、TORCHINDUCTOR_CACHE_DIR等，所以统一覆盖应使用QWEN38_CACHE，而非只在外层改其中一个子目录变量。

主要实际路径：

```text
qwen38-sft-cache/
  triton/<hostname>/rank-<LOCAL_RANK>/
  inductor/
  miopen-db/<hostname>/rank-<LOCAL_RANK>/
  miopen-kernels/<hostname>/rank-<LOCAL_RANK>/
  triton-home/
  torch-extensions/
  ...
```

Triton/MIOpen的host/rank追加逻辑在`examples/qwen/qwen38_text_sft_worker.py`；Inductor当前使用公共inductor目录。MPI会转发这些配置，两节点能访问对应路径时使用相同根目录。

### 是否保持目录不变就会直接复用

框架会自动尝试复用，不需要额外“加载缓存”命令。保持同一入口、QWEN38_CACHE、节点/LOCAL_RANK映射、GPU/软件版本及模型计算路径（SEQ/TP/精度/算子后端/重计算等）时，已有匹配产物可复用；换SAVE_DIR不会自动换缓存根目录。可以继续原命令并换新的输出目录，无需改变缓存设置。

但“目录没删”不是保证所有算子零编译的充分条件：换节点会进入另一hostname子目录，换输入形状/后端可能生成新key，版本或GPU差异可能不兼容，部分调优/进程初始化仍有成本。缓存不会省掉HF权重导入，也不能解释或修复run01的step11后退化。本次未进行热缓存GPU重跑，不承诺下一次首步具体耗时。

本轮修改仅为上述文本launcher、1个launcher回归测试和本handoff（当前状态、检查器表格及本节），缓存内容/布局与编译线程配置均未修改。

## 2026-11-16：run02性能、默认缓存路径及Ctrl+C残留清理

本次按用户要求分析`outputs/text4096-run02`，确认并清理两端该运行的残留worker。仅更新本handoff及诊断JSON；未改变生产训练源码、缓存配置、编译策略，未重启训练，未提交Git。

### run02实际结果

逐步日志解析见[performance.json](../../../runs/qwen38_text4096_run02_review_20261116/performance.json)。

| 指标 | run01 | run02 |
| --- | ---: | ---: |
| slots文件mtime→第一步完成（非最外层精确计时） | 15:19:56→15:50:12，30分16秒 | 18:46:49→18:53:30，6分41秒 |
| HF导入 | 224.824秒 | 227.616秒 |
| 首步 | 624.260秒 | 49.082秒 |
| 已记录步数 | 16 | 22 |

run02 step2–22（21步）均值15.213776秒、最小15.0056秒、最大15.5787秒、P95线性插值15.3425秒、总体标准差0.119873秒。记录至18:58:49/704样本，loss=0.8752431，梯度范数1.188；已记录22步无NaN/跳步。前16步loss/grad norm与run01在日志打印精度下一致，不构成逐参数位级证明。

取消自动预检消除了此前约13分钟的训练前扫描窗口；首步另降92.14%，与同节点同配置复用编译缓存一致，进一步支持run01首步有显著冷编译开销。HF导入仍约228秒，未被编译缓存消除。run01 step11后32–41秒的退化本次没有复现，不能由此反推当时根因，或声明某项已定位修复。此次为用户中断短测，未到评估/保存节点，不代表4K保存恢复/长测通过。

### 不自行设置缓存目录，还会不会缓存

会，但应区分“用户没有手工export”和“移除项目的全部缓存设置”：

1. **当前脚本下用户不手工export：** runtime仍自动选择`/public/home/yugeng/work/Megatron-bridge-das-qwen38-sft-review/outputs/qwen38-sft-cache`，并设置各子目录。run02残留worker的环境确认QWEN38_CACHE为此路径；改变SAVE_DIR不改变缓存根目录。没有修改它的必要。
2. **完全不使用项目缓存设置、使用库默认值：** 按当前容器Torch2.11.0/Triton3.6.0安装源码，直接Triton默认为`${TRITON_HOME:-$HOME}/.triton/cache`；Inductor默认为`<Python临时目录>/torchinductor_<用户名>`，未显式设置TRITON_CACHE_DIR时，它生成的Triton代码可使用`<Inductor目录>/triton/<device序号>`。容器用户为root，若所有相关变量均未设置，典型位置为`/root/.triton/cache`、`/tmp/torchinductor_root`及其triton子目录。TMPDIR会影响Python临时目录；实际项目`env/env.sh`还设置`TRITON_HOME=/tmp`，若保留这一行而删掉其它覆盖，直接Triton默认将变成`/tmp/.triton/cache`。
3. **只删runtime设置、保留当前worker：** worker仍有项目fallback逻辑，TRITON_CACHE_DIR未设置时使用`/tmp/qwen38-text-sft/triton/<host>/rank-<LOCAL_RANK>`；这仍不是完全裸库默认。不能简单只删除一个根目录赋值就认为会得到同一套默认路径。

默认缓存同样可以复用，显式缓存路径的价值是使位置确定、可持久保留、便于跨运行诊断，并避开容器临时目录清理/空间问题。库会按匹配条件自动加载已有产物；目录保留不保证输入形状、算子后端、软件/GPU版本变化后全部命中，也不保证完全免调优。不要为了恢复默认而将当前已热缓存删除或搬到已满根盘。安装源码依据：Triton `knobs.py::cache_knobs`、Torch `runtime/cache_dir_utils.py`和`runtime/triton_heuristics.py`。

### Ctrl+C残留的事实与原因边界

19:16后检查`yg_qwen38`：

| 节点 | 本次残留PID | 归属确认 |
| --- | --- | --- |
| n05 | 540508–540515，共8个 | worker命令含本review树的绝对路径；SAVE_DIR精确匹配run02；RANK0–7 |
| n06 | 3201336、3201337，共2个 | 相同worker路径及SAVE_DIR；剩余为RANK14–15 |

10个worker均PPID=1，且PGID=SID=各自PID，已被PID1接管；原MPI父进程/节点shell/torchrun在检查时均不在。n05 worker的wchan为`kfd_wait_on_events`，可证明在GPU驱动事件等待，不能据此独立证明RCCL内部死锁。n05各卡仍占约76–78%显存，n06 GPU6/7仍占约80%，说明是持有GPU资源的活动进程，不是仅剩僵尸条目。

当前启动关系为：终端 → `exec mpirun` → PRTE/SSH → 每节点`bash train_qwen38_text_sft.sh sft` → `python -m torch.distributed.run ... | tee` → 8个worker。终端Ctrl+C作用于前台进程组，不自动覆盖所有后代会话。现有TorchElastic源码中`SubprocessHandler._popen`明确使用`start_new_session=True`，退出时需由agent逐worker进程组发信号，默认等待30秒后强杀。MPI直接管理的是节点shell，当前shell的前台管道没有自定义INT/TERM/EXIT监督和兜底清理逻辑；shell/agent先退出时，独立会话worker可继续留存。

因此可以确认**本次上层退出没有完整清理worker，当前多层启动链缺少兜底清理**。合理推断是MPI/节点shell退出与TorchElastic的子进程回收发生竞态或信号未完整转发；部分rank先退出后，其余rank可能停在GPU/分布式等待。没有用户Ctrl+C时最外层MPI/agent日志或信号跟踪，不能确定是PRTE先强杀agent、shell未转发、重复Ctrl+C，还是其它路径首先触发，不能把具体顺序写成已经复现。

与直接Bridge/torchrun相比，这条链增加了MPI管理的节点shell和tee管道，训练worker的退出不能只依赖终端信号；旧Birds MPI入口也有类似多层结构，所以用户历史没有遇到不等于该结构绝不会残留。该问题与本次取消独立预检没有直接信号处理代码联系。

机制参考：[Open MPI信号处理](https://docs.open-mpi.org/en/main/man-openmpi/man1/mpirun.1.html)、[Python信号处理](https://docs.python.org/3/library/signal.html)。前者描述MPI尝试终止所管理任务，不保证任意另建会话的后代都被操作系统自动递归终止；后者说明长时间C调用可能延迟Python处理信号。此处库源码证据优先于对历史信号顺序的猜测。

### 本次清理结果及后续修复方向

使用临时清理工具`/tmp/qwen38_run02_cleanup.py`：逐PID读取命令及SAVE_DIR精确确认，记录进程启动ticks，发送信号前复核身份，先TERM、8秒后才允许KILL兜底；不是pkill所有python或整容器清理。两端10个worker全部TERM后立即退出，**未使用KILL**，清理结果中remaining均为空。

原始身份/信号/结果：[cleanup_n05.json](../../../runs/qwen38_text4096_run02_review_20261116/cleanup_n05.json)、[cleanup_n06.json](../../../runs/qwen38_text4096_run02_review_20261116/cleanup_n06.json)。随后再次核对两容器均无本worker进程；n05八卡利用率/显存为0，n06 GPU6/7显存为0，所有卡利用率为0。n06 GPU0–3另有约5%显存占用，未将其当成本次worker继续清理。

持久修复方向：在节点launcher引入有明确PID/启动身份记录的训练监督，处理INT/TERM/EXIT，先要求agent清理，再给足其回收时间，最后仅对本次运行的子进程组兜底；必要时MPI主进程在退出后通过SSH做逐节点确认。应以CPU假worker模拟独立session、阻塞及信号，验证Ctrl+C、TERM、worker失败、正常退出和不误杀其它运行，再验证真实双节点短测。不能只给父shell加一个kill子PID的trap就假定覆盖了另建session的孙进程。本次用户要求为分析和清理，尚未实施这一生产代码修复。

本节及当前状态为本次handoff修改位置；新增上述3个诊断JSON，未修改训练源码，未运行GPU训练或单元测试。

## 2026-11-16：对照2.4T旧入口，核实run01/run02数据一致性

本节进一步回答用户关于旧`train_qwen38_2.4t_pretrain.sh`的Ctrl+C表现以及预检是否筛选数据的问题。仅补充本handoff，未修改训练代码或提交Git。

### 旧入口与当前入口的具体信号链差异

实际读取原工作树`/public/home/yugeng/work/Megatron-bridge-das/examples/qwen/train_qwen38_2.4t_pretrain.sh`，review树同名脚本亦核对。**旧脚本多节点模式也使用MPI**，不能把本次差异简单写成“旧Bridge没有MPI、现在多了MPI”。

旧脚本多节点分支（约第240–251行）：

```text
主节点：bash → mpirun | tee
远端：  bash -c 'source vendor && source env && torchrun ...'
```

当前文本入口（`qwen38_sft_mpi.sh`末尾及`train_qwen38_text_sft.sh`第61–63行）：

```text
主节点：exec mpirun
每节点：bash train_qwen38_text_sft.sh → python -m torch.distributed.run ... | tee
```

关键区别是**tee管道从MPI外层移到了各节点torchrun这一层**。旧`bash -c`末尾是单条外部命令，Bash可以优化为用该命令替换自身；当前末尾是管道，shell需要管理torchrun与tee两个子进程，不能同样直接替换成torchrun。用CPU小进程实验（source空环境后运行短暂Python sleep，对照Python sleep管道cat）观察到：旧形式Popen的PID直接成为Python；新形式同一PID仍为Bash，并有Python/cat两个子PID。该实验只验证shell结构，不模拟完整GPU/PRTE信号竞态。

在旧形式发生末尾命令替换时，MPI直接管理的进程可以是torchrun agent；在当前形式中直接管理的是节点shell，worker退出更依赖shell→agent→独立worker会话的转发/回收。`torchrun`命令与`python -m torch.distributed.run`本身属于同一类TorchElastic入口，不能仅用这两个名字的差别解释残留；`exec mpirun`也不等于禁止信号传递。

如果旧实验实际是NNODES=1，它根本不经过MPI，其`torchrun | tee`在本地前台运行，Ctrl+C到agent的路径又比当前双节点短。未取得用户旧次运行的NNODES、agent版本及信号日志，不能断言历史始终是同一种进程树或完全重建当次退出过程。已确认的是当前两节点残留worker的PPID=1，缺少完整回收；旧入口的结构差异支持解释为何过去更容易正常停止，但不是具体中断顺序的严格复现。

### 两次是否使用同一数据，预检是否筛选

- 两份`resolved_config.yaml`逐行比较，只有checkpoint.save、logger.save_config_filepath、logger.tensorboard_dir这3个输出路径不同；其余内容一致。两次均指向`/public/home/yugeng/work/datasets/ultrachat-qwen38-train16384-val256-seq4096-disabled`，dataset seed1234、训练RNG seed5678、SEQ4096、padding=true、MBS1/GBS32、NUM_WORKERS=0，HF初始权重路径一致。
- 当前training.jsonl的SHA256=`bc09f3ababc420402db2c1b184a24284b073a66f91c763b1a2403239e4cd8fd4`，validation.jsonl的SHA256=`6748833b76f42ed91c5e01b56178541a56ee1f18bda4dd0138be59f754f8d791`，均与run01预检manifest相同；两份文件mtime均早于两次训练。run02取消预检后没有独立数据hash快照，因此这是当前文件对历史run01的比对，不能冒充run02单独采集的hash证据。
- 两次共同的step1–16，step编号/loss/grad norm三元组逐项一致（日志打印精度）。配置、数据内容比对及实际训练轨迹共同支持：**两次使用同一批prepare后的数据，没有run01启动时额外筛选而run02绕过筛选的情况**。
- `check_qwen38_text_sft.py`以只读方式遍历JSONL，校验失败时`raise ValueError`终止启动，不是`continue`后发布一个更小训练集；检查器不写回training/validation，不向trainer提供过滤列表或复用分词结果。它在独立进程结束后，torchrun才重新创建训练Dataset。
- 真正的筛选发生在此前独立`prepare`阶段：data_manifest记录training读取16394条、排除3条invalid和7条token_filtered，输出16384条；validation输出256条。这个已准备目录为两次共同使用。取消训练前预检不会把此前过滤的10条原始记录重新加回JSONL。

### 第11步后退化是否与预检有关

没有证据证明直接相关。预检先退出才启动训练，不在step11继续运行，不改变训练行数、顺序或batch配置；两次前16步数值轨迹也一致，所以“run01多筛了一批数据，导致后面样本变重”不符合代码和日志。

从run01到run02，同时存在预检移除、缓存已热和运行时间/资源状态变化，不能把后续步耗时稳定全部归功于移除预检。启动扫描影响CPU/文件系统页缓存等间接状态的可能性不能凭现有日志绝对排除，但没有当时资源采样可证明因果，也没有第11步触发预检的代码。当前准确结论仍是：预检移除解释训练前等待缩短；缓存复用支持首步明显缩短；run01第11步后的性能退化未定位、run02未复现。

## 2026-11-16：恢复旧式MPI启动结构，补充8192性能测试操作

用户要求参照旧`train_qwen38_2.4t_pretrain.sh`调整启动方式。本次复用其“主节点mpirun外层tee、每节点直接运行torchrun”的组织方式；当前节点路径显式exec agent，避免依赖Bash自动优化。保留已修复的PRTE临时路径、hostfile slots=1、SSH11463及IB配置，不恢复自动预检。

### 实际修改位置

| 文件 | 修改 |
| --- | --- |
| `examples/qwen/qwen38_sft_mpi.sh::qwen38_launch_mpi` | 创建train_logs；将`exec mpirun ...`改为`mpirun ... 2>&1 \| tee mpi_日期-时分.log`，保留pipefail下的失败退出码；MPI结束后退出协调脚本，避免意外继续落入本地torchrun分支 |
| `examples/qwen/train_qwen38_text_sft.sh`末尾 | MPI rank分支使用`exec PYTHON_BIN -m torch.distributed.run ...`，没有节点内tee；单节点及显式NODE_RANK模式保留本地node-N.log，dryrun保持既有行为 |
| `examples/qwen/run_qwen38_vl_sft_full.sh`末尾 | 因共用MPI公共函数，同步让MPI节点exec agent，避免VL残留旧节点管道；不改VL recipe/数据/训练目标 |
| `tests/unit_tests/qwen38_text_sft/test_full_workflow.py::LauncherTests` | 新增MPI统一日志、MPI失败码、text/VL节点exec保留PID、8192参数及SAVE_INTERVAL=0透传回归 |
| `docs/models/qwen/qwen38-sft-long-sequence-and-vl.md` | 更新已验证状态、日志位置、当前hostfile和取消自动预检的说明，新增完整8192准备/性能命令 |

当前长序列包装已有默认值GBS64/RECOMPUTE=none，本次发现并保留，没有将用户已有设置改回历史GBS32/full；下面的首测命令显式覆盖。训练数据、HF初始化、MTP mask及缓存布局均未改。

### 验证证据与边界

- `test_full_workflow.py` 16项、`test_tools.py` 16项、`test_worker_exit.py` 2项，共34项通过。3个修改shell脚本的`bash -n`通过。文本自动预检仍不执行，手工preflight仍有效。
- 真实n05+n06容器SSH/MPI纯CPU探针：[agent_probe.py](../../../runs/qwen38_old_style_launcher_20261116/agent_probe.py)、[validation.json](../../../runs/qwen38_old_style_launcher_20261116/validation.json)。通过真实长序列入口传SEQ8192/GBS32/full，但PYTHON_BIN替换为标准库假agent，不导入Torch，不读取模型/数据，不占GPU。
- 正常退出：主launcher exit0，两端子进程均正常回收。模拟一次Ctrl+C（向前台会话进程组发送SIGINT）：主launcher exit130，两端agent均收到SIGTERM，各自独立session子进程均被TERM并wait回收。记录中agent父命令直接为prterun/prted，不是bash。
- 此结果验证启动结构和CPU信号路径，不证明真实TorchElastic在GPU/RCCL阻塞时一定在同样时间内退出。未执行8K GPU训练或新的真实GPU中断验收，不将CPU结果冒充GPU验收。

### SEQ8192数据准备及性能短测

检查时默认8K数据目录尚不存在。以下为在n05既有`yg_qwen38`容器中执行的用户操作步骤，本次没有代为prepare或启动GPU训练。

```bash
cd /public/home/yugeng/work/Megatron-bridge-das-qwen38-sft-review
unset NODE_RANK MASTER_ADDR RESUME_FROM
DATA8=/public/home/yugeng/work/datasets/ultrachat-qwen38-train16384-val256-seq8192-disabled

# 只准备一次；输出目录已存在时工具拒绝覆盖。
SEQ_LENGTH=8192 THINKING_MODE=disabled DATA_PATH="$DATA8" \
  LOCAL_HF_PATH=/public/home/yugeng/work/models/Qwen3.8-27B-text-64layers \
  TRAIN_SAMPLES=16384 VALIDATION_SAMPLES=256 MTP_NUM_LAYERS=0 \
  bash examples/qwen/run_qwen38_text_sft_long.sh prepare

# 保持TP4/PP2/DP2、MBS1、GBS32、full重计算，无MTP；50步自然结束。
RUN8="$PWD/outputs/text8192-perf-$(date +%Y%m%d-%H%M%S)"
SEQ_LENGTH=8192 THINKING_MODE=disabled DATA_PATH="$DATA8" SAVE_DIR="$RUN8" \
  LOCAL_HF_PATH=/public/home/yugeng/work/models/Qwen3.8-27B-text-64layers \
  TP=4 PP=2 MTP_NUM_LAYERS=0 GLOBAL_BATCH_SIZE=32 RECOMPUTE=full \
  TRAIN_ITERS=50 WARMUP_ITERS=2 EXIT_INTERVAL=0 EVAL_ITERS=0 SAVE_INTERVAL=0 \
  LOG_THROUGHPUT=true MASTER_PORT=30122 \
  bash examples/qwen/run_qwen38_text_sft_long.sh sft
```

hostfile当前n05+n06，各8卡，只在一个容器终端启动一次。保持两节点资源可用、单独运行。prepare从原始UltraChat Parquet保留官方train_sft/test_sft，按8K完整对话长度及assistant监督过滤，发布16384/256条JSONL及data_manifest/token审计；不改HF权重、RoPE或模型转换。prepare的过滤与已取消的“每次训练启动重复预检”是两件事。

性能短测`EVAL_ITERS=0`排除验证，`SAVE_INTERVAL=0`利用当前Bridge `train.py`中的常规/正常最终保存判断同时关闭两者，`EXIT_INTERVAL=0`避免主动退出保存路径；训练仍执行真实前反向及optimizer更新。其目的是性能比较，不验收checkpoint恢复，不需要用Ctrl+C提前终止。不要仅把SAVE_INTERVAL设得大于50就以为末步不保存，当前框架正常结束会补最终checkpoint。

先检查真实8K能否容纳，再统计step11–50（若仍未稳定则延长测试）的平均、P50、P95和两节点显存，确认loss/grad有限、无NaN/skip。新8K形状可能首次编译；同配置复测只换RUN8，保留现有缓存根目录。不能根据4K步耗时简单乘2预测结果。GBS32/16GPU时，padding token/s/GPU=`32×8192÷16÷平均步秒数=16384÷平均步秒数`，samples/s=`32÷平均步秒数`；有效文本/监督token吞吐需另外统计。

若目的仅为与run02做严格相同样本的固定4K/8K形状性能对照，可直接将上面训练的DATA_PATH设为已有seq4096目录，此时不需新prepare，但8K多出来的部分是padding，不代表真实长对话训练。新prepare的8K数据会纳入更多长样本，不能声明与run02逐样本相同。

本轮未提交Git。当前修改已具体列于上表，历史run01/run02日志和诊断证据未改。

## 2026-11-16：文本入口整合，文本/VL独立启动

按用户要求，当前SFT shell入口简化为两个，各自独立维护，不再设置公共MPI入口。文本直接使用原生算子，其余已支持的训练参数通过环境变量设置。本次未提交Git。

### 修改与删除清单

- 新增`examples/qwen/run_qwen38_text_sft.sh`：合并文本模型/数据默认值、prepare/convert工具调用、手工preflight、环境/缓存设置、hostfile/MPI、torchrun。按旧pretrain脚本的分区组织配置与执行逻辑，不再逐层跳转shell包装。
- 重构`examples/qwen/run_qwen38_vl_sft_full.sh`：内置自己的环境、缓存、hostfile/MPI设置；不source文本入口或任何共用MPI/runtime文件。保留VL独立的训练参数配置。
- 删除7个冗余脚本：`run_qwen38_text_sft_full.sh`、`run_qwen38_text_sft_long.sh`、`run_qwen38_text_sft_native.sh`、`run_qwen38_text_sft_single_node.sh`、`train_qwen38_text_sft.sh`、`qwen38_sft_mpi.sh`、`qwen38_sft_runtime.sh`。没有保留同名转发包装；后续命令直接使用新文本入口。仓库中其它旧模型/pretrain实验脚本未删除。
- 修改`tests/unit_tests/qwen38_text_sft/test_full_workflow.py`和`test_tools.py`：迁移新入口，验证MTP/拓扑/手工preflight及两入口互相独立；不因删除单节点包装丢失NNODES=1或显式NODE_RANK方式。
- 更新本文当前入口/操作链接、长序列/VL指南、全层文本指南、UltraChat历史减层指南。历史handoff章节保留当时真实脚本名及证据，不把旧路径作为当前可执行入口。

源码变更便于逐项查看：[launchers.patch](../../../runs/qwen38_launcher_consolidation_20261116/launchers.patch)。该patch对照整合前8个启动/helper文件及新文本入口，记录新增、改写与删除；不代表其它既有未提交工作树的完整diff。

### 统一文本入口的行为和默认值

| 项目 | 当前行为/默认值 |
| --- | --- |
| 无参数 | dryrun；显式`sft`才启动训练 |
| 模式 | `convert \| prepare \| preflight \| dryrun \| sft`，prepare仍按数据准备策略过滤；训练不调用自动checker |
| 算子 | 直接使用框架原生TE RMSNorm和FusedAdam |
| 序列/模型 | SEQ4096，thinking disabled，64层无MTP文本HF；`MTP_NUM_LAYERS=1`默认选择mtp1目录。自定义减层模型显式设置EXPECTED_NUM_LAYERS和LOCAL_HF_PATH |
| 数据准备 | TRAIN_SAMPLES16384、VALIDATION_SAMPLES256。默认DATA_PATH按样本数、SEQ、thinking生成；SEQ1024且默认数量/disabled时复用现有steady数据目录。指定DATA_PATH始终优先 |
| 训练 | GBS64、TRAIN_ITERS500、WARMUP_ITERS20、LR1e-5、RECOMPUTE=none，保留此前长序列入口的已有设置；均可通过环境变量覆盖 |
| recipe限制 | MBS1、CP1仍是现有文本recipe支持边界，本轮未扩展recipe；不能把任意环境变量误当作新增支持 |
| 验证/保存 | EVAL_ITERS8/EVAL_INTERVAL100、SAVE_INTERVAL250；性能短测可显式EVAL_ITERS=0 SAVE_INTERVAL=0 EXIT_INTERVAL=0 |
| 并行 | GPUS8、TP4/PP2；NNODES按hostfile推导，默认hostfile当前n05+n06。单节点用NNODES=1；手工逐节点保留NODE_RANK方式 |
| 输出 | 默认`outputs/qwen38-text-<backend>-seq<SEQ>/<时间戳>`，可设SAVE_DIR；MPI主节点统一`train_logs/mpi_日期-时分.log`，节点exec agent，保留失败码；单节点/手工rank写node-N.log |
| 缓存/通信 | 默认仍是`outputs/qwen38-sft-cache`，已有缓存未移动或清理；两脚本各自可设置QWEN38_CACHE。保留PRTE home临时目录、SSH11463、slots=1及IB插件强制策略 |

### 8192数据与新的调用命令

本轮检查时已看到用户的8K准备目录正式发布，未发现仍运行的prepare进程。仅只读核对`data_manifest.json`，没有重新prepare、改写或全量扫描数据：SEQ8192/thinking disabled、training16384/validation256；training read16387、invalid3、token_filtered0，validation无过滤。此信息是准备manifest记录，不是新GPU验收。

已准备目录仍为`/public/home/yugeng/work/datasets/ultrachat-qwen38-train16384-val256-seq8192-disabled`。无需因入口改名重新生成。后续在n05的原容器中：

```bash
cd /public/home/yugeng/work/Megatron-bridge-das-qwen38-sft-review
unset NODE_RANK MASTER_ADDR RESUME_FROM

# 原生算子；显式控制性能短测配置，50步自然结束。
SEQ_LENGTH=8192 GLOBAL_BATCH_SIZE=32 RECOMPUTE=full \
  TRAIN_ITERS=50 WARMUP_ITERS=2 EVAL_ITERS=0 SAVE_INTERVAL=0 EXIT_INTERVAL=0 \
  SAVE_DIR="$PWD/outputs/text8192-native-perf-$(date +%Y%m%d-%H%M%S)" \
  bash examples/qwen/run_qwen38_text_sft.sh sft

# 新数据准备仍是：SEQ_LENGTH=8192 DATA_PATH=/新的输出目录 bash examples/qwen/run_qwen38_text_sft.sh prepare
```

显式DATA_PATH、LOCAL_HF_PATH、HOSTFILE、TP/PP、LR等继承用户环境时仍优先于默认值。当前只使用框架原生算子；历史run的完整运行配置以各自日志和审计文件为准。

### 验证与边界

- `test_full_workflow.py` 18项、`test_tools.py` 16项、`test_worker_exit.py` 2项，共36项通过；两个保留shell的`bash -n`通过。初次工具回归暴露独立checker测试未设置项目PYTHONPATH，已修正测试调用环境并重跑16项通过，未借此更改生产checker逻辑。
- 独立性回归在只有两个SFT脚本、无公共helper的临时工作树中运行；分别删除另一入口后，文本/VL仍可正确派发MPI。验证prepare/convert仍为本地调用，sft/dryrun仍不预检，8192/GBS32/full/SAVE_INTERVAL=0能传到MPI。
- 真实n05+n06 SSH/MPI CPU探针分别覆盖text-normal、text-interrupt、vl-normal、vl-interrupt；正常均exit0，一次模拟Ctrl+C均exit130，两端假agent收到TERM并wait回收独立session子进程。agent父进程均直接为prterun/prted。
- 证据：[agent_probe.py](../../../runs/qwen38_launcher_consolidation_20261116/agent_probe.py)、[validation.json](../../../runs/qwen38_launcher_consolidation_20261116/validation.json)。探针仅使用标准库假agent/worker，不导入Torch或读取模型/训练数据。未执行GPU训练或重新准备数据，不将CPU信号验证当作GPU内核或RCCL阻塞退出验收。

## 2026-11-16：恢复single_node、三个入口增加profiler及run03 OOM诊断

用户指定日志目录`outputs/text4096-run03`。本次恢复单节点配置包装，为三个入口增加profiler参数，分析两次OOM并清理已确认的n06残留。未提交Git，未修改Bridge/MCore源码。

### 代码修改位置及single_node默认配置

- 恢复`examples/qwen/run_qwen38_text_sft_single_node.sh`，作为显式单节点配置包装，末尾exec统一文本入口。强制NNODES=1/NODE_RANK=0并清除继承的OMPI rank，默认16层HF、EXPECTED_NUM_LAYERS16、SEQ1024、8GPU、TP4/PP2/DP1、GBS8/MBS1、native、full重计算、50步/warmup2、EVAL_ITERS0/SAVE_INTERVAL0/EXIT_INTERVAL0。
- 保留原single_node用途为**16层链路/性能实验**，不是完整27B训练。用64层必须同时显式设置EXPECTED_NUM_LAYERS=64和对应LOCAL_HF_PATH，并重新做显存验证；本次未自动减层替代用户请求的完整模型。单节点64层TP4/PP2/DP1的稳态状态估算约56.36GiB/卡，不能因为脚本支持启动就认定4K/8K可容纳。
- 修改`run_qwen38_text_sft.sh`和`run_qwen38_vl_sft_full.sh`的Torch Profiler配置区、各自MPI透传列表、Worker Launch参数。single_node暴露同名参数并交由统一文本入口解析。无公共MPI模块重新引入。
- 本轮读取时统一文本入口默认RECOMPUTE已为full，保留这一用户已有改动；没有按上一轮记录的none值覆盖回去。
- 修改`tests/unit_tests/qwen38_text_sft/test_full_workflow.py`：single_node配置及强制本地行为、三个入口profile参数真实argv、两套MPI透传回归。

### Profiler使用方式

| 环境变量 | 默认 | 说明 |
| --- | --- | --- |
| PROFILE | false | true/1开启PyTorch profiler；false/0关闭；不启用nsys |
| PROFILE_STEP_START / PROFILE_STEP_END | 10 / 11 | end为排除边界，默认1个active step；TRAIN_ITERS应覆盖完整窗口 |
| PROFILE_RANKS | `[0]` | 全局rank列表；双节点TP4/PP2可用`[0,8]`观察两个流水阶段，或`[0,4,8,12]`覆盖DP副本 |
| PROFILE_RECORD_SHAPES / PROFILE_WITH_STACK | false / false | 需要形状/调用栈时开启，会增加记录开销 |

复用现有Bridge `training/profiling.py`机制，runner覆写`profiling.use_pytorch_profiler`、`profile_step_start/end`、`profile_ranks`、`pytorch_profiler_collect_shapes/callstack`，输出为`SAVE_DIR/torch_profile/rank-<global_rank>.json.gz`。这是Chrome trace压缩文件，可解压后用Perfetto等读取；不是checkpoint。profile窗口会影响时间与内存，不将窗口内吞吐当作无profile基线。

```bash
cd /public/home/yugeng/work/Megatron-bridge-das-qwen38-sft-review
unset NODE_RANK MASTER_ADDR RESUME_FROM

# 完整64层双节点文本，20步内抓取2个active step。
PROFILE=true PROFILE_STEP_START=10 PROFILE_STEP_END=12 PROFILE_RANKS='[0,8]' \
  SEQ_LENGTH=4096 GLOBAL_BATCH_SIZE=32 RECOMPUTE=full \
  TRAIN_ITERS=20 WARMUP_ITERS=2 EVAL_ITERS=0 SAVE_INTERVAL=0 EXIT_INTERVAL=0 \
  SAVE_DIR="$PWD/outputs/text4096-prof-$(date +%Y%m%d-%H%M%S)" \
  bash examples/qwen/run_qwen38_text_sft.sh sft

# 单节点默认16层；只在该节点执行。
PROFILE=true PROFILE_STEP_START=10 PROFILE_STEP_END=12 PROFILE_RANKS='[0,4]' \
  TRAIN_ITERS=20 SAVE_DIR="$PWD/outputs/single16l-prof-$(date +%Y%m%d-%H%M%S)" \
  bash examples/qwen/run_qwen38_text_sft_single_node.sh sft

# 完整VL使用自己的入口和参数，不与上面并行占用同一节点。
PROFILE=true PROFILE_STEP_START=10 PROFILE_STEP_END=12 PROFILE_RANKS='[0,8]' \
  TRAIN_ITERS=20 WARMUP_ITERS=2 SAVE_INTERVAL=0 \
  SAVE_DIR="$PWD/outputs/vl-prof-$(date +%Y%m%d-%H%M%S)" \
  bash examples/qwen/run_qwen38_vl_sft_full.sh sft
```

### run03为什么关闭重计算仍OOM：不要混用两次日志的配置

该输出目录被连续用于两次实验，`resolved_config.yaml`及model_rank等sidecar已被后一次写入；不能只看当前YAML的full就认为第一次也是full。

| 日志 | 配置与失败阶段 | 结论 |
| --- | --- | --- |
| `mpi_2026-11-16-2200.log` | SEQ4096、GBS64/MBS1、TP4/PP2/DP2、native、recompute_granularity=null；首次训练前向中的TE GEMM/MLP GLU等位置OOM。典型PyTorch allocated约56.16–56.27GiB、未用reserve约0.77–0.79GiB，GPU剩余约0.17–0.54GiB | 关闭重计算需要保存更多语言模型中间激活，实际已逼近容量；不能以去掉视觉塔推断一定够用 |
| `mpi_2026-11-16-2216.log` | GBS64/full重计算，rank14/15在DDP参数/梯度buffer构建阶段失败；新进程PyTorch allocated仅6.33GiB，却只剩2.44/2.50GiB，申请6.26GiB失败 | 当时GPU6/7仍被22:00那次残留worker占约85%；不是full重计算已被证明无效，也不是本次新进程自己已经用满整卡 |

逐日志配置和OOM行见[oom_evidence.json](../../../runs/qwen38_run03_oom_prof_20261116/oom_evidence.json)。第一次报错即使小额34/68MiB也可能失败，错误信息中的free/reserve并不能证明所有空闲空间都可满足当时申请；无连续性/碎片及vendor workspace剖析，不能直接将allocator参数作为根治。

按源模型safetensors header实际shape计数：语言26,895,998,464参数、视觉460,730,096参数、MTP424,699,392参数。视觉约占源总参数1.7%，去掉它并未去掉26.896B语言主干；语言模型仍有64层、宽embedding/head、GDN和MLP。

以TP4/PP2/DP2估算，每GPU本地参数约3.36B，BF16模型参数及FP32梯度不由DP整体均分，FP32 master/Adam状态由DP分片。理论稳态常驻量如下：

| 状态 | 约GiB/GPU |
| --- | ---: |
| BF16权重 | 6.26 |
| FP32梯度buffer | 12.53 |
| FP32 master参数（DP2分片） | 6.26 |
| FP32 Adam m/v（DP2分片） | 12.53 |
| 合计 | 37.58 |

这还未计激活、通信buffer、GEMM/attention/GDN临时内存、非PyTorch库分配、allocator预留。估算公式为本地参数×`(6+12/DP)`，而不是总参数×2字节÷16GPU；DP2复制两个模型副本，并非把语言模型直接切成16份。估算是稳态模型，不证明首个前向时Adam状态已全部初始化，不能直接用56GiB减37.58GiB精确推算激活。

估算见[memory_estimates.json](../../../runs/qwen38_run03_oom_prof_20261116/memory_estimates.json)。重计算主要减少需长期保留的激活，不会消除上述参数/梯度/Adam状态；见[PyTorch说明](https://docs.pytorch.org/docs/stable/checkpoint)。MBS固定1时，将GBS64降到32主要将梯度累积microbatch从32降至16，不应声称会把1F1B流水线的激活峰值直接减半。优先在干净GPU上用4K/full/GBS32重建基线，再评估8K或其它拓扑/offload；不同时改变后端、重计算和拓扑后只按一项解释结果。

### n06两个进程为何未释放：本次有异常时序与CPU复现

本次残留为PID3216623/3216624，global rank14/15、local rank6/7；命令为本项目text worker、SAVE_DIR精确为run03、PPID=1、PGID=SID=自身PID，wchan为`kfd_wait_on_events`。SIGTERM未被忽略/屏蔽，本次直接TERM即正常消失，不是必须重启GPU才能释放。

真实日志：n05 agent在22:07:19报告本地OOM worker失败；MPI随后报告非零退出并终止其它节点。n06 agent在22:07:24.223收到TERM，24.225–24.446只记录向PID3216617–3216622（前6个worker）发送TERM；末尾3216623/24没有相应记录，且正是30分钟后仍存活的两个。agent此时已退出，未完成全部子进程回收。

本机PRRTE版本3.0.5rc2。邻近发布版[PRRTE v3.0.5源码](https://github.com/openpmix/prrte/blob/v3.0.5/src/mca/odls/base/odls_base_default_fns.c#L1654)中终止序列使用约250ms等待后发KILL；TorchElastic则按worker逐个发信号，之后最多等待30秒。结合日志，最有力的机制是MPI的agent终止窗口短于其GPU worker清理时间，agent在发完最后两个信号前被终止；独立session worker因此不随agent进程组一起消失。引用是邻近版本源码机制，不是对本机二进制逐条指令跟踪。

为验证该机制，使用实际n05+n06 MPI跑纯CPU探针：n05假agent非零退出，n06假agent管理8个独立session子进程，转发每个TERM间隔70ms。结果只转发前4个就中断，最后4个成为残留，agent完成标记未写入；探针随后按命令+SAVE_DIR精确清理了这4个CPU子进程。这复现了“上层已直接管理agent，仍会清理一半就被MPI终止”的竞态，说明此前快速单子进程/正常Ctrl+C探针不足以代表OOM场景。

证据：[mpi_abort_probe.py](../../../runs/qwen38_run03_oom_prof_20261116/mpi_abort_probe.py)、[mpi_abort_validation.json](../../../runs/qwen38_run03_oom_prof_20261116/mpi_abort_validation.json)。探针还记录了profile配置已传到真实远端agent argv；不使用Torch/GPU，不重演真实显存OOM。

### 清理结果、验证与剩余事项

- 已精确核验run03身份后对PID3216623/24发送TERM，全部退出，无需KILL；[cleanup_n06.json](../../../runs/qwen38_run03_oom_prof_20261116/cleanup_n06.json)记录启动ticks/信号状态/结果，remaining为空。复查GPU6/7显存及利用率归零；n06 GPU0–3另有约5%占用，未清理其它任务。
- 生产启动器本轮未添加跨节点退出兜底。要持久修复，需要给agent完整回收机会，或在MPI退出后用run级身份记录进行独立兜底；不能再将“改成旧脚本结构”或快速CPU测试通过称为彻底修复。此项尚未实现，本轮完成用户要求的定位与现存进程清理。
- `test_full_workflow.py` 21项、`test_tools.py` 16项、`test_worker_exit.py` 2项，共39项通过；三个shell语法通过。未执行新的GPU训练或生成GPU trace，profile文件生成依赖后续实际跑过采样窗口。
- 真实Bridge配置dryrun：single_node解析出16层/full、profile start5/end7/ranks[0,4]，VL解析出profile start5/end7/ranks[0,8]；两者use_pytorch_profiler=true、shapes/stack=false，均exit0，单节点worker退出标记success=true。只解析配置，不构建训练模型或生成trace。[验证汇总](../../../runs/qwen38_run03_oom_prof_20261116/validation_summary.json)、[single配置](../../../runs/qwen38_run03_oom_prof_20261116/single_dryrun/resolved_config.yaml)、[VL配置](../../../runs/qwen38_run03_oom_prof_20261116/vl_dryrun/resolved_config.yaml)。

## 2026-11-16：视觉27层参数解释、TP2/PP4缓存错误与隔离修正

用户指出视觉亦有27层、常驻状态估算仍剩20余GiB，并提供`outputs/text1024-TP2PP4-run01`。本次核对真实配置/权重header，细化显存解释，定位初始化缓存错误并修改worker缓存路径。未提交Git，未代跑GPU训练。

### 为什么视觉27层仅约1.7%参数，不能用层数代替宽度

依据原HF `config.json`及safetensors真实key/shape累加：[analysis.json](../../../runs/qwen38_tp2pp4_cache_review_20261116/analysis.json)。

| 项目 | 视觉塔 | 语言主干 |
| --- | ---: | ---: |
| 层数 | 27 | 64 |
| hidden_size | 1152 | 5120 |
| intermediate_size | 4304 | 17408 |
| 单层参数 | 15,239,504 | GDN层383,273,184；full attention层372,255,232 |
| 所有block参数 | 411,466,608 | 24,353,196,544 |
| 其它参数 | 49,263,488（包含merger等） | 2,542,801,920（包含embedding/head等） |
| 合计 | 460,730,096 | 26,895,998,464 |

源MTP另有424,699,392参数，因此视觉占完整源tensor参数约1.658%。投影矩阵参数取决于输入/输出宽度；例如视觉block主要矩阵约`4×1152² + 2×1152×4304 = 15.225M`，加norm/bias为真实15.240M；语言层仅SwiGLU MLP就有`3×5120×17408 = 267.387M`参数，尚未算attention/GDN。因此“27层”并不意味着接近27个语言层的规模。

用户关于视觉激活和优化器也会消失的判断正确：去掉视觉确实同时去掉它们。**参数占比1.7%不等于视觉训练显存或显存节省比例只有1.7%**，视觉激活还由图像数量、分辨率/patch token数、是否冻结、并行切分和重计算决定。不能只用参数占比估计删除视觉后的全部峰值收益，也不能与不同样本/长度/训练设置的VL历史实验直接比较。

### 为什么37.6GiB常驻之外的26.4GiB仍可能不够

63.98GiB卡容量减37.58GiB理论稳态常驻，约剩26.40GiB，需要同时容纳语言激活、反向保存值、流水在途microbatch、通信/算子工作区、非PyTorch库分配及allocator预留；不是26.4GiB全部可用于某一种激活。此估算不包含上述项，也不代表首步Adam状态已全部初始化。

具体量级：SEQ4096/MBS1/TP4下，一个MLP FC1输出形状对应`4096×(2×17408/4)`个BF16元素，约68MiB；门控后的同类张量约34MiB。32个本地层中会出现多种需保留的张量，GDN还包括QKV、归一化、卷积、chunk状态及反向中间量；PP流水可能同时保留多个microbatch的前向结果。不能只按一个hidden_state张量估算全部激活，亦不能简单用层数乘这两种张量算出完整峰值。

run03无重计算实际已报PyTorch allocated约56GiB，并且余量很小，这是“当次完整开销超过可用容量”的直接证据；日志的34/68MiB申请与上述量级相符，但不是完整内存剖析。此前解释说明了可能来源，没有测得每种激活/工作区的精确占比；要回答26.4GiB具体花在哪，仍需在对应配置下做allocator snapshot/算子内存剖析，而不能把估算写成已测分项。

另一个拓扑细节：同为16卡/DP2，TP2/PP4的最重stage状态估算为41.13GiB（中间stage约34.02），并非必然低于TP4/PP2的37.58。TP由4变2后，首/末stage的巨大embedding/head每卡分片变大，PP4并未把它们分散到中间stage；且更多PP阶段可能增加在途microbatch数。换PP不能机械理解为显存按层数等比例下降。这里只说明估算边界，本次TP2/PP4错误并未到显存容量验证阶段。

### TP2/PP4这次实际错误

日志`outputs/text1024-TP2PP4-run01/train_logs/mpi_2026-11-16-2309.log`：SEQ1024、TP2/PP4/DP2、GBS64/MBS1、无MTP、无重计算，已初始化TP/PP进程组，随后在`setup → set_jit_fusion_options → _warmup_jit_function`失败，尚未构建模型/导入HF/开始训练。

- rank2在bias_swiglu JIT预热中报`InductorError: FileNotFoundError`，读取`outputs/qwen38-sft-cache/inductor/gh/cgh7...py`失败。
- rank15在bias_dropout_add JIT调优中报`FileExistsError`，将临时文件rename为`inductor/kk/...best_config`时失败。
- 后续`ChildFailedError`、其它rank的SIGTERM是这些异常引起的联动退出，不能当作原始错误。没有证据把这次归为TP2/PP4不受支持、显存OOM或历史IB配对问题；也不能由此宣布该配置可训练。

当前两节点共享的Inductor根目录原来没有host/rank隔离，16worker会并发发布相同key的Python代码/调优文件。报错所指的两个目标文件随后在n05/n06均可见，独立诊断目录的顺序rename覆盖探针也成功，排除了“路径永远不存在”或“该文件系统一概不能覆盖”的简单解释。现有证据强烈指向并发缓存发布/文件系统可见性异常，但没有隔离出具体PyTorch或文件系统内部缺陷；共享缓存并非在所有环境都必然失败。

### 本次修改及验证

- `examples/qwen/qwen38_text_sft_worker.py`在已有Triton/MIOpen隔离循环中增加TORCHINDUCTOR_CACHE_DIR；`examples/qwen/qwen38_vl_sft_worker.py`在import torch之前增加同样的Inductor隔离。实际路径变为`<QWEN38_CACHE>/inductor/<hostname>/rank-<LOCAL_RANK>`，避免本次16worker写同一目录；不是删除或关闭缓存。
- `QWEN38_CACHE`默认根目录不变，旧Inductor共享文件保留。首次使用新目录可能重新编译，以后相同节点/rank及匹配配置自动复用。多个运行若刻意共用同一节点/rank，仍需使用不同QWEN38_CACHE区分；本轮没有承诺隔离能消除所有文件系统故障。
- `tests/unit_tests/qwen38_text_sft/test_worker_exit.py`新增两worker、两host、两个local-rank的缓存路径回归，确认调用runner前已使用4个不同路径；同时维持正常/非零退出证据测试。
- worker测试3项、launcher测试21项通过。初轮launcher测试仍假定默认TP4/PP2，但用户已把当前脚本默认改为TP2/PP4；测试改为显式传入其要验证的TP4/PP2值后通过，没有覆盖用户当前默认TP2/PP4/RECOMPUTE=none设置。
- 未重新启动TP2/PP4 GPU训练，因此当前结论为已隔离易冲突路径并通过CPU回归，实际初始化/显存/训练性能仍需后续重跑验证。本轮未修改训练recipe、模型权重、数据或通信后端。

## 2026-11-17：run01/run02 TFLOPS对照、worker职责及2048 OOM测试脚本

用户要求复核两次4K运行的TFLOPS差异、详解worker作用，并提供复用1024数据的2048测试命令。本次重新读取两份配置、两端模型审计与逐步日志；没有启动训练、修改生产启动/worker代码或提交Git。新增[对照统计](../../../runs/qwen38_run01_run02_tflops_20261117/comparison.json)与[2048测试脚本](../../../runs/qwen38_run01_run02_tflops_20261117/test_seq2048_from1024.sh)，本handoff同步记录。

### TFLOPS差异：保存的训练设置相同，分母耗时不同

逐行diff两份resolved_config.yaml，只有checkpoint.save、logger.save_config_filepath、logger.tensorboard_dir三处输出目录不同。均为64层文本HF、SEQ4096、GBS32/MBS1、TP4/PP2/DP2、full重计算、MTP关闭、BF16/FP32 Adam、NUM_WORKERS0、相同seed/data路径。model_rank_0/8证据确认节点均为n05+n06，本节只比较这两个历史run，完整算子设置保留在各自model_rank记录中，不能套用当前入口的TP2PP4/GBS64等默认值。

`training/utils/train_utils.py`使用该日志区间的模型估算FLOPs增量除以区间耗时、world_size和1e12，得到MODEL TFLOP/s/GPU。它不是硬件指令计数，也不是GPU利用率百分比。正常步骤中，日志TFLOPS×步耗时均近似为1286 TFLOP/GPU/step，说明两次日志的估算工作量没有明显变化；显示只有一位小数，尤其首步2.1会放大反算舍入误差。

| 同步比较的窗口 | run01平均步秒数 | run02平均步秒数 | run01 / run02估算TFLOP/s/GPU |
| --- | ---: | ---: | ---: |
| step1 | 624.260 | 49.082 | 2.1 / 26.2 |
| step2–5 | 23.690 | 15.266 | 54.3 / 84.3 |
| step6–10 | 16.075 | 15.270 | 80.1 / 84.2 |
| step11–16 | 37.753 | 15.193 | 34.1 / 84.7 |

多步TFLOPS为日志值按时间加权的近似（区间总估算工作量÷区间总耗时），不是简单平均每行TFLOPS。相近快窗口只有约5%的差距；首步大差异与run01冷编译、run02缓存复用一致；后半段差异与run01步耗时约2.48倍退化相对应，原始资源瓶颈仍未定位。缺少历史GPU时钟/功耗/CPU/通信采样，不把这部分归因于某个未证实开关。

run01的全量数据预检先结束、再启动torchrun，不计入training interval-time，故它解释训练前等待但不能直接解释每步TFLOPS，更没有step11重新执行预检的代码。两次前16步loss/grad norm在打印精度下一致，亦无run01筛选出另一训练集的证据。

### qwen38_text_sft_worker.py按执行顺序的作用

该文件是每个torchrun worker实际执行的适配和审计入口；训练主体仍为Bridge/MCore，缓存管理只是其中一项。

| 代码位置/阶段 | 实际作用与边界 |
| --- | --- |
| 缓存目录 | 加入项目导入路径，取得global rank/输出目录；为Inductor、Triton及MIOpen添加host/local-rank缓存子目录，且在Torch导入前完成。库负责判断cache key和复用产物，worker不手工载入编译二进制 |
| GPU绑定 | 按LOCAL_RANK绑定GPU，ROCm中仍用torch.cuda接口；直接使用框架原生RMSNorm/Adam。先绑定再导入TE/runner是已有设备问题修复 |
| 框架初始化 | 导入run_recipe安装HCU适配；只在启用MTP时安装MTP审计。无MTP的run01/run02不执行这项MTP采样 |
| 模型审计 | 包装GPTModelProvider.provide：调用真实构建后核对无视觉/冻结参数、MTP不是DeltaNet；记录模型类型、本地参数数、层号/attention类型、后端等到model_rank_N.json。只遍历参数元数据，不复制一份完整模型；不是之前13分钟的全量数据预检 |
| 权重导入审计 | 包装AutoBridge.load_hf_weights：仍由原函数执行真实导入，成功后写hf_loaded_rank_N.json；完成标记不等于逐元素证明权重正确，恢复路径若未调用HF导入则不会新生成此标记 |
| 执行与退出记录 | 调用run_recipe.main解析recipe并进入训练；返回/退出时写rank_N_final.json，记录成功、exit code、起止时间及本进程PyTorch峰值allocated/reserved。正确处理SystemExit(0)。SIGKILL或进入try之前失败时不保证产生最终标记；峰值也不等于整卡所有库/其它进程的显存 |

这些包装只在当前Python进程内替换方法，不改Bridge/MCore源码。删除worker会同时删除上述绑定和审计逻辑，不能把它理解成只为缓存复用新增的一层。模型/数据/优化器策略来自recipe和启动参数，forward/backward及分布式调度来自框架。

### 用1024数据测2048 OOM：可以，不必重新prepare

当前1024数据manifest记录SEQ1024/thinking disabled、training16384/validation256。使用相同模板/模型时，将模型与collator SEQ统一设为2048，原完整对话仍能装入；`pad_to_max_length=true`把实际训练张量补到2048，padding loss_mask=0。虽然JSONL目录名带seq1024，实际GPU处理的固定形状仍为2048，所以适合测试2048显存和固定padding吞吐。

该方式不会恢复此前被1024prepare过滤掉的长对话，不能声称验证了真实接近2K长样本的训练效果/有效token吞吐。要训练更多真实长对话，另从原Parquet prepare到2048新目录。

已提供可直接执行的`runs/qwen38_run01_run02_tflops_20261117/test_seq2048_from1024.sh`：

- 完整64层HF、既有1024数据、SEQ2048、双节点16GPU，默认保持最近实验的TP2/PP4、GBS64、native、无重计算；MTP关闭、offload关闭。
- 30步/warmup2，PROFILE=false、EVAL_ITERS0/SAVE_INTERVAL0/EXIT_INTERVAL0，结束自动退出，不测验证/保存开销；自动生成包含拓扑/重计算模式及时间的输出目录。
- 清除手工rank和恢复环境，显式固定DATA_PATH，避免继承某次prepare/16层实验的模型与数据路径。TEST_TP/TEST_PP/TEST_GBS/TEST_RECOMPUTE/TEST_ITERS/TEST_SAVE_DIR可选择试验参数。

```bash
cd /public/home/yugeng/work/Megatron-bridge-das-qwen38-sft-review

# 在n05既有容器中启动一次；复用已准备的1K数据，无重计算容量测试。
bash runs/qwen38_run01_run02_tflops_20261117/test_seq2048_from1024.sh

# 如要做full重计算对照，自动使用另一个输出目录。
TEST_RECOMPUTE=full \
  bash runs/qwen38_run01_run02_tflops_20261117/test_seq2048_from1024.sh

# 选择TP4/PP2、GBS32；不必修改生产启动脚本默认值。
TEST_TP=4 TEST_PP=2 TEST_GBS=32 \
  bash runs/qwen38_run01_run02_tflops_20261117/test_seq2048_from1024.sh
```

此脚本仍调用统一文本入口，没有新增生产launcher或公共MPI模块；`bash -n`通过，未实际执行。2048能否容纳由实测决定；发生OOM后再次测试前应确认本轮两端进程已退出，避免已知残留污染下一次显存判断。未增加或重启GPU任务。

## 2026-11-17：TP overlap性能对照（已完成）

已完成用户要求的两种拓扑测试，并为每种拓扑补充同条件关闭基线，共4组独立HF初始化、各50步。正式比较使用step26–50共25步，排除首步/预热、step20/21采样及step22导出；step11–19保留为辅助数据。四组均为native、完整64层无MTP、全参数可训练、无视觉模块，n05+n06共16GPU、DP2/CP1、GBS64/MBS1、SEQ4096、full/uniform/1重计算、BF16参数及FP32梯度/master/Adam、distributed optimizer、无CPU offload。相同4K数据、seed5678、LR1e-5/warmup2，EVAL_ITERS0/SAVE_INTERVAL0/EXIT_INTERVAL0。

证据与统计：[plan.json](../../../runs/qwen38_tp_overlap_20261117/plan.json)、[运行环境](../../../runs/qwen38_tp_overlap_20261117/environment.json)、[完整汇总](../../../runs/qwen38_tp_overlap_20261117/summary.json)、[解析日志](../../../runs/qwen38_tp_overlap_20261117/analysis.log)。plan记录数据及3个训练源文件SHA256；结束后复核全部一致。每对resolved_config只差TP overlap两处布尔值和三个输出路径，未改变其它训练设置。

### 性能结果

| 拓扑 | TP overlap | 平均步秒数 | P95秒 | 标准差秒 | samples/s | MODEL TFLOP/s/GPU |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| TP4/PP2 | 关闭 | 29.033956 | 29.52216 | 0.270493 | 2.2043 | 88.63 |
| TP4/PP2 | 开启 | 34.975920 | 35.37734 | 0.253811 | 1.8298 | 73.59 |
| TP2/PP4 | 关闭 | 25.210616 | 25.47034 | 0.159611 | 2.5386 | 102.08 |
| TP2/PP4 | 开启 | 29.666372 | 29.86842 | 0.168239 | 2.1573 | 86.74 |

- TP4/PP2：开启后步耗时增加20.47%，样本吞吐下降16.99%。
- TP2/PP4：开启后步耗时增加17.67%，样本吞吐下降15.02%。

本次工作负载下两种拓扑的TE默认TP overlap配置均为负收益，保持生产入口默认`TP_COMM_OVERLAP=false`。本次四组中TP2/PP4关闭最快，但显存更高；不据此推广至其它SEQ、batch、MTP或经过调优的UB配置。MODEL TFLOP/s/GPU为日志模型公式估算、按时间加权的近似，不是硬件计数或MFU；samples/s=64/平均步秒数。采样时频率大多约1500MHz，未人为锁频；原始利用率/频率/功耗/温度在每组resources.jsonl，汇总仅取相同正式窗口。

| 拓扑 | TP overlap | PyTorch峰值allocated GiB | PyTorch峰值reserved GiB | 正式窗口整卡VRAM采样最大值 MiB |
| --- | --- | ---: | ---: | ---: |
| TP4/PP2 | 关闭 | 40.4281 | 40.9023 | 48732 |
| TP4/PP2 | 开启 | 40.6779 | 41.1426 | 49817 |
| TP2/PP4 | 关闭 | 46.1999 | 46.8164 | 51467 |
| TP2/PP4 | 开启 | 46.4499 | 47.2129 | 52551 |

PyTorch峰值覆盖该进程整个运行，包含初始化及采样，不等于稳态常驻或整卡占用；整卡采样还包含库分配及n06 GPU0–3原有约3.4GB占用，15秒采样也可能漏过瞬时峰值。四组均无OOM。

### prof及实际overlap证据

每组4个gzip Chrome trace：`runs/qwen38_tp_overlap_20261117/<phase>/torch_profile/rank-{0,4,8,12}.json.gz`。16份全部完整解压/JSON解析成功，每份均有实际GPU kernel与`ProfilerStep#20`、`ProfilerStep#21`；开启组每份还出现真实userbuffers GPU kernel，关闭组没有这些kernel。TP4/PP2日志确认4个TP通信域，TP2/PP4确认8个；通信及模型配置中的开关均与实验一致。

各组rank0直链：[TP4关闭](../../../runs/qwen38_tp_overlap_20261117/tp4_pp2_overlap_off/torch_profile/rank-0.json.gz)、[TP4开启](../../../runs/qwen38_tp_overlap_20261117/tp4_pp2_overlap_on/torch_profile/rank-0.json.gz)、[TP2关闭](../../../runs/qwen38_tp_overlap_20261117/tp2_pp4_overlap_off/torch_profile/rank-0.json.gz)、[TP2开启](../../../runs/qwen38_tp_overlap_20261117/tp2_pp4_overlap_on/torch_profile/rank-0.json.gz)。TP4/PP2选中rank覆盖两个PP stage的两个DP副本；TP2/PP4覆盖四个PP stage的一个DP副本。record_shapes/with_stack均关闭。

机制线索：TP4/PP2的rank0在相同两步采样内，GPU kernel事件数从271950增至386638，开启组新增`kuserbuffers_pushrecv/pushsend`及`userbuffers_fp16_sum_inplace_gpu_*`，并切换了部分GEMM kernel路径。这支持额外通信/切分/launch开销抵消收益的解释，不能仅相加各stream的kernel时长作为步耗时或完整临界路径归因。本次未调TE默认num_sm/num_splits/aggregate。

### 正确性、退出及验证范围

四组均满足连续step1–50、消费3200样本、launcher exit0、每组16rank成功最终标记与HF加载标记、64层无冻结/视觉参数、无NaN/skip、实际native/full及正确拓扑/batch。每组16份transport日志均确认shca IB及`Using network NCCL RDMA Plugin v8`，没有切换Socket数据通信。`assessment.json`保存各组逐项核验；summary的`all_four_complete_and_verified=true`。

TP4/PP2开启/关闭的最大日志loss绝对差0.0133797、grad norm差8.469（第3步16.100对7.631）；TP2/PP4两组50步loss和grad norm在日志打印精度下逐步一致。上述不等于逐元素权重一致或收敛质量验证；本次是50步性能测试，不含验证集评估、4K保存/恢复或长测验收。

每组结束后按精确SAVE_DIR核对残留，四组均无需额外信号清理。结束后再次检查n05/n06均无本study的driver/agent/worker，两节点卡利用率均0%；n05每卡约2MiB，n06 GPU4–7约4MiB，GPU0–3原有3410–3474MiB保持未动。证据：[n05结束检查](../../../runs/qwen38_tp_overlap_20261117/post_study_m09r2n05.json)、[n06结束检查](../../../runs/qwen38_tp_overlap_20261117/post_study_m09r2n06.json)。这证明本次正常结束可释放进程；生产脚本Ctrl+C/异常退出竞态未在本轮修改。

验证已通过：入口/透传/拓扑等22项CPU回归；真实容器中的TP overlap false/true两项recipe测试；bash语法检查；源文件及训练/验证数据SHA256复核。证据：[unit_tests.log](../../../runs/qwen38_tp_overlap_20261117/unit_tests.log)、[recipe_tests.log](../../../runs/qwen38_tp_overlap_20261117/recipe_tests.log)、[source_and_test_audit.json](../../../runs/qwen38_tp_overlap_20261117/source_and_test_audit.json)。未提交Git。

### 修改位置与诊断限制

| 文件 | 本轮修改 |
| --- | --- |
| `examples/qwen/run_qwen38_text_sft.sh` | 新增默认false的TP_COMM_OVERLAP并透传MPI；未改用户的native、TP2/PP4、RECOMPUTE=none等默认值 |
| `hcu_megatron_bridge/src/hcu_megatron/bridge/recipes/qwen38/bw1000/qwen38_sft.py` | 从环境开关设置CommOverlapConfig.tp_comm_overlap，DP grad/param overlap仍关闭 |
| `tests/unit_tests/qwen38_text_sft/test_full_workflow.py` | 增加true/false的MPI透传回归 |
| `tests/unit_tests/qwen38_text_sft/test_recipe.py` | 清理fixture开关环境并增加两项recipe配置测试 |
| `runs/qwen38_tp_overlap_20261117/run_study.py` | 顺序调度4组、采设备数据、保存计划/日志/配置/完成状态，并按本组身份检查残留 |
| `runs/qwen38_tp_overlap_20261117/analyze_study.py` | 解析逐步指标、16份trace、GPU/UB/步骤及IB证据，核验配置/模型/退出/诊断时间并生成assessment/summary |
| `runs/qwen38_tp_overlap_20261117/capture_live_stacks.py` | 临时实验诊断：按精确SAVE_DIR及rank非阻塞读取Python栈 |
| 本handoff | 更新当前状态、结果、产物链接与验证限制 |

TP2/PP4关闭组首步631.2813秒包含诊断干扰，不能当作干净冷启动对照。一次GDB符号加载超时，分离后核对身份并SIGCONT恢复；另一次有界无符号采样约0.3秒完成且带恢复保护。之后使用仅安装于临时目录的py-spy 0.4.3非阻塞采样，未修改训练依赖。局部栈的P2P同步等待不足以判定通信死锁；全阶段采样亦显示Triton dispatch及前后向，任务随后正常推进，未修改P2P/IB/模型/数据/训练源码。最后一次栈文件比step26开始早约476秒，正式窗口独立于诊断。详见[诊断限制](../../../runs/qwen38_tp_overlap_20261117/tp2_pp4_overlap_off_diagnostic_notes.json)及该组diagnostics目录。其它历史孤儿进程清理记录保留在[prior_orphan_cleanup_n06.json](../../../runs/qwen38_tp_overlap_20261117/prior_orphan_cleanup_n06.json)。

## 2026-11-17：统一使用原生RMSNorm/FusedAdam

文本、VL和单节点入口统一沿框架原生算子路径执行，不再选择RMSNorm/Adam实现。删除两份替代实现及其专用测试；清理启动环境/MPI透传、worker加载和手工预检中的多余后端字段，模型审计统一记录`kernel_backend=native`。缓存隔离、GPU提前绑定、MTP审计、prof和退出状态记录保持原有行为。默认输出目录仍为`outputs/qwen38-text-native-seq<SEQ>/<时间戳>`。

本轮修改位置：

- `examples/qwen/run_qwen38_text_sft.sh`、`run_qwen38_vl_sft_full.sh`、`run_qwen38_text_sft_single_node.sh`：直接启动原生实现，清理多余环境设置及MPI透传。
- `examples/qwen/qwen38_text_sft_worker.py`、`qwen38_vl_sft_worker.py`：直接加载训练框架；文本模型元数据记录固定的原生实现标识。
- `tools/checkpoint/check_qwen38_text_sft.py`：手工检查仅记录当前原生实现，保留模型/数据/拓扑/MTP等恢复检查。
- `tests/unit_tests/qwen38_text_sft/test_full_workflow.py`、`test_tools.py`、`test_worker_exit.py`：同步单一执行路径和审计字段，保留有效回归。
- `runs/qwen38_run01_run02_tflops_20261117/test_seq2048_from1024.sh`、`runs/qwen38_tp_overlap_20261117/run_study.py`：调用当前统一入口，不设置实现选择变量。
- 本handoff及`qwen38-text-sft-full.md`、`qwen38-text-sft-ultrachat.md`、`qwen38-text-sft-steady-plan.md`、`qwen38-sft-long-sequence-and-vl.md`：清理旧方案与操作命令，保留当前启动说明及有效实验结论。

验证：40项CPU回归通过，修改的Python和Shell语法检查通过，当前启动代码/工具/测试/操作文档的失效引用扫描通过。本轮未启动GPU训练；原生执行证据沿用上一节四组各50步验收，VL未重新进行GPU验收。历史日志、配置、prof与JSON审计原样保留；其中源文件哈希及测试结果对应当时版本，不改写为清理后的版本。本轮未提交Git。

## 2026-11-17：四组prof瓶颈及旧16层VL对照

本轮只读取已有prof/配置/源码/权重，新增离线分析脚本；未改训练代码或参数、未启动GPU训练、未提交Git。输入为四组文本共16份trace，以及用户指定旧VL目录`/public/home/yugeng/work/checkpoints/mantis-birds-one-node-16layers-20261109_105654`的rank0/4两份trace。全部18份已解析；PP kernel通过GPU correlation→HIP launch→nccl:coalesced→紧邻c10d::send/recv归属，未把所有同名Generic NCCL kernel一概当PP。结果中未归属NCCL kernel数量为0。

证据：[逐trace统计](../../../runs/qwen38_prof_bottleneck_20261117/trace_metrics.json)、[瓶颈汇总](../../../runs/qwen38_prof_bottleneck_20261117/bottleneck_summary.json)、[跨rank等待核对](../../../runs/qwen38_prof_bottleneck_20261117/wait_cross_checks.json)、[权重及位置参数核对](../../../runs/qwen38_prof_bottleneck_20261117/conversion_checks.json)、[head/loss核对](../../../runs/qwen38_prof_bottleneck_20261117/head_loss_breakdown.json)、[等待对齐图](../../../runs/qwen38_prof_bottleneck_20261117/tp2_pp4_wait_alignment.png)。

### CheckpointFunction与Linear、权重转换

旧VL语言16层、TP2/PP4、DP1、不开重计算、数据worker2；当前文本64层、DP2、full/uniform/1、worker0。不能直接对比总步耗时。旧rank0还包含视觉塔，语言结构比较优先使用旧rank4的纯语言stage。旧rank4在1个采样step/64个microbatch内，_LayerNormLinear和_Linear各512次；当前TP2/PP4 rank0在2个step/每步32个microbatch内各2048次，均完整出现在CheckpointFunctionBackward内的autograd线程732332，而非主线程731115；CheckpointFunction本身1024次=16层×32microbatch×2step。按语言层数/microbatch归一化，二者每种TE函数均为2次/层/microbatch。

MCore `tensor_parallel/random.py`的CheckpointFunction.forward在torch.no_grad下调用层，backward在torch.enable_grad下重跑前向再反传。因此外层主要显示CheckpointFunction、Linear出现在backward重算线程是预期记录方式，不是没有Linear计算。参见[checkpoint实现](../../../3rdparty/Megatron-Bridge/3rdparty/Megatron-LM/megatron/core/tensor_parallel/random.py)。不要只为改变trace外观关闭full；历史4K无重计算已有OOM。

当前导出文件851个语言tensor的key/shape/dtype全部与预期及转换manifest一致；8个首/末层norm、GDN、attention、MLP tensor共545290240字节（约520MiB）逐字节SHA256与源VL及转换时记录一致；64层、hidden5120、head256、attention/GDN结构及rope_parameters保留。未重新哈希全部约54GB权重，未做HF/Megatron全模型输出对齐；现有现象没有指向转换错误。

### DataLoader 1.6秒的来源及优化

TP2/PP4关闭rank0两个step的DataLoader.__next__分别1.617008/1.431269秒；首个范围内已记录Torch子操作区间仅约13.047毫秒，约1603.961毫秒未被Torch子算子覆盖，pin_memory/_pin_memory调用约1毫秒量级，无巨大tril构造开销。CPU trace不足以把未记录部分完全归因于某一个Python函数；源码表明JSONL取样后仍进行HF chat template/tokenization、assistant mask、padding/collate，不是prepare时已缓存token IDs。get_attention_mask_from_fusion默认true，此处不是生成未使用4K×4K mask导致的1.6秒。

SFT batch sampler每次返回本DP副本32条样本；train_step先prepare_finetuning_batch取整批再分32个microbatch。四个PP stage都在step开头取数，捕获副本中各stage约1.5–1.7秒/step且大体同时发生，不能跨rank相加成全局损失。旧VL使用Energon/2worker/persistent_workers，rank0/4的64次主进程取数分别总计约19.6/11.0毫秒；数据/预处理不同，不能把差异全归给worker数量。

首个低成本候选：在相同4K/full/TP2PP4条件下仅将NUM_WORKERS从0改为2（再测4），保持data/seed/batch/overlap不变，比较step26–50及DataLoader/GPU空闲。现入口已支持该环境变量，本轮没有实际验证提速。长期方案是预先保存token IDs及assistant loss_mask，并校验thinking/template/EOS/labels完全等价；当前JSONL缓存仅为原始文本及行索引，不应误认为token cache。进一步减少各TP/PP rank重复CPU预处理需要保持数据对齐，不能直接删除中间stage取数。

### hipDeviceSynchronize、首次长NCCL与PP负载

TP2/PP4关闭rank0在两个采样step内有75次hipDeviceSynchronize，总19.488531秒；70次紧随PP coalesced launch（间隔<500微秒），合计19.488470秒。该CPU同步范围内同时覆盖约7.380秒GPU计算、10.967秒PP kernel。它是CPU等待当前设备已有任务/通信完成的时间，不是可以与GPU时间相加的独立计算开销。模型配置batch_p2p_comm=true、batch_p2p_sync=true；[MCore P2P实现](../../../3rdparty/Megatron-Bridge/3rdparty/Megatron-LM/megatron/core/pipeline_parallel/p2p_communication.py)在req.wait后显式torch.cuda.synchronize防race。不能据此直接删同步或认定纯通信慢。

Generic_4同名GPU kernel可执行TP、PP、DP等不同工作。按时间排序，rank0首个Generic NCCL是TP reduce-scatter，约0.295毫秒；rank4/8/12首个为PP接收，约172.538/215.784/338.745毫秒。rank0最长PP kernel约1.490310秒，关联到send/recv，起于采样step开头约2.180秒；同一窗口rank12仍执行约1.336514秒计算，rank8约0.674119秒、rank4约0.388919秒。各rank CPU step起点差小于0.5毫秒。长条主要反映流水线填充/等待反向依赖，不能当成单个张量在线路上传了1.49秒；算子聚合视图也不能当作单次事件。

以下为TP2/PP4关闭组每step GPU区间（两个采样step平均）；PP包含传输和等待，不是纯链路耗时。

| rank / PP stage | 语言层 | 计算kernel秒 | PP kernel秒 | CPU取数秒 |
| --- | --- | ---: | ---: | ---: |
| 0 / PP0 | 1–16 | 16.320 | 5.484 | 1.524 |
| 4 / PP1 | 17–32 | 16.233 | 5.026 | 1.507 |
| 8 / PP2 | 33–48 | 16.066 | 4.646 | 1.525 |
| 12 / PP3 | 49–64 | 19.896 | 0.372 | 1.581 |

末段head/loss已按CPU scope→HIP launch→GPU correlation单独核对：64次head前向及64次反向关联192个GPU kernel，两个step累计约6.016300秒（每step约3.008秒）；64次交叉熵前后向关联1472个GPU kernel，两个step约1.218806秒（每step约0.609秒）。合计每step约3.62秒，与末段相对其它stage多出的计算基本一致，是优先做负载均衡/优化head-loss的明确目标。当前cross_entropy_loss_fusion=false。

现有PP等待在前段约每step4.65–5.48秒、末段仅约0.37秒，且长等待期间其它stage正在计算，支持负载与依赖问题，不支持仅根据NCCL kernel长就断言IB链路带宽差。SP下每条TP lane的PP隐藏张量约为4096/TP×1×5120×2字节：TP2约20MiB、TP4约10MiB；kernel持续时间不是这笔数据的纯拷贝耗时。旧单节点VL的rank4同样有约27.05秒PP等待/35.70秒profile step，说明单节点也会因不均衡产生长NCCL等待；不能把旧VL直接作为当前64层性能基线。

优化顺序：先测NUM_WORKERS=2/4；再尝试末段少放语言层、前段分摊，或优化当前约3.62秒/step的head/loss（融合CE仅覆盖其中约0.61秒，不能期望消除整个head代价）。非均匀layout需验证全局混合层索引、数值、显存；目前脚本未直接暴露此布局。VPP/interleaved及PP异步P2P可作为后续方案，需确认GDN/HCU支持和激活显存，不能直接在当前非interleaved配置中删除同步。增加GBS只会降低理想填充气泡比例，同时改变优化batch与步工作量，不是无条件同等训练优化。

### 四种情况的GPU路径与mRoPE

下表GPU指标仅为捕获rank0的采样区间/step，类别有重叠，不能逐项相加。稳态步秒数来自原实验step26–50；实际profile步秒数在TP4关闭/开启约31.10/37.83秒、TP2关闭/开启约25.31/29.89秒，采样开销不同，不能用profile壁钟替代稳态测速。

| 配置 | 稳态秒/步 | GPU计算秒 | 普通TP NCCL秒 | TP Userbuffers秒 | PP kernel秒 |
| --- | ---: | ---: | ---: | ---: | ---: |
| tp2_pp4_overlap_off | 25.211 | 16.32 | 2.55 | 0.00 | 5.48 |
| tp2_pp4_overlap_on | 29.666 | 20.14 | 0.30 | 3.35 | 6.03 |
| tp4_pp2_overlap_off | 29.034 | 17.96 | 6.26 | 0.00 | 1.34 |
| tp4_pp2_overlap_on | 34.976 | 21.77 | 0.71 | 8.90 | 3.07 |

开启后确实进入Userbuffers路径，普通TP NCCL减少，但新增UB GPU占用及不同/更多GEMM kernel，计算区间也变长。默认UB配置没有净收益，保持TP_COMM_OVERLAP=false；低PP数的TP4/PP2虽减少部分PP等待，却有更高TP成本，不能因PP4等待长就盲目改PP2。

纯文本[Qwen35Bridge](../../../3rdparty/Megatron-Bridge/src/megatron/bridge/models/qwen/qwen35_bridge.py)明确设置position_embedding_type=rope；VL [Qwen35VLBridge](../../../3rdparty/Megatron-Bridge/src/megatron/bridge/models/qwen_vl/qwen35_vl_bridge.py)设置mrope。mRoPE作用于多模态语言序列的时间/高/宽坐标，并非视觉塔内部一个需要复制的线性权重。纯文本三轴位置相同，mRoPE的轴选择退化为普通RoPE；源/导出rope_parameters完全相同，theta=10000000、rotary_percent=0.25、head_dim=256即旋转64维。抽样位置0/1/17/4095的等轴角度构造差为0，不代表全模型数值对齐。当前apply_rope_fusion=false，RoPE可表现为aten点运算而无独立命名模块；旧rank0 gzip全文搜索mrope/rotary也无同名字符串，配置/代码与prof中模块名称应分开判断。

本轮新增离线脚本均在`runs/qwen38_prof_bottleneck_20261117/`：`inspect_trace.py`（事件字段）、`analyze_traces.py`（18份trace/CPU-GPU区间）、`summarize_bottlenecks.py`（PP关联及统计）、`focus_waits.py`（同步与跨rank对齐）、`head_loss_breakdown.py`（head/loss关联）、`check_conversion.py`（结构及权重抽检）、`plot_focus.py`（对齐图）。仅新增这些分析工具与产物、更新本handoff。

## 2026-11-17：单节点减层入口独立化（最终简化版本）

按用户最新要求，只完善启动脚本，不再启动GPU训练测试。单节点入口沿主入口已有组织方式独立设置模型、数据、训练、prof、环境/缓存和本机torchrun，不再调用`run_qwen38_text_sft.sh`，不读取hostfile或启动MPI。继续共用现有Python worker/recipe，不新增公共启动层。

- 单节点现有设置保留：16层HF目录、SEQ4096、8卡、TP2/PP4、GBS32/MBS1、RECOMPUTE=none、50步/warmup2、NUM_WORKERS=4、评估/保存/定期退出均关闭；PROFILE=true，窗口[5,6)，rank[0,2,4,6]，shapes/stack关闭。
- 所有默认目录保持原值：模型`/public/home/yugeng/work/models/Qwen3.8-27B-text-16layers`；数据仍由原SEQ/样本数规则选择；SAVE_DIR仍是`outputs/qwen38-text-native-seq<SEQ>/<时间戳>`，没有新增单节点默认目录；QWEN38_CACHE、TMPDIR及其它缓存路径均未改变。可继续显式设置SAVE_DIR。
- 主文本脚本仅补充prof窗口与输出路径注释；配置、默认参数、默认目录、双节点启动和日志逻辑均保持修改前原样。脚本之外的worker、预检和临时目录改动已撤回。
- 最终版本仅做Shell语法检查；对应测试断言同步到当前单节点参数，并检查单节点在主入口/hostfile不存在时仍可执行。按用户要求不再运行训练测试，不能把先前的试验结果当作最终版本GPU验收。

修改位置：`examples/qwen/run_qwen38_text_sft_single_node.sh`、`examples/qwen/run_qwen38_text_sft.sh`的prof注释、`tests/unit_tests/qwen38_text_sft/test_full_workflow.py`和本handoff。未提交Git。

```bash
cd /public/home/yugeng/work/Megatron-bridge-das-qwen38-sft-review
SAVE_DIR="$PWD/outputs/single-node-prof-run01" \
  bash examples/qwen/run_qwen38_text_sft_single_node.sh sft
```

训练日志在`SAVE_DIR/train_logs/node-0.log`；prof在`SAVE_DIR/torch_profile/rank-{0,2,4,6}.json.gz`。默认只采第5步，第6步包含导出开销；需要采两步时可显式设置`PROFILE_STEP_END=7`。默认目录没有改变。

## 2026-11-17：DataLoader说明、非均匀PP参数与RoPE fusion开关

本轮修改主文本入口`examples/qwen/run_qwen38_text_sft.sh`、文本recipe `hcu_megatron_bridge/src/hcu_megatron/bridge/recipes/qwen38/bw1000/qwen38_sft.py`、CPU启动测试`tests/unit_tests/qwen38_text_sft/test_full_workflow.py`及本handoff。未改VL/单节点Shell；共用文本recipe在新环境变量未设置时保持原配置。未提交Git，未启动训练。本轮容器直接导入TE时报告No CUDA GPUs are available，因此未进行GPU融合前反向数值或性能验收，不能声称RoPE fusion已在当前环境实测通过。

### 取数、worker与预处理缓存

旧VL并非没有DataLoader：旧rank0/4在捕获的一步内分别有64次主进程取数，总计约19.6/11.0毫秒；与当前文本一次取32条、约1.5秒的长块不同。旧VL使用Energon和2个worker，部分数据处理在后台完成且不会出现在主进程CPU trace中。当前四组文本prof的worker=0，取样、模板、tokenization、mask、collate同步占用主进程。每次取本DP副本的32条样本是batch sampler/GBS64/DP2的结果，并非把整份数据集一次读入，也不等于GPU microbatch变成32（MBS仍为1）。数据路径、处理逻辑及取数粒度均不同，不能只归因于“一次读多了”。

NUM_WORKERS指定每个训练rank的DataLoader子进程数，不是线程数、GPU数或batch size。大于0时子进程可在GPU训练当前batch期间预取并处理后续batch，主进程next主要等待/接收结果；不足时仍然会等待。16个rank各设2通常会创建32个数据worker，需考虑CPU、内存及共享存储压力，不保证越多越快。当前主入口和单节点入口默认已经是4（本轮保留）；旧四组实验配置0不代表当前脚本默认值。recipe仍persistent_workers=false，跨完整数据迭代器重建可能重建worker，不影响正常迭代中的预取。

当前prepare的--check-tokens只做长度/监督检查并输出对话JSONL；.idx.npy是行索引，训练仍重复tokenize。若要“处理一次、训练直接读取”，建议独立的未packing token缓存：离线复用当前GPTSFTChatDataset相同预处理，保存每样本input_ids、监督mask及collate所需contexts/answers/元信息或可等价重建的信息；保存tokenizer/chat_template/thinking/seq/EOS/源数据指纹；运行时使用与当前sampler兼容的缓存Dataset，保留原shift/padding/collate语义，绕过模板/tokenization。接入前比较同样本tokens、labels、loss_mask、position_ids（特别是label与mask的移位、padding为0、多轮assistant边界），再做同seed/batch短训和prof。当前入口没有这种token缓存Dataset，不能只将DATA_PATH指向随意写出的token JSONL/Parquet。此方案本轮只说明，未实现或生成缓存。

Bridge另有GPTSFTDatasetConfig.enable_offline_packing/offline_packing_specs及预处理packed Parquet路径，但packing会改变样本拼接、边界、位置及GDN状态处理；需单独验证当前HCU/GDN路径。它与保持一条会话一个样本的token缓存是不同改动，不作为本轮直接替换方案。

### 非均匀PP：末段少放语言层

主入口新增并经MPI转发：NUM_LAYERS_IN_FIRST_PIPELINE_STAGE、NUM_LAYERS_IN_LAST_PIPELINE_STAGE；默认均为空，recipe对应None，保持均匀划分。非空时转为整数设置MCore同名小写字段，其余stage均分剩余语言层。沿用框架原生配置校验，没有恢复启动前数据扫描。参数表示decoder层数，不包括embedding、LM head、loss或额外MTP模块；MTP若启用仍需计入末段实际负载。

64层/PP4的建议首轮候选仅设置末段13，即[17,17,17,13]；全局范围1–17、18–34、35–51、52–64，各段仍各4个full-attention层，前段各增加1个GDN、末段减少3个GDN。也可first=20,last=12得到[20,16,16,12]，但首段增加4层、5个full-attention层，可能增加首段显存/计算压力。只设last=12时剩余52无法被3整除，不是有效配置。参数并非层数越少越快；需比较相同数据/重计算/GBS下的稳态step、prof及峰值显存。

源码experimental_attention_variant_module_specs.py先按全局层号生成混合层spec，再根据get_transformer_layer_offset/get_num_layers_to_build切片；各stage无需单独满足4层周期，不应把局部第一层重新当全局第一层。CPU执行抽取的实际MCore两个函数，核对[17,17,17,13]、[20,16,16,12]、[18,18,18,10]的数量/offset及全局0–63无重复无遗漏通过；未进行非均匀PP的分布式HF加载/数值/GPU吞吐验收。已有model_rank_*.json的layer_numbers/attention_types可核对实际构建结果。首次对照应从同一个HF初始权重开始，不使用旧布局的RESUME_FROM作为性能对照。

```bash
cd /public/home/yugeng/work/Megatron-bridge-das-qwen38-sft-review
NUM_LAYERS_IN_FIRST_PIPELINE_STAGE= NUM_LAYERS_IN_LAST_PIPELINE_STAGE=13 \
TP=2 PP=4 SEQ_LENGTH=4096 GLOBAL_BATCH_SIZE=64 RECOMPUTE=full \
NUM_WORKERS=4 TP_COMM_OVERLAP=false APPLY_ROPE_FUSION=false \
TRAIN_ITERS=50 WARMUP_ITERS=2 EVAL_ITERS=0 SAVE_INTERVAL=0 \
PROFILE=true PROFILE_STEP_START=20 PROFILE_STEP_END=22 \
RESUME_FROM= SAVE_DIR="$PWD/outputs/text4096-pp-17-17-17-13" \
  bash examples/qwen/run_qwen38_text_sft.sh sft
```

均匀对照将两个NUM_LAYERS_IN_*变量都设为空，其余条件相同并另设SAVE_DIR。若与旧worker0实验比较，先固定NUM_WORKERS，不把worker和PP变化同时归为布局收益。上述指令本轮未执行。

### RoPE fusion及后续合并VL

主入口新增APPLY_ROPE_FUSION=false，支持true/false/1/0并MPI转发；recipe设置cfg.model.apply_rope_fusion。设APPLY_ROPE_FUSION=true即可选择MCore→TE fused_apply_rotary_pos_emb路径；仍是position_embedding_type=rope，没有更改theta、rotary_percent或频率定义。容器安装TE 2.13.0（HCU定制版），本地有FusedRoPE前后向实现；框架入口支持当前普通RoPE/partial rotary路线。它不同于fused_single_qkv_rope；后者当前gated attention不支持，本轮没有设置。建议单独A/B：保持PP均匀和其它条件不变，只切换此开关并换SAVE_DIR；先验证前反向、loss/grad及prof，再决定是否默认开启。此优化作用于full-attention层，48层GDN不因此融合；不能预先承诺明显整步收益。

RoPE/mRoPE及fusion开关本身不引入需合并的可学习投影权重。纯文本t=h=w时，源mRoPE选择的三轴频率一致，可退化为相同参数的普通RoPE；此前等轴角度抽检通过，不代表完整模型等价验收。fusion只是计算实现，允许浮点舍入差异，不要求训练后“转换RoPE权重”。Bridge的validate_rope_fusion_compatibility当前仅允许position_embedding_type=rope；VL的mrope不能据此直接套用文本融合开关，原VL配置仍保持关闭。

合并时应先将训练后的Megatron语言权重正确导出为HF（不能直接拼接分片），把语言层、embedding、norm、LM head按源VL键名放回，保留视觉塔/merger及原VL架构、text_config.rope_parameters（包括mrope_section=[11,11,10]、mrope_interleaved=true、theta=1e7、partial_rotary_factor=0.25）、processor及多模态位置ID生成逻辑。不要用纯文本config.json覆盖VL配置；后续VL训练按原VL provider走mrope，而非沿用文本provider的rope。文本SFT可能改变语言能力和视觉语言对齐，不能承诺合并后多模态效果完全不变；应分别做纯文本及图文样本的加载/前向和效果回归。此类效果变化与是否融合旋转运算需要区分。

验证：主入口bash -n、修改Python文件AST语法检查、22项CPU topology/launch测试通过；新增覆盖非均匀PP参数和fusion布尔值的MPI环境传播（含空值、1/0归一化）。GPU训练与融合kernel数值/性能验证均未执行，无新的OOM或提速结论。

## 2026-11-17：单节点实际prof核对DataLoader预取

用户提供`outputs/qwen38-text-native-seq4096/20261117-163907`，本轮读取其resolved_config、日志及4份prof，与`runs/qwen38_tp_overlap_20261117/tp2_pp4_overlap_off`配置和rank0 prof对照。未修改训练代码、未启动训练、未提交Git。

两者SEQ4096、相同UltraChat JSONL数据目录、batch loader、MBS1、assistant监督、固定长度padding；单节点16层/TP2PP4/DP1/GBS32/worker4/无重计算，旧双节点64层/TP2PP4/DP2/GBS64/worker0/full重计算。每次各DP副本取数都是GBS/DP=32条，不能用“单节点读的数据少”解释差异。数据源相同不代表不同DP/采样step取得的具体32条会话完全相同。

单节点ProfilerStep#5约7.904秒，其中rank0/2/4/6的enumerate(DataLoader)#_MultiProcessingDataLoaderIter.__next__分别0.124153、0.126692、0.128982、0.145253毫秒；旧双节点rank0在step20/21的_SingleProcessDataLoaderIter.__next__分别1617.008317、1431.268940毫秒。多进程事件名与配置worker4相符，支持单节点后台预处理/预取已让主进程采样时基本无需等数据；这里测到的是主进程取出准备好batch的等待/交接时间，不是所有CPU分词工作总和，也不能据此宣称分词本身加速上万倍。主进程trace未包含所有worker的CPU工作。

同SEQ只规定训练张量长度，不决定CPU处理放在哪个进程或能否被GPU计算覆盖。单节点此步仍有约7.9秒GPU训练时间可供worker准备后续批次；观察到4个rank取数均短，说明这一采样步供数充足。不能仅靠单节点一个采样步保证双节点也能达到相同等待：双节点rank总数/CPU压力不同，尚需在相同全参配置下只改NUM_WORKERS=4进行稳态和prof验证。当前主启动脚本默认已是4，历史四组worker0配置不会随脚本更新而变化。本轮只补充本handoff，不需要为此更改数据或训练代码。

## runs清理

按用户要求删除38项历史checkpoint目录及参数比较副本，约8.34TiB；runs从9,172,657,116,672字节降至约4,087,723,008字节（3.81GiB）。保留约3.5GiB编译缓存及日志/曲线/JSON/CSV/脚本，原始模型与数据均未动。清单见[cleanup_manifest.json](../../../runs/maintenance_20261116/cleanup_manifest.json)，被清理实验根目录有`CHECKPOINTS_REMOVED.json`。历史result.json成功标记是当时测试结论，不表示checkpoint仍存在；后续训练请使用新的outputs目录。


## 2026-11-17：当前实现提交快照与跨服务器迁移

按用户要求整理待提交文件，不增加环境变量开关，不修改训练默认参数和目录，不执行 git add/commit/push。完整说明见 [qwen38-sft-commit-20261117.md](qwen38-sft-commit-20261117.md)，逐文件清单见 `tools/release/qwen38_sft_files.txt`。

- 新增 `tools/release/commit_qwen38_sft.sh`：默认 `--preview` 只读预览；用户审阅后手动运行 `--commit` 才暂存清单并提交，不包含未列出的本机文件。
- 官方 Bridge 是子模块，当前只有 recipes/__init__.py 的 HCU 导出和 training/gpt_step.py 的 MTP loss_mask 传递两处差异；嵌套 Megatron-LM 无修改。新增 `patches/qwen38-sft/*.patch` 保存这两处差异，保留官方 gitlink，不创建未推送的子模块 commit。
- 新增 `tools/release/apply_qwen38_vendor_patches.sh`：接收服务器初始化递归子模块后手动 `--apply`，检查固定基线及重复应用/冲突。recipes 文件原本也由 env/env.sh 复制生成；MTP 补丁不能只靠该复制恢复。
- 新增 `examples/qwen/hostfile.example`，不提交本机 hostfile。当前本机 hostfile 为 n05/n07，迁移或双节点运行前必须按实际节点核对，未修改它。
- 不提交模型、数据、缓存、outputs/runs、prof 或容器；env/env.sh 的两行注释变化也排除。默认绝对路径保留，新服务器需准备相同路径或使用已有覆盖参数。
- 本次实际仅新增提交材料、追加文档，以及清理两个文本 shell 的 PROFILE 行尾空格。完成 shell 静态检查和只读预览；在临时目录验证两个补丁从官方 HEAD 能精确复现现有文件。未启动 GPU 训练，不代表新服务器验收，当前单节点失败另需关注空闲显存和 hipErrorNoBinaryForGpu。
