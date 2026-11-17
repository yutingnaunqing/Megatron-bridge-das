# Qwen3.8 长序列文本 SFT 与完整 VL 全参 SFT 启动说明

日期：2026-11-16（北京时间）。既有500步验收覆盖纯文本SEQ1024；4096的run02已完成22步短测（首步49.082秒，step2–22平均15.214秒），尚未验收保存恢复。8192及完整VL仍未完成GPU训练验收。最新MPI启动结构参照旧pretrain入口，主节点统一保存日志；文本训练已取消自动preflight。文本统一入口为`run_qwen38_text_sft.sh`，默认native；旧full/long/native/train包装及共用MPI/runtime脚本已删除，single_node现已恢复为单节点配置包装。

## 吞吐开关与性能判断

文本recipe现默认设置 `cfg.logger.log_throughput=True`，启动时可用 `LOG_THROUGHPUT=true` 控制；关闭使用 `false`。已有通用runner可直接传 `logger.log_throughput=true`，但原文本shell入口不接受任意尾随KEY=VALUE，因此新增环境变量是该入口的可用方式。MPI显式转发此变量。此开关影响主iteration行；rank0原有独立MODEL_TFLOP/s/GPU输出仍保留。TensorBoard吞吐为另一开关，本次不默认启用。

SEQ1024、MBS1、TP4意味着一些矩阵较小，长序列可能改善计算与通信摊销，但现有证据不能证明短SEQ是主要瓶颈。full重计算、GDN/attention混合结构、PP气泡、关闭通信重叠都影响结果，具体占比需profiler验证。长SEQ增加激活和计算；不要根据SEQ等比例外推性能或显存。

本次固定长度padding，不packing。对同一批短对话，只把SEQ从1024改成8192会增加padding，不能据此认定有效吞吐提高。应同时报告实际输入token/s、监督token/s、padding token/s，并在相同数据/训练目标下比较。SEQ1024时MTP1采样峰值已约57,931MiB/卡（设备65,520MiB），长SEQ先关闭MTP，MBS1/full重计算，先测20～30步再决定长跑。新VL路径增加视觉塔及激活，同样不能保证16卡能容纳。

## 现有文本数据与长度变化

500步验收数据：`/public/home/yugeng/work/datasets/ultrachat-qwen38-steady-train16384-val256-seq1024-disabled`。官方UltraChat train_sft/test_sft，16,384训练样本、256验证样本，thinking disabled，assistant-only，保存原始messages JSONL。它不是每条都有1024个真实token，而是按1024训练输入上限筛选完整对话，collator补齐到1024。自回归shift前允许最多1025个token，shift后的训练输入最多1024个。超过上限的完整对话被过滤，不截断后冒充完整样本。

本轮用训练相同的真实tokenizer、thinking disabled模板及`GPTSFTChatDataset._process_example`在CPU全量重算，长度为自回归shift后、padding前的输入位置数：

| 划分 | 样本数 | 最短/最长 | 平均 | P50 | P95 | 平均监督token |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| train | 16384 | 86 / 1024 | 714.46 | 741 | 996 | 503.12 |
| validation | 256 | 188 / 1024 | 725.58 | 754.5 | 992 | 507.96 |

当前train样本的平均输入占SEQ1024的69.77%；若完全沿用这些短样本，补到4096/8192时分别只占17.44%/8.72%。这是完整数据集的静态分布，不冒充实际训练消费顺序的有效token吞吐。证据：[长度统计](../../../runs/qwen38_long_vl_launch_review_cpu_20261116/current_dataset_lengths.json)、[统计脚本](../../../runs/qwen38_long_vl_launch_review_cpu_20261116/inspect_lengths.py)。当前HF文本配置`max_position_embeddings=262144`，4096/8192无需改RoPE或重新转换HF权重。

已经筛好的短JSONL可以用于4096/8192配置兼容性检查，无需改JSONL格式；但长对话已在此前过滤掉，调大SEQ不能恢复它们。要训练更长对话，从原始Parquet分别重新prepare到独立目录。只放宽最大长度会纳入更多长对话，不保证所有样本都接近4096/8192，也不保证两种长度选出的样本完全相同。旧`.idx`只属于相应文件，不手工拷到新目录。

需要同步变化：模型/数据SEQ（recipe统一设置）、数据准备长度、数据目录、独立SAVE_DIR；MBS/TP/PP/重计算及GBS需考虑显存和训练预算。若GBS32固定，4096/8192的每步padding token预算分别为131,072/262,144，是1024时4/8倍。LR无需按SEQ机械缩放，样本和监督token量改变后需重新评估。原500步checkpoint的scheduler已衰减到0；以下均从原始HF独立初始化，不直接拿旧checkpoint跨数据/SEQ续跑。最新文本launcher不再自动检查恢复manifest。

## 文本准备与启动

以下命令在既有`yg_qwen38`容器内执行（节点宿主机先 `docker exec -it yg_qwen38 bash`），项目共享挂载。数据只在n05准备一次。

```bash
cd /public/home/yugeng/work/Megatron-bridge-das-qwen38-sft-review
SEQ_LENGTH=4096 bash examples/qwen/run_qwen38_text_sft.sh prepare
SEQ_LENGTH=8192 bash examples/qwen/run_qwen38_text_sft.sh prepare
```

默认分别输出 `/public/home/yugeng/work/datasets/ultrachat-qwen38-train16384-val256-seq4096-disabled` 与 `...-seq8192-disabled`。默认16384/256；设置 `TRAIN_SAMPLES=0 VALIDATION_SAMPLES=0` 可准备全量，但需使用另一新DATA_PATH。输出目录已存在时不覆盖，后续直接复用。

可选dryrun只解析配置（n05即可）；正常sft不再执行数据扫描/分词等自动preflight：

```bash
SEQ_LENGTH=4096 bash examples/qwen/run_qwen38_text_sft.sh dryrun
SEQ_LENGTH=8192 bash examples/qwen/run_qwen38_text_sft.sh dryrun
```

双节点训练在n05的现有容器中执行一次即可。默认读取`examples/qwen/hostfile`（当前n05、n06各一行，以实际文件为准），自动设置NNODES和MASTER_ADDR，MPI通过容器SSH端口11463拉起远端，各节点通过exec直接成为torchrun agent，再启动8GPU worker。无需在n06再执行命令。

```bash
cd /public/home/yugeng/work/Megatron-bridge-das-qwen38-sft-review
unset NODE_RANK MASTER_ADDR RESUME_FROM
SEQ_LENGTH=4096 SAVE_DIR="$PWD/outputs/qwen38-text-sft-4096-run01" \
  GLOBAL_BATCH_SIZE=32 RECOMPUTE=full LOG_THROUGHPUT=true \
  bash examples/qwen/run_qwen38_text_sft.sh sft

# 另一独立实验，不与上面同时运行。
SEQ_LENGTH=8192 MASTER_PORT=30122 SAVE_DIR="$PWD/outputs/qwen38-text-sft-8192-run01" \
  GLOBAL_BATCH_SIZE=32 RECOMPUTE=full LOG_THROUGHPUT=true \
  bash examples/qwen/run_qwen38_text_sft.sh sft
```

可用`HOSTFILE=/绝对路径/hosts`、`MPI_SSH_PORT=11463`覆盖；hostfile每行一个节点，首行为主节点，slots会统一转为每节点一个父进程。默认500步、warmup20、LR1e-5。显式NODE_RANK保留手工逐节点模式，日常hostfile启动请unset；NNODES=1保留单节点。

首次显存/功能试跑时，可设置 `TRAIN_ITERS=30 WARMUP_ITERS=2 SAVE_INTERVAL=30 EVAL_ITERS=0`验证末步保存，并使用另一SAVE_DIR。只测性能则用下面的50步配置关闭评估和保存。短测通过后500步另起新目录。8192与4096不要同时使用相同节点/端口。MPI日志现位于主节点 `SAVE_DIR/train_logs/mpi_YYYY-MM-DD-HHMM.log`，包含两节点输出及launcher错误；MPI模式不再生成node-0/1.log。显式NODE_RANK或单节点仍保留node-N.log。`SAVE_DIR/resolved_config.yaml`应有 `log_throughput: true`。

## SEQ8192的50步性能测试（最新操作）

当前统一文本入口直接使用原生算子，默认GBS64、RECOMPUTE=none；训练参数可通过环境变量设置。下列命令显式设置GBS32/full。历史run的完整算子配置以当时记录为准。数据准备按8192过滤完整对话，仅prepare时执行检查，训练启动不重复检查。

```bash
cd /public/home/yugeng/work/Megatron-bridge-das-qwen38-sft-review
unset NODE_RANK MASTER_ADDR RESUME_FROM
DATA8=/public/home/yugeng/work/datasets/ultrachat-qwen38-train16384-val256-seq8192-disabled

# 数据目录不存在时只执行一次；已有同配置数据则直接复用。
SEQ_LENGTH=8192 THINKING_MODE=disabled DATA_PATH="$DATA8" \
  LOCAL_HF_PATH=/public/home/yugeng/work/models/Qwen3.8-27B-text-64layers \
  TRAIN_SAMPLES=16384 VALIDATION_SAMPLES=256 MTP_NUM_LAYERS=0 \
  bash examples/qwen/run_qwen38_text_sft.sh prepare

# 独立性能短测，50步自动结束，不执行评估/常规或最终checkpoint保存。
RUN8="$PWD/outputs/text8192-perf-$(date +%Y%m%d-%H%M%S)"
SEQ_LENGTH=8192 THINKING_MODE=disabled DATA_PATH="$DATA8" SAVE_DIR="$RUN8" \
  LOCAL_HF_PATH=/public/home/yugeng/work/models/Qwen3.8-27B-text-64layers \
  TP=4 PP=2 MTP_NUM_LAYERS=0 GLOBAL_BATCH_SIZE=32 RECOMPUTE=full \
  TRAIN_ITERS=50 WARMUP_ITERS=2 EXIT_INTERVAL=0 EVAL_ITERS=0 SAVE_INTERVAL=0 \
  LOG_THROUGHPUT=true MASTER_PORT=30122 \
  bash examples/qwen/run_qwen38_text_sft.sh sft
```

`SAVE_INTERVAL=0`在当前Bridge实现中关闭常规和正常结束时的最终保存；无需依靠Ctrl+C避开末步保存，也不能用该性能run验收checkpoint恢复。没有改变全参训练、梯度或优化器更新。不要用旧手工preflight检查这条关闭保存的性能命令：旧checker仍要求正SAVE_INTERVAL，而自动预检已取消。

首次8192可能产生新的编译/调优；保留QWEN38_CACHE，重复短测时仅换RUN8。统计step11–50的mean/P50/P95；如果末段仍在明显变化，延长独立测试后再判定稳态。采集两节点GPU显存/利用率，确认无NaN/跳步；8192可否容纳仍需真实实测。GBS32、16GPU时，padding token/s/GPU=`16384 / 平均步秒数`，samples/s=`32 / 平均步秒数`。同时报告实际非padding/监督token吞吐，不能把padding吞吐当有效文本吞吐。

若只想与run02严格比较同一批样本的4K/8K固定形状性能，可以不prepare新数据，令8192训练的DATA_PATH指向已准备的seq4096目录（完整对话均在4K以内，补到8K）。这是相同样本的padding性能对照；新prepare的8K目录会纳入更多此前过滤的长对话，样本集合可能不同。两种测试应分别标注。

## 完整视觉语言全参 SFT

三个入口（统一文本、VL、single_node）均支持相同profiler环境变量。默认PROFILE=false，不抓取trace；开启示例：

```bash
PROFILE=true PROFILE_STEP_START=10 PROFILE_STEP_END=12 PROFILE_RANKS='[0,8]' \
  SEQ_LENGTH=4096 GLOBAL_BATCH_SIZE=32 RECOMPUTE=full \
  TRAIN_ITERS=20 WARMUP_ITERS=2 EVAL_ITERS=0 SAVE_INTERVAL=0 \
  SAVE_DIR="$PWD/outputs/text4096-prof-$(date +%Y%m%d-%H%M%S)" \
  bash examples/qwen/run_qwen38_text_sft.sh sft
```

输出`SAVE_DIR/torch_profile/rank-N.json.gz`；END为排除边界，以上抓2个active step。PROFILE_RECORD_SHAPES/PROFILE_WITH_STACK默认false，按需设true。单节点包装`run_qwen38_text_sft_single_node.sh`默认16层/SEQ1024/TP4/PP2/GBS8/full/native，使用同样开关但rank列表应在单节点范围内，如`[0,4]`；它不是完整27B性能验收。VL把命令末尾换成VL入口并使用其图文数据/模型，避免继承纯文本DATA_PATH/LOCAL_HF_PATH。profile会增加时间和内存，不拿采样窗口做普通吞吐基线。

使用独立入口 `examples/qwen/run_qwen38_vl_sft_full.sh` 和recipe `qwen38_vl_27b_sft_bw1000_bf16_config`：加载原始 `/public/home/yugeng/work/models/Qwen3.8-27B` 的完整语言/视觉/projection预训练权重；64语言层，三个freeze开关均false，PEFT无，MTP关闭，BF16/FP32 Adam与distributed optimizer，full/uniform/1重计算。与旧VL性能入口的随机初始化不同。

纯文本可以用于多模态模型的语言能力训练，但没有图像时视觉塔和projection不会经历真实视觉前向/反向；这不能验证视觉全参SFT。当前文本messages JSONL也不是本入口Energon读取格式。要验证三个部分一起训练，使用图像和对话均存在的 `/public/home/yugeng/work/datasets/mantis-birds-energon-v1`。assistant回复loss通过图像特征回传至视觉模块，不需要对image token本身计算语言loss。

Birds现成Energon无需因SEQ改为4096/8192重新打包；SEQ上限在在线processor/collator生效，分辨率由 `MIN_PIXELS`/`MAX_PIXELS`控制（默认200704/1003520），图像视觉token也占语言序列。已有历史核验最长有效输入702，这批数据本身较短；本轮未重新全量核验图像数据，不应把SEQ8192当成真实8K图文长上下文测试。修改图像分辨率需重新核验有效长度。

该目录`.nv-meta/split.yaml`只有train，没有独立val；新VL入口关闭validation，不伪造held-out指标。长跑要判断泛化，应先按样本ID准备独立train/val划分再启用验证。默认GBS32×500=16000次样本消费，会多次遍历2649条Birds样本，需注意数据重复和过拟合。

先在n05容器检查：

```bash
cd /public/home/yugeng/work/Megatron-bridge-das-qwen38-sft-review
SEQ_LENGTH=4096 bash examples/qwen/run_qwen38_vl_sft_full.sh dryrun
```

在n05容器中执行一次：

```bash
cd /public/home/yugeng/work/Megatron-bridge-das-qwen38-sft-review
unset NODE_RANK MASTER_ADDR
SAVE_DIR="$PWD/outputs/qwen38-vl-full-sft-4096-run01" \
  LOG_THROUGHPUT=true bash examples/qwen/run_qwen38_vl_sft_full.sh sft
```

同样默认使用hostfile中的n05+n07，远端自动获得模型、数据、SEQ、batch、LR、分辨率等环境参数，MPI rank自动映射为torchrun node_rank。默认SEQ4096、500步、warmup20、LR1e-5、TP4/PP2/DP2、GBS32/MBS1、save250。

首次也建议30步、warmup2、save30、独立目录。完整VL比纯文本显存更高；若OOM，先降低SEQ或视觉分辨率，必要时评估optimizer offload，不自动改变配置。检查视觉/projection/语言梯度及权重更新后，才能声明全参训练实测通过；本次仅完成配置解析。

两个新入口强制数据面 `NCCL_NET="NCCL RDMA Plugin v8"`、`NCCL_IB_DISABLE=0`，IB失败应停止并换节点，禁止Socket完成。bootstrap日志出现Socket不等于数据面回退，应核对`Using network`行。没有自动选节点/自动重启。缓存、日志和临时目录位于`/public/home/yugeng/`。

## 本轮验证与修改位置

[验证结果](../../../runs/qwen38_long_vl_launch_review_cpu_20261116/validation.json)：4096/8192对现有全部短样本的token检查通过；最终两个文本入口用真实4/2样本子集dryrun，完整VL4096入口dryrun，均正常结束并核对配置。新长样本数据尚未prepare，未启动GPU训练。checker退出时有既有共享文件系统临时目录清理告警，未影响退出码及配置解析；原始日志与首轮shell验证中断记录均保留。

当前保留统一文本入口`examples/qwen/run_qwen38_text_sft.sh`、VL入口`run_qwen38_vl_sft_full.sh`和16层单节点配置入口`run_qwen38_text_sft_single_node.sh`；两者各自包含环境、缓存和MPI逻辑，无共用MPI/runtime脚本。VL worker在`qwen38_vl_sft_worker.py`。新增VL recipe在`hcu_megatron_bridge/src/hcu_megatron/bridge/recipes/qwen38_vl/bw1000/qwen38_vl_sft.py`及两层`__init__.py`导出。文本吞吐开关在`qwen38/bw1000/qwen38_sft.py`的logger配置段，环境变量/MPI转发在统一文本入口中。未改MCore、未提交Git。

2026-11-16启动与存储更新：文本和VL均通过真实n05+n07 SSH/MPI CPU启动探针；探针不加载模型、不执行训练。[handoff](qwen38-text-sft-handoff.md)已精简。旧runs内checkpoint和参数对比副本已清理，历史结论及日志保留，不能再从旧runs恢复。
