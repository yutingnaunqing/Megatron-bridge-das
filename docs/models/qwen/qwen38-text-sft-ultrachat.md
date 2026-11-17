# Qwen3.8 纯文本 SFT：官方 UltraChat Parquet 使用流程

## 数据及官方核对

源目录：`/public/home/yugeng/work/datasets/HuggingfaceH4_ultrachat_200k`。

官方仓库：[HuggingFaceH4/ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k)，本次校验固定commit为 `8049631c405ae6576f93f445c6b8166f76f5505a`。

已逐文件核对全部8个Parquet的文件名、大小、完整SHA256，均与官方LFS哈希一致；README的Git blob SHA1也一致。本地.gitattributes与该commit有元数据差异，但Parquet数据本体一致。没有改写源数据文件。

PyArrow读取全部分片footer，确认以下样本数及原生嵌套schema：

| 划分 | 样本数 | 分片数 | 用途 |
| --- | ---: | ---: | --- |
| train_sft | 207,865 | 3 | SFT训练 |
| test_sft | 23,110 | 1 | 验证来源 |
| train_gen | 256,032 | 3 | 本轮不使用 |
| test_gen | 28,304 | 1 | 本轮不使用 |

字段为 `prompt:string`、`prompt_id:string`、`messages:list<struct<content:string,role:string>>`。这份英文多轮合成对话数据经过过滤，已有assistant目标回答，适合普通SFT。当前源README的中文副本为同目录 `README.zh-CN.md`。

校验证据位于 `runs/qwen38_ultrachat_parquet_20261113/{official_repo.json,official_verification.json,parquet_metadata.json}`。复核命令（只下载元数据，不下载分片）：

```bash
python3 tools/verify_ultrachat_parquet.py \
  --source /public/home/yugeng/work/datasets/HuggingfaceH4_ultrachat_200k \
  --metadata runs/qwen38_ultrachat_parquet_20261113/official_repo.json \
  --output /tmp/ultrachat-official-verification.json
```

## 预处理规则

`data_preparation/prepare_ultrachat_sft.py`只接收完整的Parquet分片集合，直接由Arrow读取messages列表，不额外反序列化字符串或改写正文。

保留官方train_sft/test_sft边界，按首个user文本去重并阻止集合重叠；经Qwen实际chat template和collator检查，写出 `training.jsonl`、`validation.jsonl`、`data_manifest.json` 和前三条 `token_audit.json`。

初期筛选SEQ窗口内的完整对话，过滤超长或有效监督token为0的记录，保留全部assistant轮次监督，关闭packing。短窗口输出偏向短对话；正式训练需按目标长度和显存重新准备。使用test_sft调参后，应另外保留最终评测集。

## 默认入口

`examples/qwen/run_qwen38_text_sft.sh`支持 `prepare | preflight | dryrun | sft`。

| 变量 | 本页历史16层实验配置（需显式设置） |
| --- | --- |
| RAW_DATA_PATH | 当前官方Parquet目录 |
| LOCAL_HF_PATH | `/public/home/yugeng/work/models/Qwen3.8-27B-text-16layers` |
| DATA_PATH | `/public/home/yugeng/work/datasets/HuggingfaceH4_ultrachat_200k-qwen38-sft-seq1024-disabled` |
| SEQ_LENGTH / THINKING_MODE | 1024 / disabled |
| TRAIN_SAMPLES / VALIDATION_SAMPLES | 1024 / 64；只影响prepare，0表示不限制 |
| GPUS / TP / PP / CP / MBS | 8 / 2 / 4 / 1 / 1 |

训练调用 `qwen38_text_27b_sft_bw1000_bf16_config`，`--mode sft --step_func gpt_step`，从官方语言权重初始化，全语言参数更新，无视觉/MTP。

## 容器内启动

在n06宿主进入已配置好的容器：

```bash
docker exec -it yg_qwen38 bash
cd /public/home/yugeng/work/Megatron-bridge-das-qwen38-sft-review
```

统一入口已取代旧16层单节点包装。复现本页历史减层实验时，先显式设置以下参数（统一入口默认64层/native/GBS64）：

```bash
export NNODES=1 EXPECTED_NUM_LAYERS=16
export LOCAL_HF_PATH=/public/home/yugeng/work/models/Qwen3.8-27B-text-16layers
export SEQ_LENGTH=1024 TP=2 PP=4 GLOBAL_BATCH_SIZE=8 RECOMPUTE=none
export TRAIN_SAMPLES=1024 VALIDATION_SAMPLES=64 WARMUP_ITERS=1 EVAL_ITERS=0
export DATA_PATH=/public/home/yugeng/work/datasets/HuggingfaceH4_ultrachat_200k-qwen38-sft-seq1024-disabled
```



准备新数据目录；已准备成功时跳过此步。工具拒绝覆盖已有目录。

```bash
bash examples/qwen/run_qwen38_text_sft.sh prepare
```

默认8卡配置的短测流程（8张GPU均可使用时）：

```bash
export SAVE_DIR="$PWD/runs/ultrachat-parquet-smoke-$(date +%Y%m%d_%H%M%S)"
bash examples/qwen/run_qwen38_text_sft.sh preflight
TRAIN_ITERS=5 SAVE_INTERVAL=5 \
  bash examples/qwen/run_qwen38_text_sft.sh dryrun
TRAIN_ITERS=5 SAVE_INTERVAL=5 \
  bash examples/qwen/run_qwen38_text_sft.sh sft
```

本轮实测使用物理GPU4–7，4卡TP2/PP2，避开已有进程使用的0–3号卡。复现该配置时，在上面preflight/dryrun/sft命令之前设置：

```bash
export HIP_VISIBLE_DEVICES=4,5,6,7
export GPUS=4 TP=2 PP=2
```

其余参数为16层、SEQ1024、GBS8/MBS1、BF16、5步、第5步保存。物理卡根据实际资源情况选择；4卡和8卡使用独立SAVE_DIR，不互相resume。

日志为 `$SAVE_DIR/train_logs/node-0.log`，配置为 `resolved_config.yaml`，产物为 `checkpoints/iter_0000005/`，包含optimizer/RNG。此前16层全状态checkpoint约113GiB，需预留相应存储。

## 全量SFT或更长窗口

```bash
export SEQ_LENGTH=2048 THINKING_MODE=disabled
export DATA_PATH=/public/home/yugeng/work/datasets/HuggingfaceH4_ultrachat_200k-qwen38-sft-seq2048-full
TRAIN_SAMPLES=0 VALIDATION_SAMPLES=0 \
  bash examples/qwen/run_qwen38_text_sft.sh prepare

export SAVE_DIR="$PWD/runs/ultrachat-parquet-seq2048-$(date +%Y%m%d_%H%M%S)"
TRAIN_ITERS=50 SAVE_INTERVAL=10 \
  bash examples/qwen/run_qwen38_text_sft.sh dryrun
TRAIN_ITERS=50 SAVE_INTERVAL=10 \
  bash examples/qwen/run_qwen38_text_sft.sh sft
```

输出数量由去重和窗口过滤决定。全量prepare仍需时间；sft已取消自动逐样本检查。TRAIN_ITERS控制训练步数，不自动等于一个epoch；按最终样本数、GBS和目标epoch另行设置。可用 `EVAL_ITERS=2 EVAL_INTERVAL=5`开启验证，dryrun和sft保持相同设置。

本轮已在上述4卡配置完成5步SFT和完整checkpoint保存，4rank正常退出，无跳步或NaN。默认1024/64条子集已准备完成，可以从preflight开始；重新prepare需使用新的DATA_PATH。该结果验证加载/训练/保存链路，未测试全量epoch或生成效果。详细证据见handoff的“官方UltraChat Parquet数据接入”部分。
