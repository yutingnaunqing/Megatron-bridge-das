<!-- Copyright (c) 2026 Hygon Information Technology Co., Ltd. -->

# 基于Megatron-Bridge-das启动Qwen3.8-27B VL pretrain示例

## 1、环境准备

### 1.1 [光源社区](https://developer.sourcefind.cn/servicelist)拉取HCU基础docker镜像

### 1.2 创建容器

参考命令如下（请按需修改，正确挂载工作目录，修改正确镜像名称）：

```shell
docker run -it \
       --name megatron-bridge-das_test \
       --shm-size=64G \
       --device=/dev/kfd \
       --device=/dev/mkfd \
       --device=/dev/dri \
       --cap-add=SYS_PTRACE \
       --security-opt seccomp=unconfined \
       --ulimit memlock=-1:-1 \
       --ipc=host \
       --network=host \
       --workdir=/workspace \
       --privileged \
       -v /opt/hyhal:/opt/hyhal:ro \
       -v /public/home/yugeng/work:/workspace \
       REPOSITORY:TAG \
       /bin/bash
```

多节点时，各节点容器需要通过相同绝对路径访问代码、模型、数据和输出目录，并配置容器间免密SSH。

### 1.3 拉取代码仓

```shell
cd /workspace

git clone --recurse-submodules https://github.com/HYGON-AI/Megatron-bridge-das.git
```

如代码仓无法正常clone，可离线下载对应分支的`Megatron-bridge-das`、`Megatron-Bridge`及其子模块，按仓库目录结构放置。已有代码仓时直接进入对应目录。

### 1.4 安装依赖

```shell
cd /workspace/Megatron-bridge-das
pip install -r requirements.txt

source /opt/dtk/env.sh
source env/env.sh
```

## 2、数据处理

### 2.1 Mock图文数据

`mock`和`8gpu_smoke` recipe使用合成图像和文本对话，无需下载或处理训练数据。默认每条样本包含一张原始尺寸256×256的RGB图像，实际视觉token数由模型processor决定。

### 2.2 Energon图文数据

Energon recipe需要设置`DATA_PATH`，指向已转换的WebDataset tar和`.nv-meta`元数据目录。原始图片、JSON或Parquet目录不能直接作为该参数。

可以先使用仓库教学脚本生成测试数据：

```shell
cd /workspace/Megatron-bridge-das

python 3rdparty/Megatron-Bridge/tutorials/data/energon/prepare_example_data.py \
    --output-dir /workspace/datasets/qwen38-vl-energon-example \
    --num-workers 2
```

该脚本生成4条训练样本和2条验证样本，内容为纯色图像及颜色问答，仅用于验证数据加载和训练链路。

### 2.3 真实数据集 Mantis-Instruct

[TIGER-Lab/Mantis-Instruct](https://huggingface.co/datasets/TIGER-Lab/Mantis-Instruct) 提供图像和多轮对话。仓库已有匹配转换脚本，不需要从头编写 WebDataset writer。可以先下载一个子集，避免获取整个数据仓库。

例如下载 [chartqa 子集](https://huggingface.co/datasets/TIGER-Lab/Mantis-Instruct/tree/main/chartqa) 的训练标注和图片包（也可以从魔搭设区下载）：

```bash
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 \
    hf download TIGER-Lab/Mantis-Instruct --repo-type dataset \
    --include 'chartqa/train-*.parquet' 'chartqa/train_images.zip' \
    --local-dir /public/home/yugeng/work/datasets/Mantis-Instruct
```

`hf` 命令随 `huggingface_hub` 包提供，使用方式见 [Hugging Face CLI 文档](https://huggingface.co/docs/huggingface_hub/guides/cli)。在能联网的机器下载后，也可把完整子集目录复制到训练集群。下载命令临时关闭离线模式，不影响训练脚本默认离线加载模型。

转换与索引（官方转换代码报错时，可直接使用/examples/qwen/prepare_mantis_energon.py进行转换）：

```bash
python 3rdparty/Megatron-Bridge/examples/models/qwen/qwen3_vl/prepare_mantis_energon.py \
    --source-dir /public/home/yugeng/work/datasets/Mantis-Instruct \
    --output-dir /public/home/yugeng/work/datasets/mantis-energon \
    --max-samples-per-tar 1000 \
    --validation-fraction 0.01 \
    --num-workers 2
```

转换器要求源目录下存在子集目录，子集中包含 `train-*.parquet` 和 `train_images.zip`；自动解压图片，生成 `.jpgs` 和 ChatML `.json` 成员，再创建 Energon 元数据。它会跳过图片缺失或 `<image>` 占位符数量与图片数量不一致的行，需检查日志中的写入数量确实大于零。

## 3、模型准备

```shell
pip install modelscope

cd /workspace

modelscope download --model Qwen/Qwen3.8-27B --exclude "*.safetensors" --local_dir ./models/Qwen3.8-27B
```

或者[点击链接](https://modelscope.cn/models/Qwen/Qwen3.8-27B/files)，手动下载模型文件，上传至`/workspace/models/Qwen3.8-27B`路径下。

模型目录需要包含`config.json`、tokenizer、图像processor配置和chat template。当前recipe读取原生多模态结构，使用`load_weights=False`随机初始化训练，不会自动导入目录中的safetensors权重。

## 4、启动任务

### 4.1 参数填写

（1）配置hostfile文件

进入`/workspace/Megatron-bridge-das/examples/qwen`，填写`hostfile`文件，示例如下（多个节点填写多行）：

```shell
node01
node02
```

可以填写节点IP或者节点主机名，第一行作为master。SSH端口默认11463，可用`SSH_PORT`覆盖；训练通信端口默认29539，可用`MASTER_PORT`覆盖。

（2）根据任务填写`train_qwen38_vl_pretrain.sh`脚本

进入`/workspace/Megatron-bridge-das/examples/qwen`，根据上述文件路径，填写脚本中的变量，或在启动前导出：

```shell
export LOCAL_HF_PATH=/workspace/models/Qwen3.8-27B
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export TRAIN_ITERS=50
export SEQ_LENGTH=1024
export GLOBAL_BATCH_SIZE=8
export MICRO_BATCH_SIZE=1
export OPTIMIZER_OFFLOAD_FRACTION=0
```

脚本默认训练5步；设置`TRAIN_ITERS=50`后执行50步测试。默认BF16、TP4/PP2/CP1、GBS=8、MBS=1，语言、视觉和视觉投影均参与训练，MTP辅助分支关闭。

（3）选择recipe

| Recipe | 模型与数据 | 建议测试方式 |
| --- | --- | --- |
| `qwen38_vl_27b_pretrain_8gpu_smoke_bw1000_bf16_config` | 4层语言主干、27层视觉编码器，合成图文数据 | 单节点8卡，默认关闭optimizer offload |
| `qwen38_vl_27b_pretrain_bw1000_bf16_mock_config` | 完整64层语言主干、27层视觉编码器，合成图文数据 | 两节点16卡；单节点可显式开启25% optimizer offload |
| `qwen38_vl_27b_pretrain_bw1000_bf16_config` | 完整多模态模型，Energon图文数据 | 同上，额外设置`DATA_PATH` |

完整模型在单节点关闭optimizer offload时曾发生Adam状态分配OOM。单节点测试完整模型可显式设置`OPTIMIZER_OFFLOAD_FRACTION=0.25`；脚本与recipe默认均为0，不自动开启。

### 4.2 执行脚本

进入容器：

```shell
docker exec -it "your container name" bash
cd /workspace/Megatron-bridge-das/examples/qwen

bash train_qwen38_vl_pretrain.sh <RECIPE> <MODE> [HOSTFILE] [NODES] [KEY=VALUE ...]
```

参数说明：

```text
RECIPE: 上表中的recipe名称
MODE: dryrun或train，dryrun不需要HOSTFILE和NODES参数
HOSTFILE: hostfile文件路径，单节点时省略
NODES: 使用的节点数量，省略时使用hostfile中的全部节点
KEY=VALUE: 可选配置覆盖，例如optimizer.lr=1e-5
```

以下命令均在`examples/qwen`目录执行。

生成配置：

```shell
bash train_qwen38_vl_pretrain.sh \
    qwen38_vl_27b_pretrain_8gpu_smoke_bw1000_bf16_config dryrun
```

单节点8卡减层测试：

```shell
TRAIN_ITERS=50 bash train_qwen38_vl_pretrain.sh \
    qwen38_vl_27b_pretrain_8gpu_smoke_bw1000_bf16_config train
```

两节点16卡完整模型Mock测试：

```shell
TRAIN_ITERS=50 bash train_qwen38_vl_pretrain.sh \
    qwen38_vl_27b_pretrain_bw1000_bf16_mock_config train hostfile 2
```

单节点8卡完整模型Mock测试：

```shell
TRAIN_ITERS=50 OPTIMIZER_OFFLOAD_FRACTION=0.25 \
    bash train_qwen38_vl_pretrain.sh \
    qwen38_vl_27b_pretrain_bw1000_bf16_mock_config train
```

两节点16卡Energon测试：

```shell
TRAIN_ITERS=50 DATA_PATH=/workspace/datasets/qwen38-vl-energon-example \
    bash train_qwen38_vl_pretrain.sh \
    qwen38_vl_27b_pretrain_bw1000_bf16_config train hostfile 2
```

### 4.3 日志与checkpoint

输出默认保存在`examples/qwen/<recipe>_tp<TP>_pp<PP>_cp<CP>/`目录，可用`CHECKPOINT_PATH`或兼容变量`SAVE_DIR`覆盖：

```text
<recipe>_tp4_pp2_cp1/
├── <recipe>.yaml
├── tb_logs/
└── train_logs/mpi_日期-时间.log
```

`dryrun`日志位于`train_logs/dryrun.log`。`CHECKPOINT_PATH`只设置日志和配置输出目录，不自动启用模型checkpoint。`CHECKPOINT_DIR`和`LOAD_CHECKPOINT`默认均为`null`，需要保存或恢复Megatron checkpoint时分别指定目录。

