# Qwen3.8-27B

[纯文本与 Energon 全参 SFT 单/双节点指南](SFT_HANDOFF.md)

## Supported
| Hardware | Recipe | Precision and optimizer |
|----------|--------|-------------------------|
| BW1101, 32 GPUs | `qwen35_vl_27b_pretrain_32gpu_bw1101_bf16_mock_config` | BF16 Adam |
| BW1101, 16 GPUs | `qwen35_vl_27b_sft_16gpu_bw1101_bf16_config` | BF16 Adam |
| BW1101, 4 GPUs | `qwen35_vl_27b_peft_4gpu_bw1101_bf16_config` | BF16 Adam |


## 1、环境准备

### 1.1 [光源社区](https://developer.sourcefind.cn/servicelist)拉取HCU基础docker镜像

### 1.2 创建容器

参考命令如下（请按需修改，正确挂载工作目录，修改正确镜像名称）：

```shell
docker run -it \
       --name megatron-bridge_test \
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
       -v /home:/home \
       REPOSITORY:TAG \
       /bin/bash
```

### 1.3 拉取代码仓

```shell
cd /workspace

git clone --recurse-submodules -b dev https://github.com/HYGON-AI/Megatron-bridge-das.git
```

### 1.4 安装依赖

```shell
cd /workspace/Megatron-bridge-das
pip install -r examples_hcu/requirements.txt
```

## 2、数据处理（pretrain不需要）

### 2.1 sft、peft

#### 数据集下载
```python
cd /workspace
hf download mmoukouba/MedPix-VQA --repo-type=dataset --local-dir ./MedPix-VQA
```
或
[点击下载](https://huggingface.co/datasets/mmoukouba/MedPix-VQA/tree/main)数据集。

假设下载后文件目录如下
```
├── data
│   ├── train-00000-of-00006.parquet
│   ├── train-00001-of-00006.parquet
│   ├── train-00002-of-00006.parquet
│   ├── train-00003-of-00006.parquet
│   ├── train-00004-of-00006.parquet
│   ├── train-00005-of-00006.parquet
│   └── validation-00000-of-00001.parquet
├── gitattributes
└── README.md
```

## 3、模型准备

```shell
pip install modelscope

cd /workspace

modelscope download --model Qwen/Qwen3.5-27B --local_dir ./Qwen3.5-27B
modelscope download --model Qwen/Qwen3.8-27B --local_dir ./Qwen3.8-27B
```

或者手动下载模型文件，上传至 `/workspace` 路径下：

- [Qwen3.5-27B](https://www.modelscope.cn/models/Qwen/Qwen3.5-27B/files)
- [Qwen3.8-27B](https://www.modelscope.cn/models/Qwen/Qwen3.8-27B/files)

## 4、启动任务

### 4.1 参数填写

（1）配置hostfile文件

进入`/workspace/Megatron-bridge-das/examples_hcu/qwen38_27b`，填写`hostfile`文件，示例如下（多个节点填写多行）

```shell
127.0.0.1
127.0.0.2
```

可以填写节点ip或者节点主机名。

（2）根据任务填写脚本变量并执行

#### pretrain
```shell
docker exec -it "your container name" bash
cd /workspace/Megatron-bridge-das/examples_hcu/qwen38_27b

PORT=${PORT:-12345} # Set the SSH port by running /usr/sbin/sshd -p <port> inside the Docker container on each node. For example: /usr/sbin/sshd -p 12345
export LOCAL_HF_PATH=${LOCAL_HF_PATH:-/path/to/Qwen3.8-27B}
export LOCAL_CORE_PATH=${LOCAL_CORE_PATH:-/path/to/Qwen3.8-27B-core}

./train_qwen38_27b_pretrain.sh <RECIPE> <MODE> [HOSTFILE] [NODES]

RECIPE: SUPPORTED_RECIPES in train_qwen38_27b_pretrain.sh
MODE: dryrun or train # 配置dryrun时不需要再跟HOSTFILE和NODES参数
HOSTFILE: path to hostfile
NODES: The number of nodes you use
```

#### sft
```shell
docker exec -it "your container name" bash
cd /workspace/Megatron-bridge-das/examples_hcu/qwen38_27b

PORT=${PORT:-12345} # Set the SSH port by running /usr/sbin/sshd -p <port> inside the Docker container on each node. For example: /usr/sbin/sshd -p 12345
DATA_PATH=${DATA_PATH:-/path/to/MedPix-VQA}
export LOCAL_HF_PATH=${LOCAL_HF_PATH:-/path/to/Qwen3.8-27B}
export LOCAL_CORE_PATH=${LOCAL_CORE_PATH:-/path/to/Qwen3.8-27B-core}

./train_qwen38_27b_sft.sh <RECIPE> <MODE> [HOSTFILE] [NODES]

RECIPE: SUPPORTED_RECIPES in train_qwen38_27b_sft.sh
MODE: dryrun or sft # 配置dryrun时不需要再跟HOSTFILE和NODES参数
HOSTFILE: path to hostfile
NODES: The number of nodes you use
```

#### peft
```shell
docker exec -it "your container name" bash
cd /workspace/Megatron-bridge-das/examples_hcu/qwen38_27b

PORT=${PORT:-12345} # Set the SSH port by running /usr/sbin/sshd -p <port> inside the Docker container on each node. For example: /usr/sbin/sshd -p 12345
DATA_PATH=${DATA_PATH:-/path/to/MedPix-VQA}
export LOCAL_HF_PATH=${LOCAL_HF_PATH:-/path/to/Qwen3.8-27B}
export LOCAL_CORE_PATH=${LOCAL_CORE_PATH:-/path/to/Qwen3.8-27B-core}

./train_qwen38_27b_peft.sh <RECIPE> <MODE> [HOSTFILE] [NODES]

RECIPE: SUPPORTED_RECIPES in train_qwen38_27b_peft.sh
MODE: dryrun or peft # 配置dryrun时不需要再跟HOSTFILE和NODES参数
HOSTFILE: path to hostfile
NODES: The number of nodes you use
```