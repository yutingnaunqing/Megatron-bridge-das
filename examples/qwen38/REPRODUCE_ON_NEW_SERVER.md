# 在另一台 DCU 服务器复现本分支

本分支 `feat/qwen38-flash-next-dcu` 包含两条**不同**的 Qwen3.8-Flash-Next 文本测试路径：

1. [官方 Transformers 路径](DCU_TRANSFORMERS_REPRODUCE.md)：在 `yg_qwen3.8` 或等价的 PyTorch/HIP 容器里运行官方 `Qwen4ExpForCausalLM`，已覆盖单卡、八卡 DDP、八卡 TP。优先从这条路径开始。
2. [Megatron-Bridge 参考实现](README.md)：使用本仓库 `hcu_megatron_bridge/.../qwen38` 的文本骨干和 mock 数据，当前仅 DP，**不导入官方权重**。它依赖另一套 Bridge/Core/Energon 环境。

两条路径的通过记录都来自**缩小配置和随机权重**；拉取本分支不会获得官方 125B+51B checkpoint、`runs/` 中的随机 checkpoint 或训练日志。

## 1. 拉取代码

在新服务器的工作目录执行：

```bash
git clone --branch feat/qwen38-flash-next-dcu \
  https://github.com/yutingnaunqing/Megatron-bridge-das.git
git -C Megatron-bridge-das submodule update --init --recursive
cd Megatron-bridge-das
```

如果已经克隆了仓库：

```bash
git fetch origin feat/qwen38-flash-next-dcu
git switch --track origin/feat/qwen38-flash-next-dcu
git submodule update --init --recursive
```

`3rdparty/Megatron-Bridge` 是本仓库唯一被追踪的子模块。**不要复制旧机器的 `ckpt/`、`runs/`、根目录下未跟踪的 `3rdparty/Megatron-LM` 或 DeepSeek 实验输出**；这些不属于本分支。

## 2. 官方 Transformers 路径

把项目目录挂进能识别 BW1000 的容器，并确保容器内项目路径与下文命令一致。当前通过环境为 8 × BW1000 64G、Python 3.12.7、PyTorch 2.10.0、HIP 6.3.26113、Transformers 5.18.0.dev0。参考镜像标签：`10.16.1.152:5000/jenkins/model_test_env/mbridge:0.6.0-ubuntu22.04-dtk2604-py3.12-20260909-1017`；它是当前环境记录，不保证新服务器自动有权限拉取。

与本项目并列检出已验证的官方 Transformers 提交：

```bash
cd ..
git clone https://github.com/huggingface/transformers.git transformers_qwen38
git -C transformers_qwen38 checkout 8e6083374735b8c95f13b1cf2f80cd9c7486c389
cd Megatron-bridge-das
export PYTHONPATH="$(cd ../transformers_qwen38 && pwd)/src:${PYTHONPATH:-}"
```

若容器内工作目录不是 `/public/home/yugeng/work`，请改用实际挂载路径，并保持上述 `PYTHONPATH` 指向源码的 `src`。先执行 [完整复现手册](DCU_TRANSFORMERS_REPRODUCE.md)第 2 节的环境检查，再按单卡参考路径 → BF16 加速路径 → 八卡 DDP → 八卡 TP 的顺序执行。不要直接拿全量官方 checkpoint 运行本分支的随机权重训练脚本。

## 3. Megatron-Bridge 参考实现所需的额外源码

此路径在旧服务器的 `yg_mbridge` 容器中验证。除 `git submodule update --init --recursive` 外，旧环境还使用了以下**没有纳入本分支**的第三方源码版本：

| 源码 | 已验证提交 | 用途 |
|---|---|---|
| Bridge 子模块 | `51885cf132b2814188b6855c25a8588254274c2a` | 本仓库 gitlink 自动检出 |
| Bridge 内部 Megatron-LM | `43e45e13074f36da68b34c7a189cd997a01d1b41` | 与 Bridge 默认 gitlink `cd4afffa...` 不同 |
| Megatron-Energon | `0397ac7f653468851dc0991525c7bc4a51d58d8b` | 放在本项目 `3rdparty/Megatron-Energon` |

在项目根目录额外准备：

```bash
git -C 3rdparty/Megatron-Bridge/3rdparty/Megatron-LM fetch origin \
  43e45e13074f36da68b34c7a189cd997a01d1b41
git -C 3rdparty/Megatron-Bridge/3rdparty/Megatron-LM checkout \
  43e45e13074f36da68b34c7a189cd997a01d1b41

git clone https://github.com/NVIDIA/Megatron-Energon.git 3rdparty/Megatron-Energon
git -C 3rdparty/Megatron-Energon checkout \
  0397ac7f653468851dc0991525c7bc4a51d58d8b
```

这些命令只提供**已验证源码版本**，不能代替 DCU 版 PyTorch、Transformer Engine 等容器依赖的安装。随后按 [README.md](README.md) 中的 `tiny dryrun`、单卡训练、八卡 DP 顺序测试。此参考实现无需修改 Bridge 子模块的 `recipes/__init__.py`；旧机器该文件存在与 DeepSeek 有关的本地改动，未纳入本分支。

## 4. 判断是否成功

- 单卡 Transformers：日志末尾为 `PASS: reduced official text model forward/loss/backward/update`。
- 八卡 DDP：`PASS: 8-rank DDP forward/loss/backward/all-reduce/update`。
- 八卡 TP：`PASS: 8-rank native Transformers TP forward/loss/backward/update`，并在 `result.json` 看到 PLE 的 `colwise_gather_output` 和 `dtensor_parameter_count > 0`。
- Megatron 参考实现：`tiny`/`smoke` 运行正常结束，各 rank 写出 `result.rank<N>.json`。

原始 loss 不要求新服务器逐位一致。失败时先看具体 rank 的 `rank<N>_error.txt`、`rank<N>.json` 或 `train_logs/train.log`；`ChildFailedError` 通常只是汇总。请保存 `result.json`、运行命令、容器镜像标签、`torch/hip/transformers` 版本和源码提交，方便复查。
