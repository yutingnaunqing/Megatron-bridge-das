# Qwen3.8-27B HCU SFT 交接

更新于 2026-11-19。工作树 `Megatron-bridge-das-dev-qwen38-sft`，分支 `feat/qwen38-sft-dev`，基线 `5f202d7`，Megatron-LM `ba4951582`。review 的起始迁移依据为 `6f31f0d` 及迁移时的工作树。

## 项目目的

为客户交付 Qwen3.8-27B 后训练方案。首轮不要求多模态：以 64 层去视觉文本全参 SFT 为主，重点验证 256K 上下文下的多轮 Agent 和工具调用场景，并实测去视觉的显存和吞吐收益。训练数据优先使用来源可追溯、获准使用的官方发布数据；区分可用于训练的轨迹与仅用于评测的官方基准，交付可复现的测试方案。另保留原始 64 层 + ViT 的 VL 全参后训练方案，独立做 4K 图文验收。

## 待解决问题

| 事项 | 当前状态与下一步 |
|---|---|
| 分支训练实测 | 四个训练入口已有；本分支仅有减层文本单节点 CP1 完成记录。待验证双节点 16 层与 64 层、完整 VL 全参，以及 checkpoint 保存恢复。 |
| 多 CP 与 256K | 双节点文本入口已传 `CP`，但 DP 前置校验没计入 CP；待修正校验、先做 CP1/CP2 正确性对照，再逐档升长。256K 尚未跑通。 |
| Agent 数据与测试 | 待确认客户可提供的官方训练数据及授权，补齐工具调用/结果数据适配，固定评测集、运行协议和基线。 |
| 客户宿主机 | 宿主机节点间免密已配置，仍须用训练账号双向复核；HCU 驱动、设备权限、RDMA 和本地盘尚未核验。本轮先做下方宿主机检查。 |
| 非共享存储 | 节点之间无共享盘；待确认本地容量和副本分发方式，以及正式双节点 checkpoint 可用的共享存储。 |
| 镜像与训练环境 | 镜像待上传和校验，容器内设备、Python、MPI、RCCL 和 GPU 训练尚未验证；不在本轮宿主机检查范围内。 |

## 实现和改动位置

- [qwen38_sft_recipes.py](qwen38_sft_recipes.py) 调用 dev 的 `AutoBridge.get_model_config()`，使用当前 Bridge builder。新建文本与 Energon VL 两条 recipe，共用 BF16、FP32 Adam、训练和 checkpoint 配置。数据、tokenizer 和前向 step 按模型选择。
- [qwen38_sft_worker.py](qwen38_sft_worker.py) 按主机和 local rank 隔离 JIT 缓存，绑定本地设备，调用 dev 自带的 `recipe_runner` 和 `finetune`，保存配置及 rank 退出记录。
- 四个 Bash 入口各自定义模型、数据、训练、并行、checkpoint 和 profiler 参数。各脚本以 Bridge 的 `key=value` 参数数组向 Python worker 传参。单节点直接启动本机 agent，双节点入口用 hostfile 启动每节点一个 MPI 进程，再运行同一套训练参数。HCU 网络仍由 `examples_hcu/env.sh` 配置。
- 四个入口及 [hostfile 模板](hostfile.sft.example) 见下节。原有 MedPix SFT 入口仍可独立使用。
- [HF 纯文本转换](convert_qwen38_vl_to_text.py)、[通用 JSONL/Parquet 数据准备](prepare_qwen38_text_sft.py)、[UltraChat 数据准备](prepare_ultrachat_sft.py) 沿用 review 工具，可直接读取现有导出权重和训练数据。
- 新流程所有源文件位于 `examples_hcu/qwen38_27b/`。它在进程内直接调用上游 runner API，启动时无需 apply patch。Core 保持固定版本，采用 dev 已有的 GPT MTP loss mask 和 Qwen 权重映射。

## 启动

在挂载该工作树和输入数据的 HCU 容器内，从仓库根目录执行。使用现有容器里的 Python 环境，或用 `PYTHON_BIN` 指向对应的解释器。全部四个脚本的默认模式为 `dryrun`。双节点 dryrun 使用目标 world size 校验配置，在本机执行。双节点实际训练要求两边容器内的代码、HF 权重和数据路径一致，可使用经过校验的本地副本；分布式 checkpoint 则需要两节点共同可见的存储。两边还需通过 MPI 使用的 SSH 端口互通。传入相对 hostfile 时，先按命令执行目录查找，再按仓库根目录查找。从脚本目录运行时可以直接传 `sft hostfile`。

四个入口默认使用工作树同级的 `models/` 和 `datasets/`。资源放在其他位置时可用 `MODELS_DIR`、`DATASETS_DIR` 覆盖根目录。

```bash
workflows=examples_hcu/qwen38_27b

# 先检查配置。将 dryrun 换成 sft 即启动单节点训练。
bash "$workflows/train_qwen38_27b_text_sft_single_node.sh" dryrun
bash "$workflows/train_qwen38_27b_vl_sft_single_node.sh" dryrun

# 复制模板，将两行替换为可访问的节点名，再启动双节点训练。
bash "$workflows/train_qwen38_27b_text_sft_dual_node.sh" dryrun
bash "$workflows/train_qwen38_27b_vl_sft_dual_node.sh" dryrun
bash "$workflows/train_qwen38_27b_text_sft_dual_node.sh" sft /path/to/hostfile
bash "$workflows/train_qwen38_27b_vl_sft_dual_node.sh" sft /path/to/hostfile
```

| 入口 | 层数 | 8 卡/节点的默认布局 | 训练默认值 |
|---|---:|---|---|
| [文本单节点](train_qwen38_27b_text_sft_single_node.sh) | 16 | TP2 / PP4 / DP1 | 4K、MBS1、GBS32、50 步、无重计算 |
| [文本双节点](train_qwen38_27b_text_sft_dual_node.sh) | 64 | TP2 / PP4 / DP2 | 4K、MBS1、GBS64、500 步、full 重计算 |
| [VL 单节点](train_qwen38_27b_vl_sft_single_node.sh) | 64 + ViT | TP4 / PP2 / DP1 | 4K、MBS1、GBS32、50 步、full 重计算，optimizer CPU offload 比例 0.5 |
| [VL 双节点](train_qwen38_27b_vl_sft_dual_node.sh) | 64 + ViT | TP4 / PP2 / DP2 | 4K、MBS1、GBS32、500 步、full 重计算 |

所有入口默认 CP1；当前工作树只有文本双节点入口暴露 `CP`，尚无该入口 CP>1 的实跑结果。文本单节点可用 `EXPECTED_NUM_LAYERS=64` 和完整 CausalLM 权重切换到 64 层，此时默认 full 重计算和 0.5 optimizer CPU offload。VL 两个入口都要求完整 HF 权重，语言模型、视觉塔和投影均训练。单节点 64 层和 offload 的配置需要在目标卡及主机内存条件下验收。单节点默认关闭保存，双节点默认每 250 步保存。若需要保存单节点训练，用 `SAVE_INTERVAL=50` 或所需间隔覆盖。

文本默认复用与 `SEQ_LENGTH` 匹配的 UltraChat messages JSONL，loss 只监督 assistant。VL 默认复用 `mantis-birds-energon-v1`，使用实际图文 batch 和 `qwen3_vl_step`，默认无 held-out 验证。用 `LOCAL_HF_PATH`、`DATA_PATH` 可显式覆盖本机的模型和数据目录。

文本脚本接受 `convert` 和 `prepare`，适用于生成新目标目录。MTP 默认 0，启用时设 `MTP_NUM_LAYERS=1` 并指向包含相应 MTP 权重的文本目录。现有 64 层带 MTP 的权重可直接使用。首次训练从 HF 初始化，通过 `RESUME_FROM=/path/to/checkpoints` 显式恢复 dev 生成的训练状态，包括 optimizer 和 RNG。跨 Bridge/Core 直接恢复旧的训练状态仍需验证。

训练默认值可直接在对应脚本的配置数组中编辑，也可在命令前按需覆盖。常用覆盖项包括 `SEQ_LENGTH`、`TP`、`PP`、`TRAIN_ITERS`、`WARMUP_ITERS`、`RECOMPUTE`、`PROFILE=true`、`MPI_SSH_PORT` 和 `MASTER_PORT`。输出默认在 `nemo_experiments/` 下，可用 `SAVE_DIR` 指定。配置解析写出 `resolved_config.yaml`，训练日志在 `train_logs/`，Profiler trace 在 `torch_profile/`。建议每个独立试验使用独立输出目录。

## GDN 和长序列记录

review 的 16 层、4K、CP1 FlashTrain 对照中，FLA/FlashTrain 的稳态平均步耗时为 7.5140/7.4977 秒，样本吞吐约增加 0.22%。算子有可测收益，端到端基本持平。此次 dev 的两条主流程使用原生 FLA。dev Core 自带的前处理融合可通过 `GDN_PRE_FUSION=true` 单独比较，与 review 的 FlashTrain 递推替换属于不同机制。

review 的独立 CP 长序列结果不能计入本分支验收。已通过的 16 层 CP2 8K/16K 和 CP8 8K、32K 故障及 CP1 对照限制，保留在原项目的 [handoff](../../../Megatron-bridge-das-qwen38-sft-review/docs/models/qwen/qwen38-text-sft-handoff.md)。

## 验证记录

- 用真实本地 HF 和数据目录在 HCU 容器执行四个入口的 dryrun，另测文本双节点 MTP1 和 profiler，以及文本单节点 64 层和 0.5 CPU offload。六项退出 0，写出了目标 DP、层数、GDN 开关及 VL 三项解冻配置，PyTorch GPU 分配计数为 0。
- 12 项启动参数探针通过，覆盖四个 dryrun 入口、四个训练入口的本地及模拟 MPI 参数、convert/prepare 参数，以及缺 hostfile / 无效 batch 的报错。实际跨节点 GPU 启动仍待验证。
- 原始 Qwen3.8 HF 权重完成 64 层 MTP1 转换器 header 检查，识别 866 个待导出张量。真实 UltraChat 首轮小样本准备写出 2 条训练和 1 条验证消息，使用 dev 的 tokenizer 和 collator。
- dev Energon loader 读取 Birds 的真实 4K batch，`input_ids` 为 `[1,4096]`，有 78 个监督 token，并输出 `pixel_values` 和 `image_grid_thw`。
- 上述 dryrun 日志和配置位于 [验证目录](../../nemo_experiments/qwen38-dev-validation/)。后续本分支已实跑文本单节点 8 卡、16 层、4K、TP2/PP4/CP1、GBS32：`GDN_PRE_FUSION=true` 的 [日志](../../nemo_experiments/qwen38-text-1node-gdnpretrue/20261119-153926-4052118/train_logs/node-0.log) 到 50/50，8 个 rank 均记录成功，未保存 checkpoint；关闭融合的 [日志](../../nemo_experiments/qwen38-text-1node/20261119-150843-3977197/train_logs/node-0.log) 可见至 40/50，不记作完成。双节点 GPU 训练、多 CP、64 层权重训练、VL 全参训练和保存恢复仍待实测。

### 2026-11-19 启动脚本独立化

将原共享 Bash launcher 拆为四个独立入口并移除共享文件。每个入口的训练默认值直接写在对应的配置分区，沿用同名环境变量作可选覆盖。Python worker 接收显式路径和 Bridge override，recipe 负责从这些配置推导 SP、offload、校验集和 checkpoint 字段。两个双节点入口各自生成 hostfile slots 和 MPI 命令，按参数传递模型、数据和训练设置，MPI 环境只保留运行库、通信及离线设置。

重跑 HCU 容器内四个默认 dryrun、文本双节点 MTP1 加 profiler、单节点 64 层文本，共六项通过。逐项与迁移后、拆分前的 `resolved_config.yaml` 对照，差异限于输出路径、视觉配置中的 Transformers 版本元数据及测试过程中的数值序列化。16 项启动探针覆盖独立入口、模拟 MPI rank 与参数传递、带空格的路径、错误状态传播及工具子命令。用独立文本双节点脚本完成本地 UltraChat 2 条训练、1 条验证样本准备。产物见 [重构验证目录](../../nemo_experiments/qwen38-independent-launch-validation/)。未启动 GPU 训练。

### 默认路径及环境修复

四个入口统一补齐模型和数据的同级目录默认值。文本入口的 `convert` 与 `prepare` 复用对应根目录。输入检查失败时打印具体模型配置和数据路径。修正本工作树 `examples_hcu/env.sh` 中来自 review 的 recipe 复制操作和 Python 导入路径，使用 dev 的 `src/` 和 `3rdparty/Megatron-LM/` 布局，保留现有 HCU 网络配置。

九项启动探针通过，覆盖缺省路径、文本工具输入和缺失路径报错。HCU 容器从脚本目录执行无路径覆盖的单节点文本 dryrun 通过，实际解析为 16 层权重和 4K UltraChat。未启动训练。

## 客户节点部署与验收

主线交付是 64 层去视觉文本全参 SFT，目标为单条真实长轨迹的 256K 训练和 Agent 评测。保留 64 层原始 VL 权重和全参 VL 入口，独立验收，不把 256K VL 纳入首轮。官方模型卡的原生上下文为 262,144 token，不等于本分支已支持该长度的训练，也不代表去视觉后已有实测性能增益。

### 当前步骤：宿主机检查

把 [check_qwen38_host.sh](check_qwen38_host.sh) 放到两台客户主机，用实际训练账号分别运行，不要用 `sudo`。脚本不安装软件、不改系统配置、不写入待检查目录，也不启动容器。将示例中的路径、SSH 目标和训练网 IP 替换为现场值；在第二台执行时将对端换回第一台。只做本机盘点可省略 `--peer` 和 `--data-peer`。`--storage` 可重复，不传时检查当前目录与 `/tmp`。

```bash
set -o pipefail
bash check_qwen38_host.sh \
  --storage /data/models --storage /data/datasets --storage /data/runs \
  --peer node-b --ssh-port 22 --data-peer 192.0.2.2 \
  | tee "hostcheck-$(hostname -s).log"
```

| 检查项 | 如何判读 |
|---|---|
| OS、内核、CPU / NUMA、RAM | 两台的版本与资源配置记录下来；检查可用内存、`/dev/shm`、打开文件与锁页限制。明显差异先确认是否符合客户环境设计。 |
| HCU、驱动、设备权限 | 用 PCI 枚举及现场可用的 `hy-smi` / `amd-smi` / `rocm-smi`、`rocminfo` 核对预期卡数（本方案 8 卡/节点）、型号、显存、拓扑；训练账号应能访问所需的 `/dev/kfd`、render 和 RDMA verbs 设备。未安装诊断工具记为待核验，不等于卡或驱动缺失。 |
| SSH 和训练网络 | 宿主机 SSH 默认用 22 或现场端口，不套用容器端口。两边都应返回预期对端主机名；脚本只用已有可信主机密钥，首次接入请先通过可信渠道核验指纹。检查 `ip route get` 的源 IP 和出口网卡；核对 RDMA 端口状态、速率、link layer 与 GID 索引/网口映射。路由和链路状态正常不代表端到端 RDMA 或带宽已验证。 |
| 存储与挂载 | 比对各目录的实际设备或挂载、容量、inode 和账号权限，按权重、数据、缓存及预留 checkpoint 估算。两边目录同名不表示共享；本脚本不创建文件，不能证明共享可见性、配额余量或写入吞吐。 |

先收集两份日志，重点解决卡数/设备权限、SSH、数据网卡/RDMA 和磁盘容量问题。`[WARN]` 表示具体异常或探针失败，`[SKIP]` 表示缺少工具；进程返回 0 只表示采集结束，不是训练通过。报告包含主机名、IP、账号和挂载信息，对外分享前先脱敏。参考 [RCCL 系统信息采集](https://rocm.docs.amd.com/projects/rccl/en/latest/how-to/troubleshooting-rccl.html)。

### 镜像、存储和通信

1. **固定交付物。** 记录 Bridge HEAD、Megatron-LM SHA 和实际工作树快照；仅 `git archive HEAD` 会漏掉当前未提交的 launcher、recipe 和工具，也不包含子模块内容。以经审核的快照或内网仓库交付，另带上模型和数据的 revision、转换参数与 SHA256 manifest。客户 hostfile、SSH 私钥、凭据和机器配置不要打进镜像或提交仓库。
2. **分发同一镜像。** 能用客户内网仓库时，两节点按同一 digest 拉取。离线时，源端 `docker image save -o qwen38-sft.tar <image:tag>`，生成 `sha256sum qwen38-sft.tar > qwen38-sft.tar.sha256`，经批准的通道传到两节点，在各自目录 `sha256sum -c qwen38-sft.tar.sha256` 后执行 `docker image load -i qwen38-sft.tar`，核对 `docker image inspect` 的镜像 ID。导出前确认训练依赖已在镜像里，而不是只存在于旧容器的改动或挂载里。记录 HCU 型号和显存、宿主驱动、DTK、PyTorch、RCCL、Open MPI、关键扩展与镜像 ID；先验证兼容性，再开训练。
3. **本地副本和共享输出分开。** 两边将本地盘映射到同样的容器路径，例如 `/train/repo`、`/train/models`、`/train/data`、`/train/cache`；显式指定 `LOCAL_HF_PATH`、`DATA_PATH`、`QWEN38_CACHE`、`QWEN38_TMPDIR`。数据和权重可 `rsync` 后按清单校验。对需要跨 rank 可见的数据索引/缓存，先用匹配的 tokenizer 和配置预生成并同步完整产物，或将相应目录放共享盘；不要依赖 rank 0 在节点 A 上写出的文件自动出现在 B。检查容量、inode、写权限和缓存命中。
4. **checkpoint 优先共享。** 正式双节点任务将 `SAVE_DIR` 指向两边同路径、互相可读写的客户共享卷（如客户已有并行文件系统或批准的 NFS）。双向写读探针通过后，再验收 `torch_dist`；两侧同名本地目录不是共享存储，把两边同名分片事后拼接也不作为受支持恢复方案。无共享卷时，仅以 `SAVE_INTERVAL=0` 做双节点短跑，每次取唯一 `SAVE_DIR`，结束后分别归档两节点日志及所有 rank JSON；不要宣称保存恢复已交付。单节点可以独立用本地盘做保存恢复验证。
5. **分层验网。** 宿主机免密已配置；仍要从实际发起环境用训练账号在 MPI 使用的 SSH 端口执行 `ssh -o BatchMode=yes -p <port> <host> hostname`，验证落点能看到训练容器同样的设备、目录、解释器及库；launcher 默认端口 11463，可用 `MPI_SSH_PORT` 覆盖。先跑 MPI 每节点一个进程的 `hostname`/环境探针，再在各节点验证设备枚举、BF16 小算子和扩展导入。按客户 HCU 文档配置设备映射和 RDMA 权限；检查 `MASTER_ADDR`/`MASTER_PORT` 与数据面网卡/RDMA 的连通，SSH 通不代表 RCCL 通。核对 `examples_hcu/env.sh` 的 `IB_TYPE`、HCA、GID、网卡、插件和拓扑文件，不直接照搬原节点的设备名。该脚本在节点侧重新 source；以远端进程实际看到的环境和 RCCL 日志为准。先跑容器内单节点，再跑双节点 `all_reduce_perf`，记录正确性、带宽与传输类型；TCP 只做隔离诊断，不作 RDMA 性能验收。

### 训练顺序和通过条件

以下按 2 台 × 8 卡给出。先复用现有短数据和 16 层权重隔离环境问题，再换完整权重、并行或长度。每组先跑 `dryrun` 检查 `resolved_config.yaml`；短跑用 `TRAIN_ITERS=5 WARMUP_ITERS=1 EVAL_ITERS=0`，关闭 profiler，先用 `SAVE_INTERVAL=0`。每个阶段只有满足通过条件才进入下一阶段。

| 阶段 | 起点 / 候选配置 | 通过条件 |
|---|---|---|
| 单节点复现 | 两台分别跑文本 16 层、4K，TP2/PP4/CP1；先用已完成的 `GDN_PRE_FUSION=true` 路径 | 各 8 rank 成功，完成 HF 加载、前后向和 optimizer step，无非有限 loss 或设备错误 |
| 双节点冒烟 | 文本 16 层、4K、TP2/PP4/CP1/DP2，MBS1/GBS2 | 16 rank 全部退出成功，确认跨节点通信，连续 5 步；与单节点对照首步 loss 和监督 token 数 |
| 文本全参短跑 | 文本 64 层、4K，维持双节点 CP1，MTP0，full 重计算；再打开 held-out 验证和共享 checkpoint | 先训练通过，再保存、退出并用同拓扑 `RESUME_FROM=<save-dir>/checkpoints` 恢复；核对步数、消费样本、optimizer、LR 和 RNG 状态，及训练能继续 |
| CP 正确性 | 文本双节点：先 16 层 4K 对照 CP1/CP2，保持 TP2/PP4、相同样本、seed、MBS1/GBS2；再换 64 层、4K | CP2 时 DP1；比较首步 loss、有效监督 token 数和小步 loss，容差按 BF16 重跑波动预定；查前后向、显存和 GDN 路径，无 hang/OOM |
| 长度爬坡 | 64 层 CP2 从 8K → 16K → 32K → 64K → 128K → 256K，逐级试跑。遇显存上限先用较短档验证 TP2/PP2/CP4/DP1 等候选拓扑，再提升长度 | 每档先 1–5 步，再做稳态性能采样；验证真实非 padding token、末端监督 token、无截断/溢出，检查 loss 和 rank 退出；256K 另做 checkpoint 恢复及持续运行 |
| VL 全参备用 | 原始 64 层 + ViT 权重、Energon Birds，4K、TP4/PP2/CP1/DP2，MBS1/GBS2 | 真实图文前后向；确认语言模型、视觉塔、投影均解冻且有梯度或参数更新；再测保存恢复。Birds 是链路冒烟，不代表最终视觉质量 |

文本双节点一期参数示例：在启动容器中预设 `LOCAL_HF_PATH` 为 16 层 CausalLM 权重、`DATA_PATH` 为 4K 数据、`SAVE_DIR` 为本次运行目录，另设 `HOSTFILE` 为已验证的两节点文件。

```bash
entry=examples_hcu/qwen38_27b/train_qwen38_27b_text_sft_dual_node.sh
EXPECTED_NUM_LAYERS=16 SEQ_LENGTH=4096 TP=2 PP=4 CP=1 \
  MICRO_BATCH_SIZE=1 GLOBAL_BATCH_SIZE=2 GDN_PRE_FUSION=true \
  TRAIN_ITERS=5 WARMUP_ITERS=1 EVAL_ITERS=0 SAVE_INTERVAL=0 \
  bash "$entry" sft "$HOSTFILE"
```

**当前 CP 启动缺口：** 文本双节点 Bash 已传 `CP`，但前置校验仍按 `TP*PP` 算 DP。稠密模型规划应按 `DP=WORLD_SIZE/(TP*PP*CP)`，并校验 world size 可整除、`GBS % (DP*MBS) == 0`、模型序列长度可被 `2*CP` 整除；目前 CP2/CP4 探针选 GBS2 可同时通过旧检查和实际 DP 约束，正式支持应修正 launcher 并补启动校验测试。CP4 候选同时改变 PP，不能拿来做只改变 CP 的性能对照。两节点 256K 若跑不稳，报告实测上限、显存及网络瓶颈，再决定增节点或补 kernel/拓扑支持，不承诺靠改 `SEQ_LENGTH` 就能达标。

### 数据与 256K Agent 质量验收

- **训练集：** 优先选客户批准、发布者可追溯、授权明确的 Agent SFT 轨迹，固定版本和模板；若客户指 Qwen 官方原始后训练数据，目前未从核对的官方模型卡确认其公开下载，需另向客户确认可提供的数据。UltraChat 是短文本冒烟；压力数据独立准备，用真实长轨迹和单条可控的合成长上下文补齐边界覆盖，不混入质量评测集。
- **SFT 适配是前置任务：** 当前 `prepare_qwen38_text_sft.py` 只接受字符串 `system/user/assistant`，不能原样保留 `tools`、`tool_calls`、`tool` 结果。后续先用现有 Bridge 对话处理能力验证或补齐转换器：保留 schema、调用、结果及多轮顺序；检验 Qwen 模板渲染、thinking 设置、assistant-only loss mask、工具结果不计 loss、尾部响应没有被截断。用训练实际 tokenizer/模板统计总长、非 padding 和监督 token，并按任务/仓库或原始轨迹分组去重切分。
- **官方评测：** 用 [LongBench v2](https://github.com/THUDM/LongBench) 测长上下文理解，用 [τ-bench](https://github.com/sierra-research/tau2-bench) 文本任务测多轮工具与任务成功；客户明确是 coding agent 时，再用 [SWE-bench Pro](https://huggingface.co/datasets/ScaleAI/SWE-bench_Pro) 的固定官方版本/协议。评测数据和任务衍生轨迹均不得进入 SFT。若只跑子集，报告样本数和版本，不宣称复现官方榜单分数。
- **单独验 256K Agent：** 官方长上下文任务与 Agent benchmark 分别对照，再用固定任务在不同工具历史长度桶做联合测试。输入预算按 `262144 - 预留生成 token` 算，系统提示、工具 schema、历史和生成共用窗口；记录重渲染 token、截断率和尾部证据位置。对比原始官方模型、去视觉未训和去视觉 SFT 三组，固定任务、模板、推理后端、thinking、工具环境、采样及预算，报告准确率、任务成功率、工具参数合法率和超时/截断率。训练侧记录步时、有效 token/s、峰值显存/主机内存及通信日志。去视觉的性能收益用同文本负载的可比结果报告；真实图文 VL 单独报告，不混比。

参考：[Qwen3.8-27B 官方模型卡](https://huggingface.co/Qwen/Qwen3.8-27B)、[Docker 镜像导出](https://docs.docker.com/reference/cli/docker/image/save/)、[Docker 镜像导入](https://docs.docker.com/reference/cli/docker/image/load/)、[ROCm 容器设备映射](https://rocm.docs.amd.com/en/latest/install/docker-containers.html)、[RCCL 环境验收](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/system-setup/prerequisite-system-validation.html)。
