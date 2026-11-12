# 保留 Conv3d 后的视觉 CUDA/HIP Graph 详细方案

> 最新执行：G1冻结patch Conv3d Graph已实现为opt-in（默认none）；101项定向测试通过，完整27层TP2的24项数值比较全部逐位一致。G0/G1各20步对照结果/进度见 `checkpoints/qwen38_graph_20261111/state.json` 与主handoff。下文未实施描述如与此冲突，以本条为准；G2及后续尚未实施。

日期：2026-11-11。状态：**方案，尚未实现或开启Graph**。当前Linear已备份并从活动训练路径移除；Graph必须以Conv3d的数值路径为准，不重新引入Linear或降低精度。语言M-RoPE不在范围内。

## 1. 目标与实现顺序

目标是减少host逐kernel调度开销，保留原计算算法、顺序、dtype与attention边界。平台为BW1000/DTK HIP；PyTorch仍使用torch.cuda命名，必须实测HIP Graph支持，而不能把NVIDIA路径存在视作本机已可用。

建议顺序与原因：

| 阶段 | 捕获范围 | 优先理由 | 第一版不包括 |
| --- | --- | --- | --- |
| G0 | 形状统计、计时、捕获可行性 | 确认Graph是否针对可见host瓶颈 | 不启用训练Graph |
| G1 | **冻结patch Conv3d forward** | 回退后每1568个patch仍约3137个kernel；该段不含TP collective，适合作为最小试点 | metadata、ViT decoder、merger、语言、optimizer |
| G2 | 冻结视觉MLP等可独立子模块 | 若patch Graph收益有限，验证TE小算子的调度成本 | attention仍eager，不更改帧边界 |
| G3 | 冻结ViT单层/层组，最后考虑整个decoder | 再覆盖QKV、RoPE、varlen attention、MLP与TP通信 | patch/merger先保持各自路径，不捕获语言PP通信 |
| G4 | 可选形状bucket、merger或训练反向扩展 | 只有前面正确且实测有益才做 | 不作为第一版验收条件 |

与先前Linear计划的区别：**G1现在值得首先评估patch本身**。保留Conv3d后，小kernel来源再次存在；无需先进入复杂的带TP通信decoder图。

Graph不会自动消除im2col/GEMM工作量。G1重放仍预计包含原有卷积kernel，目标是用一次Graph dispatch替代大量host launch；不能用“GPU kernel个数必须下降”作G1成功标准。Q/K联合旋转仍是另一个尚未实施的可选优化，不在本Graph试点中混入。

## 2. 基线与启动前检查

- 基线为Conv3d + 已验证的flash视觉RoPE；prepared metadata只在精度与单独性能验收后默认启用。最新状态见handoff及C0/C1报告，不使用含Linear的历史B2直接作为新Graph基线。
- 保持TP2/PP4、GBS64/MBS1、SEQ4096、真实mantis-birds、16语言层+27层完整ViT、冻结vision/projection、BF16，以及既有P2P配置。
- GPU节点连续空闲；记录实际torch/HIP/FA/TE版本及源码路径，使用既有进程级DTK兼容环境。
- 先核实graph API与一个无通信算子的capture/replay；再用真实Conv3d的形状/stride/dtype测试，错误需定位到实际库调用。
- metadata准备、CPU几何读取、Python分支、shape选择、H2D输入传输均留在capture外。不能在capture内调用 `.item()`、`.cpu()`、动态backend选择或设备同步。
- 当前 `batch_p2p_sync=true` 等语言PP同步保持原样，不能通过删除同步或更换卷积算法来制造Graph收益。

### 2.1 回退后已补齐的四阶段profile

C0/C1已各完成20步，metadata的数值一致和局部准备收益已验收；单节点当前默认Conv3d+flash+prepared。单步第5步快照显示：

| 非NCCL kernel并集，秒 | 首阶段rank0 | 第二阶段rank2 | 第三阶段rank4 | 最后阶段rank6 |
| --- | ---: | ---: | ---: | ---: |
| C0 reference metadata | 9.851 | 6.120 | 6.127 | 13.580 |
| C1 prepared metadata | 9.198 | 6.112 | 6.117 | 13.590 |

对应step span约15.05/15.10秒；rank0减负并未在这个快照中变成更短step，支持关注最后语言阶段与PP依赖。并集不是严格因果关键路径证明，不能认定唯一瓶颈；但应据此限制视觉Graph的收益预期。G1先验证局部调度改善，若全step不改善，不继续默认扩大范围。

完整C0/C1的稳态计时窗口有外部干扰与异常步，不能用mean原始差23.71%宣传metadata或Graph收益。详细限制见 [验收记录](qwen38-metadata-conv-20261111.md)。

## 3. 为什么不能直接打开现有 VisionTECudaGraphHelper

当前仓库有独立视觉helper入口，但不能简单设置开关：

1. Qwen35 provider目前没有完整声明Qwen3VL的vision CUDA graph专用字段，需先显式配置，不能动态塞不存在的字段。
2. 现有视觉padding分支丢掉PackedSeqParams，无法保证两图/多帧分别attention；尾部padding mask不能代替图像/帧边界。
3. helper样例输入未完整带入本路径的实际视觉RoPE表；默认最大长度可能由2304/4得到576，这不是merge前ViT总patch数上限。
4. helper以训练反向、每层×microbatch样例为中心；当前是冻结视觉，照搬可能增加大量无用buffer与Graph内存。
5. 当前flash/prepared与原视觉TE Graph组合被显式拒绝。不要整体删除保护或把所有模型的cuda_graph_impl打开。

首选在Bridge新增视觉局部wrapper，**全局/语言 `cuda_graph_impl` 仍为none**。G1无需调用旧padding分支，完全保留原packed setup。G3若复用TE helper，必须在Bridge适配真实静态输入与生命周期；遵守其RNG及capture约束。嵌入MCore只读，不修改。

## 4. G0：统计与形状选择

1. 利用已完成的Conv3d C0/C1训练prof，区分patch、metadata、decoder、merger、各PP阶段和设备同步，先看host launch间隙是否有实际压缩空间。
2. 从CPU metadata统计前几个warmup step的有序grids、总patch数、各帧长度、输入dtype/stride。统计不复制图像内容，不额外调用GPU scalar。
3. G1先只选择一个频繁出现的**精确形状**；其他形状eager，不在计时窗口临时capture长尾。
4. 后续是否增加Graph数量，由覆盖率与内存实测决定。不要为每个microbatch独立分配一套图。
5. 建立对比指标：patch host调用时间、Graph launch数、GPU区间时间、copy/clone时间、命中率、显存、完整step。若只有局部变快但step未改善，保留为可选，不默认扩大。

### 4.1 当前真实负载的形状分布

已从回退后C0第5步rank0的3456个视觉rotary launch反推64个microbatch的patch token数（BLOCK_M=4，每microbatch固定54次旋转；全部分组一致且merge2保证token数为4的倍数）：共 **17种**。

| 精确patch token数 | 本步microbatch次数 |
| --- | ---: |
| 1680 | 18 |
| 1728 | 9 |
| 1696 | 6 |
| 1632 | 6 |
| 1648 | 6 |
| 1700 | 4 |
| 1760 | 3 |
| 1776 | 2 |
| 其余9种 | 各1～2次 |

Top1覆盖28.125%，Top4覆盖60.9375%，Top8覆盖84.375%。可以先用N1680验证单图，再按实测内存评估4/8个精确形状的价值；这些是该步的覆盖率，不保证全训练数据保持一致。该推断只足以构造**patch Graph**的形状统计，不能从总token数推断attention帧边界，G3仍必须读取真实有序cu_seqlens。

原始数据：`/public/home/yugeng/work/checkpoints/qwen38_metadata_conv_20261111/graph_shape_histogram.json`，脚本同目录 `graph_shape_histogram.py`。这些统计不表示Graph已经捕获或已加速。

## 5. G1：冻结 patch Conv3d forward

### 5.1 不改变数值路径

调用原 `Qwen3VLVisionPatchEmbed.forward`，保留其输入view、到weight dtype的cast、Conv3d、bias与输出view。不用Linear，不合并/拆分bias，不切换FMA/TF32/FP8，不改变MIOPEN_FIND_MODE或算法设置。

权重与bias指针固定，图内只重放同一个已warmup的Conv3d路径。warmup需触发MIOpen算法/模块编译及workspace分配，不能在capture时首次搜索。捕获所用输入dtype与实际输入一致：若原pixel_values为FP32，保留图内原cast，不能提前换成另一套dtype语义。

### 5.2 Graph key与存储

G1 key至少包括设备、输入shape/stride/dtype、weight dtype/layout、conv descriptor、模型实例/权重存储版本、capture模式。patch运算不依赖H/W坐标，若两种几何的patch输入shape与布局相同，可以共用patch图；不要错误地把图绑定到图片内容。

每个条目拥有：

- 与真实shape和布局匹配的静态GPU输入buffer；
- 同一模型的冻结weight/bias引用；
- Graph持有的workspace/private pool与静态输出；
- replay完成事件、内存计数与使用统计。

执行顺序：实际输入copy到静态buffer → 确保copy依赖 → replay → 输出clone/copy到调用方拥有的存储 → 继续eager位置相加与decoder。copy/clone均计入收益。

第一版仅允许实际训练的已验证连续布局；不支持的stride直接eager。后续若支持非连续输入，需按真实stride建静态buffer并单独证明算法/精度不变。

### 5.3 捕获时机与回退

G1不含TP collective，允许在固定warmup阶段捕获；仍建议rank0/1使用相同的形状白名单。采集形状后在安全step边界预热/捕获，计时阶段不重新capture。

- 模式关闭、未命中、超预算、输入requires_grad、视觉未完全冻结：明确eager回退并记录原因。
- 捕获失败只在capture外清理并回退；不吞掉已经发生的设备非法访问后继续伪装成功。
- 一个图同一时刻只在受控stream使用，先不支持并发replay或多stream输入更新。
- 第一版固定常驻白名单，不在热路径动态LRU重建图。销毁必须等最后replay完成，不释放仍被GPU使用的内存。

## 6. G2/G3：视觉MLP与decoder层组

### 6.1 保留真实varlen边界

第一版严格**不padding**。Graph key至少含总token数、batch布局、完整有序帧seqlens、maxlen、dtype/device/stride、TP布局和层组范围；可先用完整grids作为保守key。

- hidden、FP32 cos/sin、int32 cu_seqlens由Graph持有独立静态buffer。
- max_seqlen为与该图一致的Python int；PackedSeqParams引用图的静态cu_seqlens，不在图内由GPU提取长度。
- 两个样本即使总token数相同，只要帧边界不同，第一版使用不同图。
- 相同shape的新图片、新位置表每次更新buffer；不能把图当成图像内容缓存。
- metadata LRU与Graph存储分离：从metadata条目copy到图拥有的buffer，不捕获可能被驱逐的缓存地址。
- 对未知signature eager，不为了命中而合并独立图像或改用全序列attention。

### 6.2 TP2捕获必须协调

G3图内包含QKV/MLP等TP collective时，rank0/1须按相同signature、相同调用顺序捕获和重放；不能一边capture另一边eager执行同一collective。

建议由Bridge callback在 `on_train_step_end` 的安全边界协调：所有PP rank结束当前step并确认无未完成P2P → capture前的全局同步在capture外 → 仅视觉TP组按同序warmup/capture → 其他PP rank等待 → 完成后共同继续。需确认callback所在点确实满足这些条件；不能只凭事件名称推断安全。

新形状、超内存、fallback策略须在TP组达成一致。Graph内的TP通信与图外语言TP/PP通信仍需保持原有次序；不得绕开设备同步保护或改变communicator。

### 6.3 逐步扩大范围

- 先一个层或一个MLP，验证FA varlen、TE、Triton与TP各自是否可capture。
- 若attention中仍有capture不允许的同步/动态选择，MLP Graph + eager attention作为缩小范围，不改attention语义绕过错误。
- 再选择小层组，比较重放次数、host收益、内存；最终才评估整个27层decoder。
- merger第一版eager；只有前面有收益且输出存储/权重约束明确时再考虑合并。
- 训练视觉的backward/recompute是另一个阶段，需要独立autograd与PP微批生命周期设计，不承诺冻结图可直接用于解冻。

## 7. 输出寿命、精度与内存约束

**冻结视觉不等于输出可随时覆盖。** 语言backward可能保存视觉特征用于语言参数梯度；下一microbatch replay覆盖静态输出会导致旧microbatch计算错误。

第一版所有跨wrapper输出均拷贝到消费者拥有的新存储。先验证多个不同输入forward后再依次backward，不仅验证“forward一次立刻backward一次”。正确后若优化clone，只能用显式ring buffer/完成事件及PP调度证明回收时机，不能以vision无梯度为理由复用全部输出。

Graph预算同时限制条目数、GPU reserved增量与host图节点内存。Conv3d图可能包含数千节点，不能只按输入输出张量估计内存。TE helper若按27层×64微批分配图，需证明必要性，否则使用冻结forward共享条目。

模型移动设备/精度、加载权重、切换requires_grad、改变conv设置或底层库版本时销毁/重建图。训练结束及异常退出均cleanup，不依赖长进程中Python GC偶然释放。

## 8. 配置与修改位置（均为拟实施项）

建议声明与现有全局cuda_graph_impl分离的视觉字段：

| 字段 | 意义 |
| --- | --- |
| `vision_graph_mode` | `none`（默认）、`patch_embed`、`mlp`、`decoder_group` |
| `vision_graph_warmup_steps` | 收集/预热窗口，必须在性能窗口前完成 |
| `vision_graph_max_entries` | 小规模试点先限制为单一signature，再由覆盖率决定 |
| `vision_graph_max_bytes` | 实测后显式设预算；不虚构默认显存收益 |
| `vision_graph_layer_group_size` | G3的层组范围，G1不用 |
| `vision_graph_clone_outputs` | 第一版固定true，不能为追求速度直接关闭 |

拟修改文件，Bridge相对 `src/megatron/bridge/models/qwen_vl/`：

- 新增 `modelling_qwen3_vl/vision_graph.py`：key、静态buffer、预热/capture/replay、输出所有权、计数与cleanup。
- `modelling_qwen3_vl/vision_model.py`：G1调用patch wrapper；G3调用decoder wrapper，原packed setup不变。
- `modelling_qwen3_vl/utils.py`：如需要，仅暴露原Conv3d调用给wrapper；不改变计算公式、参数或state_dict。
- `modelling_qwen3_vl/transformer_block.py`：仅当选择层组时增加局部wrapper入口。
- `qwen35_vl_provider.py`、`transformer_config.py`：声明/传递字段并校验冻结、dtype及不支持的组合。
- Bridge training callback/必要的 `training/train.py`：安全capture边界与结束cleanup，尽量复用现有callback机制。
- 根测试拟新增 `test_qwen38_vision_graph.py`，本地功能测试先1/2 GPU，8卡训练作为单独实验；不跑全套测试。

保留当前旧视觉TE Graph保护。新安全路径仅在对应mode下精准放行，不删除针对旧padding/错误组合的保护。

## 9. 测试和验收

### 9.1 正确性优先，不能只看capture成功

1. eager原路径重复控制；同权重、同输入比较eager/replay，真实BF16，目标为原dtype逐位一致，不因误差小就放宽到Linear的门槛。
2. 测同shape不同图片/位置、不同H/W但相同patch数量、多图、多帧、同总长不同帧边界、shape miss与预算fallback。
3. G1比较patch输出及后续完整27层ViT输出；G3比较各层组与最终输出。
4. 用冻结视觉+下游可训练模块比较梯度，多microbatch forward后再backward，证明输出没有被覆盖。
5. TP2检查rank0/1 capture、命中、fallback及collective顺序；再检查PP4，不出现死锁/非法访问。
6. 确认更新输入确实影响输出；检查没有误复用图像内容或过期权重。
7. 评估load_state_dict、设备/精度变换后的失效和cleanup。

### 9.2 性能比较

沿用当前偏好：一次eager、一次Graph，各20步，不自动跑三轮；prof5、默认比较10～20步，shape/capture阶段未结束则明确延后窗口，不把初始化混进稳态。

记录：host发射API减少、`hipGraphLaunch`等实际重放事件、capture次数、hit/fallback率、Graph内kernel、copy/clone成本、显存峰值、全step mean/median与逐步CSV。Graph重放的kernel数可与eager相同；不能把无GPU工作或全eager fallback当成功。

C0/C1四个PP阶段的prof用于确认rank0减负后有没有改变关键路径；如果只局部加速、整步没有明确收益，则保持默认none，不宣称训练加速。

### 9.3 阶段出口

- G1：原Conv3d数值保留，重放真实发生，内存可控；包含copy/clone后局部/整步结果可复核。
- G2/G3：G1或pilot证明host优化有价值；TP、varlen、输出生命周期全部通过，才扩大范围。
- 任一精度不一致、帧边界改变或梯度覆盖问题：停止该模式并保留eager，不能仅以loss接近验收。
- 若瓶颈主要在语言PP同步/其他stage，Graph不是通用解法；P2P同步本身必须另做正确性验证，不能借Graph试点顺带修改。

本轮交付仅该方案；Graph代码、配置字段和测试均尚未实施。
