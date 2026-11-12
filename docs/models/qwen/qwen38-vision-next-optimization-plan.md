# Qwen3.8-27B 视觉性能优化执行方案

最新状态：Linear已按精度优先要求撤回，相关代码/配置/旧审计工具备份于 `/public/home/yugeng/work/checkpoints/qwen38_linear_backup_20261111`。本文Linear结果与命令为历史记录；当前Conv3d+metadata单独验收见handoff第0节，Graph最新方案见 [CUDA/HIP Graph方案](qwen38-vision-cuda-graph-plan.md)。当前审计脚本已不含Linear或bias-followup，重用历史实验须先审阅备份恢复步骤。


日期：2026-11-11。状态：S1 patch Linear、S2 metadata第一版已实现且三组训练完成；S3 Q/K联合、S4视觉Graph尚未实施。实际结果见 [handoff](../../../HANDOFF_QWEN38_PERF.md)。范围：**patch Linear → metadata 准备 → Q/K 联合旋转 → 视觉 Graph**；语言 M-RoPE 不在本轮范围内。项目背景和历史问题见 [交接文件](../../../HANDOFF_QWEN38_PERF.md)。

## 1. 目标、范围和阶段基线

以当前已验证的 `vision_rope_backend=flash` 为起点，逐项降低视觉路径的低效计算、CPU/GPU同步和kernel发射开销。先验证数学与训练语义，再检查kernel结构，最后比较端到端训练。

| 阶段 | 起点 | 本阶段唯一新增变化 | 验收后的配置 |
| --- | --- | --- | --- |
| B0 | 现有flash视觉RoPE | 无，现有实现作为起点 | flash + Conv3d + 原metadata + eager |
| S1 | B0 | patch embedding改等价Linear | B1 |
| S2 | B1 | CPU几何metadata与有界缓存 | B2 |
| S3 | B2 | Q/K联合rotary kernel | B3 |
| S4 | B3 | 冻结视觉forward Graph | B4，只有实测有收益才启用 |

独立保留旧路径用于回退。某阶段正确性失败，停止该阶段并修复；某阶段粗测未显示收益，记录为未证实，不把它默认叠入下一阶段。S3的正确性若通过但收益不明确，S4仍可同时兼容单独Q/K和联合Q/K，以上一份已确认更好的配置为基线。

统一使用当前实际负载：bw45 / yg_qwen38，8卡，TP2/PP4/CP1，GBS64/MBS1，SEQ4096；16语言层+27视觉层，BF16，冻结vision/projection，真实mantis-birds数据。保持seed、数据顺序、processor像素范围、LR schedule、parallel/overlap/recompute/offload等一致。该负载不是完整64语言层的27B训练。

不自动开启其他fusion/overlap，不更换attention backend、不改变图片/帧间可见性，不修改语言路径或嵌入MCore源码。

## 2. 优先级依据

数据来自现有flash rank0的第5步trace，属于profiler观测值，不能直接当作端到端可消除时间。

| 热点 | 已观测开销 | 方案针对的成本 |
| --- | --- | --- |
| patch Conv3d | 108916个im2col+108916个小GEMM，GPU sum约1589.740ms，占rank0 kernel数75.5% | 改为批量矩阵运算，消除逐patch分解 |
| 首层前位置/metadata | 64个microbatch，每次32个scalar读取；窗口总计约637.861ms，含等待 | 减少隐式GPU标量读取、Python循环与重复几何准备 |
| 已融合视觉RoPE | 3456个rotary，GPU sum48.571ms | Q/K联合launch、减少dispatch |
| 剩余host调度 | 完成前三阶段后重新测量 | Graph仅针对仍有价值的host供给开销 |

Graph放在最后：S1/S2可能已大幅减少小kernel和CPU工作，先完成它们才能合理选择capture范围；原有视觉Graph路径也存在多图语义问题。

## 3. S1：patch embedding 等价 Linear

### 3.1 修改位置

Bridge相对路径前缀为 `src/megatron/bridge/models/qwen_vl/`：

- `modelling_qwen3_vl/utils.py`：`Qwen3VLVisionPatchEmbed.__init__/forward`，主要计算修改处；当前类约第33行。
- `modelling_qwen3_vl/transformer_config.py`：字段及provider→视觉config传递。
- `qwen35_vl_provider.py`：当前Dense provider声明配置；共享模型接口如同时支持MoE，字段保持一致但不扩展本轮验证范围。
- 根仓库HCU recipe、`examples/qwen/train_qwen38_vl_pretrain.sh`：必要的环境/CLI透传与枚举校验，沿用既有模式。
- 定向测试：根仓库 `tests/unit_tests/hcu/models/test_qwen38_vision_patch_embed.py`（拟新增）。

拟配置 `vision_patch_embed_backend=conv3d|linear`，初始默认conv3d，实验显式开启linear。配置名以实现时检查无冲突后确定，不用动态setattr制造未声明字段。

### 3.2 计算等价与实现步骤

当前每个输入patch原本就对应一个 `[3,2,16,16]` 小立方体，Conv3d的kernel与stride均为`[2,16,16]`，padding为0、dilation为1、groups为1，每patch只输出一个位置。因此可等价表示为：

```python
# 方案示意，尚未执行
x = hidden_states.reshape(-1, 3 * 2 * 16 * 16)
w = self.proj.weight.reshape(1152, -1)
y = F.linear(x.to(dtype=w.dtype), w, self.proj.bias)
```

实施时使用config维度，不硬编码上述数字。

1. 明确输入flatten顺序等于原 `view(N,C,T,H,W)` 的C/T/H/W顺序；输出仍为 `[N,1152]`，保持patch顺序。
2. 保留原 `self.proj = nn.Conv3d(...)` 及其weight/bias参数，仅切换forward算法，确保state_dict、HF转换、optimizer参数引用不变。
3. weight reshape保持view及autograd连接；不注册第二份参数、不缓存可训练weight的detach副本。
4. dtype转换跟随原weight和autocast策略；保留输入reshape/cast成本，测量时一并计入。
5. linear模式校验本等价前提；不支持的conv结构明确报错。conv3d模式保留原实现，不静默降级后仍标称linear。

### 3.3 正确性与结构验收

- 用同一份state_dict和输入比较FP32/BF16输出；在trainable模式比较输入、weight、bias梯度，并检查保存/加载后的参数名称和数值。
- 覆盖单patch、实际token规模、非连续输入、非方图与多帧对应的flatten输入。比较下游小型真实ViT的输出和梯度，不能只验证Linear算子本身。
- 归约算法变更可能带来低精度差异，不要求Conv3d与Linear天然bitwise一致。沿用现有相关测试的精度基准；若现有测试未定义该算子的容差，先用FP32参考量化Conv3d与Linear各自误差，再固定BF16门槛，记录relative L2、max abs和误差分布；不在失败后临时放宽阈值。
- 新prof中patch路径的海量im2col和对应小GEMM应消失，改为少量大GEMM及必要cast/bias操作。实际具体kernel数由后端决定，不预设整个patch embedding只有一个kernel。
- 运行20步训练，记录相对B0的增量收益、显存、loss/grad、失败/跳步情况。旧flash日志可作历史参照；新的配对基线应在同环境按相同窗口运行一次。

## 4. S2：metadata 一次准备与位置几何缓存

### 4.1 修改位置与数据边界

主要修改Bridge `modelling_qwen3_vl/vision_model.py`：

- `fast_pos_embed_interpolate`，约237行；
- `rot_pos_emb`，约197行；
- `build_packed_seq_params`，约395行；
- `forward`中共享准备、末端split/cat，约340～390行。

如需保留CPU几何，检查 `modelling_qwen3_vl/model.py` 的 `reorganize_inputs`、视觉CP/DP重排及调用vision_model处（约601～660行），再沿实际data batch入口向上追踪；不先在dataset加一个未经消费或未经重排的CPU副本。

特别是 `collapse_thw`、视频展开、图片/帧重排之后的几何才对应实际视觉token。CPU metadata必须执行同样变换，不能直接使用原processor的顺序。

可将纯几何helper拆为新 `vision_metadata.py`，避免vision_model持续膨胀。测试拟新增 `test_qwen38_vision_metadata.py`。拟配置 `vision_metadata_backend=reference|prepared`；缓存容量由独立有界配置控制，允许关闭缓存验证无缓存prepared路径。

### 4.2 先消除同步，再增加缓存

S2a：统一几何对象，尚不增加跨microbatch缓存。

1. 为视觉forward准备CPU不可变几何 `grids=((t,h,w),...)`。最小修改先在入口做一次整表 `.detach().cpu().tolist()`，其后全部使用Python整数；进一步在确认实际重排链后复用batch中保留的CPU metadata。
2. 根据几何一次计算总patch数、每帧长度、prefix sum、Python int maxlen、merged split_sizes、spatial merge重排索引。
3. 每帧是一个attention序列，构造int32 `cu_seqlens`，从0开始，长度为帧数+1，终点为总patch数。最大长度是单帧最大patch数，不是总token数。
4. 将metadata一次传入GPU，跨27层复用同一份packed参数；移除动态repeat_interleave的输出长度推断和GPU `.max().item()`。
5. 用CPU整数控制插值和坐标生成，消除GPU标量驱动的linspace、arange、split、repeat、view和条件分支。保持原插值权重构造精度与cast位置，避免无意改为BF16插值。
6. `max_seqlen_q/kv`从源头提供Python int，检查TE实际路径是否仍产生隐式scalar同步，不能仅修改类型后假定问题全部消失。
7. 末端split后立即cat若确认无额外语义，可直接返回同顺序张量；与输出接口联动的split逻辑保留CPU尺寸。

S2b：增加有界几何缓存。

| 缓存对象 | 可否跨microbatch复用 | 失效/边界 |
| --- | --- | --- |
| H/W坐标、merge排列、插值indices/weights | 可以，与图片内容无关 | 网格、merge、位置表尺寸、构造精度、device/dtype变化 |
| cu_seqlens/长度、maxlen、split_sizes | 可以，按完整有序帧布局 | 任一帧长度/顺序变化；每设备独立 |
| 视觉compact cos/sin | 可以，按完整token位置 | rotary参数、网格顺序、device/dtype变化 |
| 最终learned pos embedding | 第一版不缓存 | 含模型权重，训练/加载权重后可能过期 |
| patch hidden states/视觉输出 | 不缓存 | 图片内容每次不同 |

- 使用容量与显存字节双上限的LRU；候选初始值在读取真实几何分布后确定，不无限缓存每种图像尺寸。
- 优先缓存每种图像几何的固定对象，按本microbatch顺序拼接；完整布局缓存作为有限的第二层，防止组合数膨胀。
- cos/sin直接保留 `[T,36]` FP32；保留reference需要的旧角度路径。避免先repeat到72维再截回36维。
- pos_embed权重trainable时每次用当前weight查表，并保留完整梯度；冻结模式也先使用同一路径，减少第一版失效机制复杂度。
- 缓存为运行时状态，不进state_dict；模型迁移设备、切换精度或相关配置改变时清空。不能驱逐仍被异步拷贝/Graph使用的存储；Graph在S4拥有独立静态buffer。

### 4.3 验收

- 对reference逐项核对token顺序、插值indices/weights/位置embedding、cos/sin、cu_seqlens、maxlen、split_sizes。
- 覆盖一图、两图、非方图、多帧、不同microbatch连续输入、命中/不命中、LRU驱逐、dtype/device迁移、权重加载/更新。纯metadata先用小型CPU测试，ViT输出与梯度用真实GPU定向集成。
- “同总token数，不同帧边界/图像顺序”的样例必须单独测试；该情况下cos/sin或cu_seqlens不能误复用。
- 验证trainable pos_embed更新后输出确实变化，排除错误缓存最终位置embedding。
- 新prof首层前scalar读取应显著少于32次/microbatch；使用保留CPU元数据时，目标是此部分无逐标量D2H，入口fallback则允许一次整表同步。
- 同时记录CPU几何准备、H2D、GPU位置查表/拼接、cache hit/miss与bytes；缓存命中率低时仍应由prepared路径本身减少同步。
- 20步训练比较B2与B1，不把S1已获得收益重复归因到S2。

## 5. S3：Q/K 联合旋转

### 5.1 修改位置

- Bridge `modelling_qwen3_vl/vision_rope.py`：新增Q/K联合接口、自定义autograd和Triton kernel；当前表准备与单输入flash路径保留。
- Bridge `modelling_qwen3_vl/attention.py`：视觉分支约199行，将两个独立调用切换为联合接口；不修改语言raw-angle分支。
- backend枚举及provider/recipe/启动校验：增加 `vision_rope_backend=flash_qk`（拟定）。
- 现有 `test_qwen38_vision_rope.py` 和 `benchmark_vision_rope.py` 增加联合路径，不复制一套无关测试框架。

### 5.2 kernel与反向设计

1. 一个launch接收Q、K各自的pointer、stride、head数和输出pointer，cos/sin共用；grid沿token块与Q/K head域展开，每个program只负责Q或K的一块，避免重复处理同一元素。
2. 使用实际局部head数，兼容当前TP切分，不硬编码16个全局head。视觉head_dim72，half-split36对，非2次幂padding必须mask。
3. 输入BF16、表FP32、寄存器FP32，最后转回输入dtype；保留分离乘加及当前舍入行为。V完全不参与。
4. 第一版仍out-of-place，输出Q/K各自独立，不因共享QKV存储而原地修改。
5. 用一个autograd Function返回 `(q_rot,k_rot)`，backward联合处理dQ/dK，以共轭旋转复用计算；支持只有一个输出需要梯度的情况，不无条件产生无用grad。
6. 在microbatch入口完成cos/sin连续化和不变约束检查，attention处保留实际shape/stride安全检查。kernel支持真实非连续Q/K，不先做昂贵的contiguous拷贝来“换取”联合launch。
7. 先保持当前tile策略，再用真实shape小范围测tile；不同时混入精度放宽或attention算法改变。

### 5.3 验收

- 对当前两次flash调用与reference分别比较Q/K输出、输入梯度；覆盖非连续stride、奇数token、T7/65/1568/2048/4096及真实batch出现的shape。
- 检查QKV alias场景下输入及V未被写入；不同microbatch表不能串用；recompute的none/uniform/block仍正确。
- 算子forward每27层：旋转kernel从54降到27；未缓存表时另有2次trig。带backward时按实际梯度需求统计，不能以冻结训练推断反向launch。
- 当前冻结训练整步rotary从3456降至1728是结构目标；cached geometry可能另减少trig，必须分开归因。kernel数量减半不代表RoPE时间或训练时间减半。
- 20步训练比较B3与B2；若耗时差异与运行波动相当，结论为收益不明确，保留可选开关。

## 6. S4：冻结视觉 forward Graph

### 6.1 范围选择与已有路径限制

第一版仅处理当前frozen vision/projection的forward；语言和optimizer不捕获，trainable vision、recompute、FP8、MTP及CP>1不在第一版支持范围。

既有 `VisionTECudaGraphHelper` 可作为参考，但不能直接打开现有视觉padding分支：它会丢弃PackedSeqParams，无法隔离两图/多帧；sample inputs未完整包含真实RoPE；其maxlen默认值也不是实际ViT输入长度。当前flash与Graph保护只能在新安全路径完成后精准放行，不能整段删除。

实现优先采用Bridge内的局部forward graph wrapper；若复用TE helper，则必须在Bridge层适配静态输入和metadata，遵守其RNG、捕获顺序等要求。无论哪种路线，不改嵌入MCore、不为视觉打开全语言cuda_graph配置。当前平台为HIP，要实测PyTorch CUDAGraph API与目标FA/TE/TP通信的支持情况。

### 6.2 分阶段实现

S4a：最小可捕获性试验。

1. 基于B3（或上一最佳已通过配置），用真实shape在侧stream预热Triton、TE和allocator，再试捕获一个视觉层/层组。
2. 将patch Linear、CPU几何处理、cache lookup、H2D输入准备留在capture外。图内只执行固定输入到固定输出的GPU计算。
3. 核查FA varlen、TE和TP collective是否含capture不允许的标量读取/分配/后端切换；失败时定位具体算子，不通过改全attention来绕过。
4. 若attention暂不能安全capture，先捕获MLP等子模块，attention保持eager；若此范围已无可见host收益，记录结论，不强行扩大。

S4b：精确signature，无padding的冻结层组Graph。

- 初始key至少包含：device、dtype、总patch数、batch布局、完整有序帧seqlens、maxlen、模型/TP布局及capture范围。不同H/W但同帧长度允许共图的前提是cos/sin每次更新且相关代码不把位置当Python常量；第一版可保守使用完整grids作为key。
- 相同总token数而帧边界不同，第一版使用不同graph，不通过修改同一图中的Python maxlen复用。
- 建立graph私有静态hidden、cos/sin、int32 cu_seqlens及输出buffer；capture只引用它们。metadata cache命中后也要copy到图拥有的buffer，不能捕获指向可驱逐缓存的地址。
- 每microbatch在同一依赖链上更新输入，等待copy完成后replay；维护stream/event依赖，避免copy/replay竞态。
- 先捕获单层/少量层，再扩大到27层decoder。patch与merger第一版留eager，后续仅在输出生命周期清晰且有收益时扩大范围。
- 未见过/不适合的signature eager执行；有界收集warmup期常见signature再capture。计时窗口内不临时capture长尾形状，记录fallback。按graph字节预算限制常驻数量，不逐microbatch无限生成图。

S4c：TP与输出生命周期。

1. rank0/rank1参与同一视觉TP通信，必须按一致signature顺序capture和replay；Graph cache命中、fallback及capture决定在TP组内保持一致。不得让一rank capture collective、另一rank eager或运行另一形状。
2. capture同步安排在训练调度允许的位置；不要在单rank的随机cache miss中插入可能阻塞其他PP rank的临时捕获。
3. 第一版将replay输出clone/copy到调用方拥有的新存储，再交给后续eager层或语言模块；该copy成本计入收益。
4. frozen并不代表输出无需保留。语言backward可能保存vision输出；连续microbatch replay不能覆盖旧输出。验证后才评估ring buffer及按PP调度回收，不能仅因当前vision无backward就复用全部输出。
5. Graph中若有随机操作，核实当前路径是否实际使用dropout和所需RNG tracker；沿用TE graph时遵守其tracker配置，不无条件关闭loss/grad检查来适配局部forward图。

S4d：可选的padding/bucket扩展，不作为第一版前提。

只有精确signature图确认有效且cache命中率不足时，才考虑merge前patch长度bucket。设计显式padding序列与真实cu_seqlens，保持帧隔离，分别处理输出裁剪和merger的merge结构；不能只把两图拼成长序列加尾部mask。padding额外算力/显存必须计入。无法正确适配时继续eager fallback。

### 6.3 修改位置与拟配置

- 拟新增Bridge `modelling_qwen3_vl/vision_graph.py`，管理signature、static buffers、capture/replay、输出所有权和统计。
- `vision_model.py`、`transformer_block.py`：仅视觉层组的调用分流和安全保护，不绕过原packed attention语义。
- `qwen35_vl_provider.py`、`transformer_config.py`：声明视觉专用配置并验证组合，不能依赖不存在的Qwen3VL provider字段。
- 需要训练启动/结束生命周期钩子时才改Bridge `training/train.py`；保持graph cleanup和TP协调，避免无必要的全局训练流程改动。
- 拟配置 `vision_graph_mode=none|frozen_forward`、层组大小、signature白名单/数量和显存预算；初始default none。
- 定向测试拟新增 `test_qwen38_vision_graph.py`；多卡capture属于本地实验，自动化功能测试遵守仓库最多2GPU限制，8卡粗测单独记录。

### 6.4 验收与停止条件

- 相同shape的两组不同图片和位置连续replay，输出跟随新输入；与eager比较完整视觉输出及下游语言参数梯度。
- 测同总token数不同帧边界、跨signature、fallback、cache驱逐、连续microbatch后统一backward，证明旧输出未被覆盖。
- 用2卡先检查TP collective捕获/replay，再上8卡PP4；检查无死锁、非法访问或collective顺序变化。
- trace须看见实际graph replay、host dispatch下降；不是每step重新capture。记录hit/miss、capture次数和耗时、图内kernel数、输入/输出copy与显存峰值。
- warmup/capture不纳入稳态计时；尽量在第5步prof前完成目标signature捕获。若无法满足，调整prof/计时窗口并明确记录，不强行使用10～20得到混合结果。
- capture成功但端到端未更快、显存代价过大或命中率不足，保持默认none。收益必须包含输入更新、输出clone和padding等全部开销。

## 7. 统一测试、记录与比较方式

### 7.1 最小实验矩阵

遵循当前用户偏好，不跑三遍、不自动扩展解冻端到端矩阵：

1. B0一次20步，S1一次20步。
2. S1验收后，其日志作为B1；S2一次20步。
3. 同理S3、S4各一次20步，复用上一阶段相邻结果，不每阶段重复baseline三遍。
4. 若阶段间环境、负载或源码发生无关变化，旧baseline不可直接配对。记录不一致，必要时仅补一轮可比baseline，不自动启动多轮统计矩阵。

上述是执行上限规划，并非本轮已启动任务。任何阶段失败先修复相关问题，不带着失败继续叠加。

### 7.2 固定采样

- 20步，prof START5 / END6，rank0和rank4，shapes/stack默认false。
- 比较第10～20步（含两端11样本）的mean ms与samples/s；保留每步CSV，同时给中位数和min/max帮助辨认单次异常，不把单次粗测包装成统计显著。
- 若需要定位新路径，可给 `patch_embed`、`vision_metadata`、`vision_decoder`、`vision_merger` 添加命名record_function/NVTX范围；保留范围命名一致，baseline/候选均使用相同采集设置。记录range本身的prof开销，避免全stack采集影响计时。
- 同时保存allocated/reserved显存与device监控；整步时间看所有PP stage的训练计时，不仅看rank0 kernel sum。
- 外部GPU作业出现则标记contaminated，保留结果和粗测限制，不杀其他作业、不自动无限等待/重跑。

### 7.3 每阶段结果表

| 字段 | 内容 |
| --- | --- |
| 配置/源码 | 实际解析配置、命令、根/Bridge版本和diff/hash、数据/processor/seed |
| 正确性 | 输出/梯度误差、测试结果、精度门槛及未覆盖范围 |
| 结构 | patch kernel数、scalar次数、rotary次数、Graph replay/capture/fallback |
| CPU/GPU | 各视觉阶段CPU窗口、GPU kernel sum、同步与copy耗时；不重复相加 |
| 训练 | 10～20步mean、median、min/max、samples/s，loss/grad与NaN/跳步 |
| 资源 | allocated/reserved峰值、缓存/graph字节、外部干扰 |
| 结论 | 相对上一阶段增量收益、相对B0累计收益、启用或保留可选 |

增量耗时下降率=`1-new/previous`；累计=`1-new/B0`，不要直接相加各阶段百分比。观测变化与波动相当时写“收益未明确”，不预先承诺加速百分比。

## 8. 工作区与交付

继续使用 `/public/home/yugeng/work/Megatron-bridge-das-perf-rope`。已有根仓库和Bridge未提交修改须完整保留；不重新初始化submodule、不覆盖原repo、不重启旧矩阵脚本。新阶段实现前记录当前diff及未跟踪文件快照。

每阶段交付：实现diff、配置开关、定向测试、20步训练日志/prof、步级CSV、简短结论，以及handoff更新。明确列出修改文件/符号与原因，方便定位。后续如需提交，根仓库和Bridge须分别处理；本轮方案整理不创建commit。

环境继续使用原实验的进程级DTK兼容方案：`DTK_ENV=/public/home/yugeng/work/checkpoints/qwen38_rope_gpu_20261111_1236/dtk_compat_env.sh`，并核实实际导入worktree源码。容器路径使用 `/public/home/...`，不要传canonical的 `/public2/...`。

本文保留最初的详细实施方案。S1/S2已完成，实际文件、配置、测试与B0/B1/B2结果见handoff；S3/S4仍为拟实施项。现有语言M-RoPE保持原状。
