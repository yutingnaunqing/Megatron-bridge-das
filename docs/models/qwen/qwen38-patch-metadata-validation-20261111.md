# Patch Linear / metadata 验证方案与记录

最新状态：Linear已按精度优先要求撤回，相关代码/配置/旧审计工具备份于 `/public/home/yugeng/work/checkpoints/qwen38_linear_backup_20261111`。本文Linear结果与命令为历史记录；当前Conv3d+metadata单独验收见handoff第0节，Graph最新方案见 [CUDA/HIP Graph方案](qwen38-vision-cuda-graph-plan.md)。当前审计脚本已不含Linear或bias-followup，重用历史实验须先审阅备份恢复步骤。


精度补充：原108项通过仅覆盖当时的算子/小模型门槛。完整27层TP2新审计发现当前Linear的真实图片输出relative L2约1.062%，不是逐位无损；RoPE融合与metadata所测输出/梯度逐位一致。详见 [最新精度与流水线分析](qwen38-accuracy-and-pipeline-20261111.md)，不以性能结构改善推断任务质量不变。


最终状态（2026-11-11 17:18:18 UTC+8）：B0、B1、B2均完成20步，控制器complete，无待运行组。B0=flash RoPE基线，B1=B0+patch Linear，B2=B1+prepared metadata；累计观测mean下降1.73%，受干扰限制。正确性合计108 passed、4 skipped。组别、结果和限制以 [handoff第1节](../../../HANDOFF_QWEN38_PERF.md) 与 [最终训练记录](qwen38-patch-metadata-training-20261111.md) 为准；下文“等待/待执行”字样为当时的过程记录。


2026-11-11。实现位于性能worktree。定向检查及RoPE回归已完成：107 passed、4 skipped；B0/B1训练已完成，B2控制器当前等候节点资源；最新进度/最终结果见 [训练记录](qwen38-patch-metadata-training-20261111.md)。

## 正确性方案

1. 直接加载本worktree实际patch类和未改的原位置插值/旋转坐标方法，避免CPU导入TE时初始化设备。测试代码只编译可信本地AST，不仿造被测计算。
2. FP32/BF16实际GPU上比较Conv3d与Linear的输出、输入/weight/bias梯度，使用现有量级的FP32 `atol=rtol=1e-4`、BF16 `1e-2`；参数名称及state_dict不变，输入包含真实1152宽度和非连续布局。
3. metadata对照单/多图、非方图、多帧：位置输出、RoPE角度与cos/sin逐项零误差，cu_seqlens/长度/顺序精确一致；检查缓存命中、同总长不同帧边界、LRU驱逐、禁用和权重更新后位置输出变化。
4. 小型真实ViT分别测试Linear-only、metadata-only、二者组合；reference/flash RoPE、冻结/解冻、交换图片顺序及重复几何。输出和输入梯度沿用1e-2逐元素门槛；metadata-only的参数梯度沿用2e-2逐元素门槛。
5. Linear改变GEMM归约路径，直接patch算子的FP32和BF16前反向已通过。初次真实ViT解冻诊断中，下游 `merger.linear_fc2.weight` 73728个元素中有1～2个近零元素未通过2e-2逐元素门槛，整体relative L2为0.443%～0.462%；metadata-only通过。保留初始失败记录，不能宣称bitwise一致。
6. 因此对含Linear的完整BF16 ViT参数梯度额外采用明确的整体误差验收：relative L2≤1%，max abs error / max abs reference≤2%；逐参数将两项写入JUnit。这不是提高原始atol，验证依赖独立patch FP32/BF16导数对照、metadata精确对照与完整ViT误差联合成立。若更长训练/解冻视觉要求更严格，需要另行验证，不能以冻结20步推断长期收敛不变。
7. 回归原RoPE定向测试，运行改动文件的Ruff和Git whitespace检查。只运行相关测试，不跑全套。

## 性能方案

三组顺序各20步，一次，不跑三遍：B0=flash+conv3d+reference metadata；B1=flash+linear+reference metadata；B2=flash+linear+prepared metadata。固定TP2/PP4、GBS64/MBS1、SEQ4096、16语言层+27视觉层、冻结vision/projection、相同数据和LR decay30。

第5步采rank0/rank4 prof，第10～20步比较mean/median/min/max及samples/s，保存源码快照、hash、配置/命令、监控、原始日志与步级CSV。新训练相对相邻组给增量收益，相对B0给累计收益。外部GPU干扰标记contaminated，不杀其他作业，不自动重复矩阵。

结构验收：B1的逐patch im2col/小GEMM应消失；B2的GPU标量提取显著减少，attention帧边界与1728次视觉varlen调用保持。共享cos/sin是否因缓存进一步减少取决于命中，不预设命中率。

实验目录：`/public/home/yugeng/work/checkpoints/qwen38_patch_metadata_20261111`。

## 已完成结果

- 新增检查31项通过，覆盖9项CPU、22项GPU；原RoPE回归76项通过。4项跳过为CPU BF16 convolution（2项，真实GPU已测）和独立interpreter入口（2项）。
- 真实ViT的Linear-only、metadata-only、组合三种配置 × reference/flash × frozen/trainable全部通过最终验收。
- JUnit记录的参数梯度最大relative L2为0.7671%，最大按参考张量peak归一化的abs误差为1.3699%；含多次不同图片排列。metadata-only仍按原逐元素门槛检查。
- 首次最终复测在另一作业启动时发生6项GPU OOM，保留 `correctness_final.log/xml`；资源稳定后重跑成功，见 `correctness_retry.log/xml`。OOM不作为数值失败或代码通过依据。
- Ruff和两级Git diff--check通过。训练尚未开始计时；`state.json` 为实时状态。

## 修改定位

- Bridge `modelling_qwen3_vl/utils.py` / `Qwen3VLVisionPatchEmbed.forward`：可选Linear，保留Conv3d参数；构造与前向校验。
- Bridge `modelling_qwen3_vl/vision_metadata.py`（新增）：完整有序网格的一次CPU读取、坐标/插值/packed参数/cos-sin准备，以及16条/64MiB双限LRU；不缓存模型输出和最终learned位置embedding。
- Bridge `modelling_qwen3_vl/vision_model.py` / `__init__`、`_apply`、`forward`：metadata backend校验、转换设备/精度时清缓存、prepared分支；保留原分支，移除prepared路径末端split/cat。
- Bridge `qwen35_vl_provider.py`、`modelling_qwen3_vl/transformer_config.py`：声明和传递 `vision_patch_embed_backend`、`vision_metadata_backend`、cache_entries/cache_bytes。
- 根HCU `recipes/qwen38_vl/bw1000/qwen38_vl.py`：环境变量选择后端；也可直接CLI覆盖。
- 根 `tests/unit_tests/hcu/models/test_qwen38_patch_metadata.py`（新增）：独立算子/metadata对照与真实ViT集成。
- 实验目录 `run_training_compare.py`（临时）：空闲检查、顺序B0/B1/B2、prof与计时解析、资源监控、源码快照。

开启示例：`model.vision_rope_backend=flash model.vision_patch_embed_backend=linear model.vision_metadata_backend=prepared`。未显式选择时维持conv3d/reference；prepared与视觉Graph仍显式拒绝组合。本轮不实现Q/K联合旋转或Graph。

## 预处理结构 prof（仅计数）

等待完整节点期间，使用实际1152宽度、1568个patch、两张28×28网格图片采集Chrome trace；外部作业占用节点，`timing_valid=false`，只用于核实调用结构，不报告耗时或速度比。

| 路径 | GPU kernel数 | im2col | GPU标量读取 |
| --- | ---: | ---: | ---: |
| patch Conv3d | 3137 | 1568 | 0 |
| patch Linear | 1 | 0 | 0 |
| 原metadata（位置插值+RoPE表+packed参数） | 64 | 0 | 32 |
| prepared metadata，缓存命中 | 6 | 0 | 0 |

prepared的0次scalar不代表完全无同步：仍做一次整表GPU→CPU几何copy。数据入口保留CPU几何是后续可选项。该样例metadata缓存1 miss/1 hit，owned张量752652 bytes（约735KiB）。

工件：实验目录 `structure_profiles/summary.json` 及四份 `.json` trace；脚本 `profile_preprocess_structure.py`。B0首次16:27启动后被外部作业打断，RCCL申请32MiB失败，没有完整step，全部归档 `invalidated/B0_external_1627/`。当前控制器等待连续四次空闲后重试三组训练，尚无有效训练加速结论。

## B1完成及监控恢复

B1日志完整20步、无NaN/跳步，并有after-training时间戳和正常结束footer；监控查询hy-smi超时导致控制器未保存Docker退出码。记录 `exit_code=null` 与completion_evidence，不伪造退出码0；恢复时复用该完成结果，不重跑B1。第10～20步mean17218.973ms、median15019.8ms，含第17步37630ms异常值，完整保留。单次平均数无法排除外部作业干扰。

控制器已修复：查询失败记录monitor_error、计入覆盖缺口，但不终止训练；空闲判定要求成功获得8卡数据且连续四次满足阈值。B0/B1已复用，仅B2待执行。

另外补充精度转换边界：模型运行后将rotary频率buffer转为BF16时，生成系数前显式恢复FP32，raw angle仍保留原dtype。新增CPU边界测试通过，受影响文件复测32 passed、2 skipped；连同此前未受影响的RoPE回归76 passed、2 skipped，合计108 passed、4 skipped。B0/B1的reference metadata不执行此分支；本次正常频率本已FP32，因此B2新增保护是no-op。前后源码manifest均保留。
