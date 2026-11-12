# HCU独立工作树迁移交接（2026-11-12最新）

## 13:04新增一轮H0/H1：Graph计时略快0.48%，仍无稳定加速结论

按用户“节点空闲，重新测试”的要求，在相同工作树、完全相同模型/测试源码上重新顺序跑H0与H1，各20步，不重复三轮或修改模型实现。新结果目录`/public/home/yugeng/work/checkpoints/qwen38_hcu_retest_20261112_124612`。H0=Conv3d+flash RoPE+prepared metadata、Graph none；H1仅增加thread_local冻结patch Graph。两组均正常退出，第5步抓rank0/2/4/6 prof，第10～20步计时。

| 本轮组别 | 第10～20步mean | median | 整轮资源标记 |
| --- | ---: | ---: | --- |
| H0_eager | 15002.127ms | 14998.2ms | contaminated=false |
| H1_thread_local | 14930.464ms | 14923.1ms | contaminated=true（前期外部进程） |

原始mean下降71.664ms，约**0.4777%**。与上一轮H1比已有H0慢0.926%的结果一起看，当前证据只支持“整步收益很小，未证明稳定加速”，继续保持Graph默认关闭。

已专门核对资源污染时间：H1最后一次记录外部进程为12:58:48；其计时窗口从step9结束13:00:54到step20结束13:03:38，两组各12次窗口内监测，均无外部进程、无查询错误。完整记录见`timing_window_resources.json`和各组monitor.jsonl；这是采样证据，不能排除采样间隙的短暂争用，也不抹去H1整轮污染标记。

H1两个视觉rank各capture4次、replay759次、白名单外eager521次、failed_signatures=0，与上一轮相同。两组20步loss及grad norm在日志输出精度下完全一致；这不是所有中间张量、参数或optimizer状态逐位一致的证明。已有固定权重视觉24项逐位一致证据继续适用，因为本轮source_sha256核对全部匹配，未修改模型代码。

本轮控制器已complete，8份prof齐全，原始command/result JSON及源码快照在结果目录。分析脚本`analyze_comparison.py`汇总本轮两个新训练结果，`analysis.json`/`report.md`为生成产物。Graph下GPU kernel事件的可见性说明仍按下节，不用事件数下降推断GPU计算减少。本次新增的执行、分析脚本均在checkpoint目录；代码仓仅更新本handoff文档。

## 最终验收：11:49 HCU迁移与H1训练完成，patch Graph未见整步加速

**本次工作树迁移及对应训练验证已完成**。工作树`Megatron-bridge-das-perf-hcu`，分支`perf/qwen38-hcu-adapter`；未创建提交或推送。官方Bridge和MCore在实际训练后仍干净。默认仍为原Conv3d+flash RoPE+prepared metadata，Graph none；语言M-RoPE、F.linear和Q/K联合旋转未加入本次实现。

| 验证 | 实际结果与证据 |
| --- | --- |
| 当前源码GPU回归 | 108 passed、2 skipped；`qwen38_hcu_graph_fix_20261112/gpu_correctness.xml`，含参数对象/checkpoint键名/strict load/撤销适配及真实模块输出梯度检查 |
| 当前源码完整27层ViT TP2审计 | 24项全部逐位一致，`qwen38_hcu_graph_fix_20261112/accuracy_checks.json`及两rank原始JSON |
| thread_local并发pin诊断 | 64MiB后台分配成功、捕获1次/重放3次、输出逐位一致；global相同场景捕获失败 |
| 双rank多shape组件 | 每rank8次capture/32次replay，所有输出一致 |
| H0已有训练 | 20步，mean10～20=14857.136ms，median14852.9ms；4 rank prof；整轮contaminated=true |
| H1本次续跑 | 11:41～11:49，20步退出码0，无跳步/NaN；mean10～20=14994.664ms，median14984.2ms；4 rank prof；contaminated=false |

H1两个视觉rank各调用1280次、capture4次、replay759次、白名单外eager521次、failed_signatures=0，覆盖约59.3%调用；每rank图数据估算占56,905,728字节（非完整allocator/工作区峰值）。已保留先前global失败和外部作业重占导致的OOM工件，不将失败记录覆盖为成功。

**性能结论：未观察到整步加速**。H1比已有H0原始均值慢约0.926%，只各一轮且H0有污染、运行时段不同，不能据此认定Graph稳定变慢0.926%。用户要求快速判断，不重复三轮或重跑baseline；继续默认关闭Graph，保留显式选项供另一节点测试。

rank0第5步prof中，普通host kernel launch从264328降至132730，另有39次hipGraphLaunch；CPU卷积调用64→25，与39次重放对应。rotary仍3456、视觉FA仍1728、scalar仍2699，语言rank2/4/6主要计数相同。**trace中kernel事件281490→152628、im2col108916→44524不能解读成GPU算术量减少**：Graph重放内kernel在该HIP trace下的可见性与eager不同；原Conv3d算法和表达式并未替换。Graph主要节省host提交开销，当前PP4中视觉阶段收益未转化为可测整步收益；瓶颈归因仍需结合完整时间线，不能仅凭计数下定论。

精度边界：固定权重/输入视觉输出及梯度通过逐位一致；独立20步训练轨迹不逐位相同。H0/H1日志loss最大绝对差0.200668，末步3.364104/3.366511，绝对差0.002407（约0.0715%）；此前相同配置重复训练也出现漂移，不能用这点证明完全排除实现影响，亦不等于下游任务accuracy变化。本轮未做预训练权重质量评测。

最终结果目录：`/public/home/yugeng/work/checkpoints/qwen38_hcu_graph_retry_20261112`。`analysis.json`/`report.md`由实际工件生成，`H1_thread_local_result.json`记录20步及Graph统计，`source_sha256.json`/`source_tested`固定已测源码；H1 prof为`H1_thread_local/torch_profile/rank-{0,2,4,6}.json.gz`。H0在`qwen38_hcu_migration_20261112/H0_hcu_flash_prepared`。续跑控制器state=complete，进程正常结束，当前无本次测试等待队列。后续需新实验时显式新建输出目录，不盲目重启已完成控制器。

本节优先于下方各历史“等待/失败/未完成”状态。代码接入位置仍见下方“代码位置及实际接入方式”；本轮收尾只更新文档与checkpoint目录下分析/验证脚本，未再修改模型计算代码。

## 11:10实测增量：新模式精度与组件通过，训练因OOM中止

10:47～10:49新版本完整27层ViT、TP2审计完成，两rank共24项比较全部逐位一致。10:59～11:00双rank多shape诊断也完成：每rank40次调用、8次capture、32次replay、8次白名单外eager回退，所有输出精确相等、failed_signatures=0。两项监控contaminated=true，仅作正确性证据，不作计时依据。

11:07启动H1_thread_local，11:09完成第1步，loss13.57141、grad norm195.243；随后rank1分配40MiB失败，报GPU1无空闲显存。同期外部8卡作业重新占用约26～38GiB/进程。日志记录thread_local下4次capture、26次replay、0个failed_signatures；本轮未复现原global捕获失效，但训练仅完成1/20步、未到prof第5步，不能宣布完整训练或加速验收。

原始工件保留在`qwen38_hcu_graph_fix_20261112`，其state=failed为真实终态，原等待控制器已结束。已核对当前全部生产/测试源文件与该轮source_sha256完全一致，因此只续跑H1，不重复108项GPU回归、24项精度审计、组件诊断或H0。续跑目录为`/public/home/yugeng/work/checkpoints/qwen38_hcu_graph_retry_20261112`，入口`run_validation.py`，复用依据`resume_evidence.json`；仍要求节点连续4次空闲，20步、prof5、10～20步计时。此项启动/结果以新目录实时state和实际进程为准。

## 10:36增量：thread_local并发诊断及GPU回归通过

新队列已实际完成`pin_probe_thread_local`：捕获期间后台线程成功申请64MiB pinned memory，无后台错误；Graph捕获1次、重放3次，3组不同输入的输出均与原Conv3d逐位一致，无eager回退。这与global诊断的失败构成受控对照，支持采用thread_local处理该并发捕获问题；不等于完整训练已经通过。原始证据`qwen38_hcu_graph_fix_20261112/pin_probe_thread_local.json`。

10:35～10:36当前源码GPU回归完成：**108 passed、2 skipped**，退出码0，资源监控contaminated=false；JUnit与日志在同目录。队列当前等待`full_vit_tp2`，随后是双rank多shape诊断和H1训练。10:36外部8卡进程再次启动，尚无新版本完整ViT、训练或prof结果。旧H0及旧24项精度结果的范围不变，不用它们替代本次后续gate。

## 10:01增量：global捕获与后台pin分配冲突已复现

09:59节点连续空闲后，新验证队列实际完成`pin_probe_global`。在捕获期间让独立线程申请64MiB pinned memory，后台线程报`hipErrorStreamCaptureUnsupported`，随后主线程在原patch的FP32→BF16转换处报`hipErrorStreamCaptureInvalidated`；与此前训练失败位置一致。结果见`qwen38_hcu_graph_fix_20261112/pin_probe_global.json`及同名log。该结果证明这一并发操作能够复现故障，不足以证明此前所有训练失败均由它导致。

下一项`pin_probe_thread_local`尚未运行：10:01外部多卡作业重新占用节点，控制器继续等待。新队列仍未产生GPU回归、完整ViT审计或H1训练结果，不能宣布thread_local修复成功。09:29后原等待进程已退出；核实缺失后恢复为可追踪会话，当前控制器PID2300302（后续以实时state和进程为准），不重复已完成H0。官方Bridge/MCore再次检查无修改。

## 09:17最新状态：HCU迁移H0已完成，Graph线程模式修复待测

**迁移主路径已实际训练通过**：01:00 GPU回归105 passed/2 skipped；修复审计helper旧导入后，08:18～08:20完整27层ViT、TP2、BF16权重审计完成，**24项比较全部逐位一致**。08:21～08:29的H0完成20步、退出码0、4 rank prof齐全；第10～20步mean **14857.136ms**、median **14852.9ms**。当前单节点脚本为16层语言(12 GDN+4 full attention)+完整27层ViT，固定初始化、不加载预训练权重；不将其称为64层全27B预训练质量评测。

H0 rank0 prof：kernel **281490**，其中im2col108916、rotary3456；scalar2699、视觉FlashAttnVarlen1728，与旧实现对应路径计数一致，原Conv3d保留、flash/prepared实际生效。整轮监控contaminated=true，不能仅据14.857s vs旧G0的14.949s宣布迁移加速。

双rank多形状Graph组件诊断完成：每rank40次调用、8次capture、32次replay，FP32/BF16和交错eager形状均精确相等。但08:41～08:44 H1在首个完整训练步前失败，无训练prof；rank0在BF16输入转换时捕获已失效，另一rank记录1680形状成功；外部作业同期重新占用8卡。此轮不再仅根据MIOpen堆栈归因为卷积计算错误或资源争用。

### 正在执行的修复与测试

已确认训练`pin_memory=true,num_workers=2`，而原捕获采用默认global。当前假设是独立pin-memory线程的HIP API干扰global捕获；它尚未被最小复现证实。PyTorch官方说明global会检查其他线程的相关操作，而thread_local只检查当前线程；当前容器2.11源码也支持该参数。参考：[PyTorch graph](https://docs.pytorch.org/docs/stable/generated/torch.cuda.graph.html)。

仅针对冻结、无随机数、无collective、无backward的patch前向图，新增显式capture_error_mode：

- HCU `vision_graph.py`：helper默认global保持可对照，支持global/thread_local、拒绝relaxed；捕获前日志增加shape/dtype/mode，统计增加mode。不改变Conv3d表达式、dtype/bias顺序、输入copy或输出clone。
- HCU `config.py`：新增`vision_graph_capture_error_mode="thread_local"`，Graph mode仍默认none。
- HCU `adapter.py`：把该选项传给helper；审计脚本也使用provider对应选项。
- `test_qwen38_patch_graph.py`：两种模式下分别验证不同输入、输出所有权与eager交错，另测拒绝relaxed。
- 修复后CPU回归 **71 passed、39 skipped**；GPU回归和完整ViT结果必须重新验证新模式，旧24项结果不冒充修复后的通过。

实验目录：`/public/home/yugeng/work/checkpoints/qwen38_hcu_graph_fix_20261112`，原H0/失败H1证据仍在`qwen38_hcu_migration_20261112`。新控制器`run_validation.py`已启动，空闲后顺序执行：global后台pin分配复现 → thread_local同例 → GPU回归 → 原27层TP2精度审计 → 双rank多shape诊断 → **H1_thread_local**20步训练（prof5，10～20计时）。保留H0，不重复跑baseline。mode参数显式进入训练CLI，训练结束必须有thread_local的capture/replay统计才算Graph生效。实际状态以新实验state.json为准；09:15仍有外部8卡作业，等待空闲。

### 精度证据的边界与loss轨迹差异

同权重/输入的视觉输出、输入梯度和参数梯度比较为逐位一致。独立20步训练轨迹不是逐步相同：H0与旧G0首步loss均13.57141，20步日志最大绝对差0.200641，末步3.364104 vs3.392509。已有同为旧源码、相同Conv3d/flash/prepared配置的C1与G0对照也有最大0.103254、末步0.025998差异，说明重复训练本身存在漂移；这不是排除迁移影响的充分证明，也不能将loss漂移称为任务accuracy下降。当前不声称20步训练轨迹逐位一致。

已比较resolved YAML：公共训练参数一致，差异为输出路径、HCU provider类型、正式序列化的原动态HF字段，以及Graph关闭时未使用的图预算/白名单配置。原始差异表在`training_config_diff.json`。后续Graph也须遵循同样精度门槛。HCU源码的新捕获策略与初始迁移快照不同，初始AST等价核查是修复前历史证据，不声称新Graph实现与旧版全文相同。

官方Bridge与MCore在H0训练、审计后仍干净。下方早期“未完成/等待”条目保留历史，以本节与各实验实时state为准。


本工作树：`/public/home/yugeng/work/Megatron-bridge-das-perf-hcu`，根与Bridge分支均为`perf/qwen38-hcu-adapter`。根基准eb51a5e，Bridge基准51885cf13，MCore基准cd4afffa6。原`Megatron-bridge-das-perf-rope`训练源码未改动，保留旧版对照。尚未创建提交。

## 当前迁移状态与执行入口

### 08:08最新实测与恢复

- 修复转换字段后，实际GPU环境dryrun于00:58通过；GPU定向回归于01:00完成，**105 passed、2 skipped**。JUnit明确显示参数对象/checkpoint键名/strict load/适配撤销检查实际运行并通过，两个skip是CPU Triton interpreter检查。GPU回归受外部作业监控标记污染，只用作正确性证据，不作计时依据。
- 完整27层TP2审计于01:04在开始比较前失败：审计脚本`--patch-graph`分支仍引用官方目录下并不存在的`vision_graph`，而实现已迁HCU。已将`examples/qwen/audit_vision_accuracy.py`该导入改为`hcu_megatron.bridge.models.qwen38_vl.vision_graph`。首次日志/源码快照归档到`attempts/02_audit_import_path/`。
- 08:08核实节点空闲，使用`run_validation_resume.py`从完整ViT审计恢复。只修改审计导入路径；生产HCU实现、训练入口及GPU测试源码与已通过结果逐项hash一致，证据`resume_evidence.json`，因此不重复dryrun/GPU回归。新快照在`source_sha256.json`/`source_tested`。
- 08:09外部容器作业重新占用8卡，当前resume控制器仍等待完整ViT审计的连续空闲窗口；已核实并非自身验证进程误报。新增overlay_import_check.json，检查HCU实现及审计脚本不再导入官方目录中不存在的3个性能helper。
- 训练H0/H1仍未完成，不能宣布迁移训练验收或Graph加速。后续以实时state和实际工件为准。


- 已迁移现有视觉flash/cached RoPE、prepared metadata和冻结patch Graph到HCU目录；不包含Linear优化，不改语言M-RoPE。Graph默认none，捕获稳定性仍需实测，不因迁移完成就认为故障已解决。
- 新工作树官方Bridge、MCore均无源码修改；`env/env.sh`已移除复制recipe文件行为，`run_recipe.py`使用进程内注册。CPU测试启动前后检查过官方工作树干净；验证控制器每阶段前后也强制检查。
- 已额外核对5个迁移方法及3个算子模块：归一化docstring缩进后AST全部一致，证据`migration_ast_equivalence.json`。该检查确认没有改写原计算语法，不替代真实GPU精度审计。
- **CPU定向回归70 passed、37 skipped**；37个跳过项含GPU检查，不能算GPU验收。Ruff、Python编译、shell语法检查通过。
- 实验目录：`/public/home/yugeng/work/checkpoints/qwen38_hcu_migration_20261112`。`run_validation.py`控制器已运行，实时状态`state.json`；逐阶段log、command/result JSON、源码快照、资源监控与prof保存在此。`analyze_validation.py`可从实际工件生成`analysis.json`/`report.md`，并比较H0与旧G0的逐步日志loss、汇总prof与Graph统计；未运行阶段明确显示未完成。以实际state及日志为准，未完成的阶段不得宣称通过。
- 顺序：训练配置dryrun → GPU定向测试（含参数对象/checkpoint键名/strict load/适配撤销/设备转换）→ 原27层ViT TP2数值审计 → **H0**20步训练（原Conv3d+flash+prepared、Graph none）→ 双rank四形状Graph诊断 → **H1**20步训练（仅增加patch Graph）。H0/H1均prof第5步(rank0/2/4/6)，计时10～20，不重复三遍。H0可参考旧G0，但两轮资源状态不同，不直接当作公平加速比。
- 无空闲GPU时连续监测，成功读取8卡、无外部>64MiB作业、各卡memory<=5%且use<=1%，连续4次才启动下阶段。查询故障不视为空闲；外部作业不会被终止，监控污染保留。每阶段超时1800s，不自动无限重跑失败作业。
- 精度gate会读取两rank JSON，要求24项逐位一致；Graph训练要求日志确有capture/replay统计，不能以eager回退冒充Graph验收。若任何阶段失败，控制器写failed并停止，需要查看日志、修正后显式续跑。

### 已发现并处理的问题

首次dryrun在配置转换时报`hf_text_config`缺失：固定上游AutoBridge会在provider构造后动态赋此字段及head_dim等，只复制原dataclass字段会丢信息。HCU provider现已正式声明这些转换字段，按目标字段复制，并保留加载hooks/private runtime state；未知公开字段明确拒绝。新增CPU回归模拟这些赋值，验证字段与hook保留；真实GPU环境dryrun仍待再次执行。首次失败归档在`attempts/01_hf_conversion_missing/`。

00:45节点重新被外部8卡作业占用（约44GiB/卡，另有作业约7GiB/卡），当前控制器等待空闲。额外尝试纯CPU dryrun时，环境中的TransformerEngine导入会无条件查询GPU，因此报No CUDA GPUs；该路径不能在完全隐藏GPU时做配置解析，不为绕开它修改依赖。CPU单元回归70项通过不受影响。

## 代码位置及实际接入方式

目录前缀：`hcu_megatron_bridge/src/hcu_megatron/bridge/models/qwen38_vl/`。

| 文件 | 修改内容 |
| --- | --- |
| `config.py` | 新增正式dataclass配置字段与视觉配置浅拷贝，避免复制process group |
| `provider.py` | HCU dense/MoE provider子类，recipe明确选择；provide构建后适配及独立vision spec两条路径均接入 |
| `adapter.py` | 仅对当前vision/decoder/attention实例绑定计算方法；不替换Parameter或插入wrapper子模块；支持detach恢复、缓存清理和Graph退出统计 |
| `compatibility.py` | 对5个官方Qwen源文件做固定SHA检查，不支持的版本明确失败 |
| `vision_methods.py` | 从已验证版本迁移视觉forward与必要辅助函数 |
| `attention_methods.py` | 从已验证版本迁移视觉attention forward，接入HCU fused rotary |
| `block_methods.py` | 迁移cos/sin跨层和recompute透传 |
| `vision_rope.py` | 已有FP32三角表+FlashAttention Triton旋转，不缓存Q/K |
| `vision_metadata.py` | 已有有界几何缓存，当前learned position仍每次读取并参与autograd |
| `vision_graph.py` | 已有冻结原Conv3d Graph、有界缓存、copy/clone与回退；原训练捕获故障仍待复验 |
| HCU `recipes/qwen38_vl/bw1000/qwen38_vl.py` | AutoBridge转换后显式构造HCU provider，保留转换配置 |
| HCU `recipes/__init__.py` | 注册HCU recipe到当前进程，检测重名冲突，不写官方文件 |
| 根 `env/env.sh` | 删除cp覆盖官方recipes行为 |
| 根 `run_recipe.py` | HIP时注册HCU recipe，finally调用HCU清理；不修改官方training/train.py |
| `examples/qwen/audit_vision_accuracy.py` | reference直接用干净官方方法，其他variant启用HCU实例适配，同权重/输入对照 |
| `tests/unit_tests/hcu/models/` | 迁移已有回归；新增test_qwen38_hcu_adapter.py验证迁移专属约束 |

当前实现使用“自定义provider + 实例方法适配”，无需再增加全局monkey patch。它解决原provider.provide直接构造整个模型、单独覆写build_vision_encoder_spec不覆盖实际训练的问题。保持官方原sharded_state_dict方法及模块树，参数与键名是否一致由实际GPU集成测试确认，不能只靠设计判断。

复现：先`source env/env.sh`，单节点仍用`bash examples/qwen/run1_qwen38_vl_single_node.sh train`；默认flash+prepared+Graph none。原路径对照需同时设`VISION_ROPE_BACKEND=reference VISION_METADATA_BACKEND=reference`并追加`model.vision_hcu_enabled=false`。Graph显式参数与旧版相同，不能与上游TE Graph同时开。切换工作树务必重新source环境，避免PYTHONPATH指向旧Bridge。

下面内容为旧性能工作树的历史优化、精度及故障证据，旧路径/旧“尚未迁移”描述不代表本工作树当前状态。

---

# Qwen3.8-27B 训练性能优化交接

更新日期：2026-11-11。本文是当前状态入口；早期记录中“等待 GPU”“训练未开始”“三轮矩阵待执行”等内容属于历史状态，以本文及实验目录 `state.json` 为准。本文包含本机路径，供当前工作区交接使用；若对外提交，应先整理环境相关内容。

## 0. 当前执行配置：Conv3d + flash RoPE + prepared metadata

最新用户决定已落实：精度优先，撤回F.linear训练优化；prepared metadata保持所测数值一致且有准备阶段收益，继续开启；本轮新增可选冻结patch Conv3d Graph原型，默认仍关闭；已完成单元测试及完整ViT精度验证，单节点性能对照见下方实时状态。语言M-RoPE未改。

### 0.1 当前代码与默认入口

- **patch已恢复原Conv3d**，`utils.py` 与Bridge基准版本无diff；provider、transformer config、HCU recipe中的 `vision_patch_embed_backend` / `VISION_PATCH_EMBED_BACKEND` 已移除，旧Linear CLI不能用于当前代码。
- 单节点入口 `examples/qwen/run1_qwen38_vl_single_node.sh` 默认导出 **VISION_ROPE_BACKEND=flash、VISION_METADATA_BACKEND=prepared**。通用provider/recipe仍默认reference，其他入口不被强制更改。Graph配置仍为none。
- `examples/qwen/train_qwen38_vl_pretrain.sh` 新增metadata枚举校验、export，并在MPI分支转发rope/metadata变量；多节点训练本轮未测。
- 关闭metadata可用 `VISION_METADATA_BACKEND=reference bash examples/qwen/run1_qwen38_vl_single_node.sh train`；保留flash RoPE和原Conv3d。需要完整原RoPE再显式设 `VISION_ROPE_BACKEND=reference`。
- 活动测试为 `tests/unit_tests/hcu/models/test_qwen38_metadata.py`；旧Linear测试已归档。活动 `examples/qwen/audit_vision_accuracy.py` 仅检查Conv3d下的RoPE/metadata，原Linear与bias审计版本在备份中。

### 0.2 Linear备份与回退验证

备份目录：`/public/home/yugeng/work/checkpoints/qwen38_linear_backup_20261111`。

- `files/` 为完整相关原文件；`sha256.json` 校验；`root.diff/status` 与 `bridge.diff/status` 记录回退前状态。
- `restore-linear-only.patch` 只恢复Linear训练实现/配置，保留metadata/RoPE；已通过 `patch -p1 --dry-run`，日志 `restore_patch_check.log`。未来复用先审阅README和精度报告，不自动恢复或启用。
- 回退后的定向检查及RoPE回归：**94 passed、2 skipped**。
- 在实际回退代码上重跑完整27层、TP2审计，**16项逐位一致检查全部成立**：真实1844-patch双图前向、多帧反向、逐层与最终输出、输入及已计算参数梯度、重复控制。
- 数值结果仅针对所测权重/输入和配置，不以随机初始化短训代替预训练权重的任务质量评测。

### 0.3 metadata单独验收：C0/C1均完成

与历史含Linear的B组分开：

| 本轮组 | patch | 视觉RoPE | metadata | 第10～20步mean / median ms |
| --- | --- | --- | --- | ---: |
| **C0** | 原Conv3d | flash | reference | 19958.291 / 18002.8 |
| **C1** | 原Conv3d | flash | prepared | 15226.827 / 15017.0 |

各一次20步，prof5、rank0/2/4/6，计时10～20。两组退出码0；控制器已complete，无待运行组。两组计时窗口都监测到外部进程，C0含44001.5ms异常步，全部保留；**原始mean下降23.71%不能解释为metadata独立训练加速**。

保留prepared的直接收益证据是独立准备函数配对计时：三个真实/多帧网格，reference median **2.74～3.25ms**，prepared命中 **0.151～0.189ms**，冷缓存 **0.785～0.831ms**，且位置输出/cos/sin/cu_seqlens逐项相等。只报告准备阶段改善，不由此推断整步百分比。

完整训练rank0中，scalar调用 **11659→2699**，首层前 **33→1**，kernel总数 **288587→281490**；保留108916个im2col、3456个rotary、1728次视觉varlen调用，确认未重新引入Linear或改变帧边界。

实验目录：`/public/home/yugeng/work/checkpoints/qwen38_metadata_conv_20261111`。`summary.json`、`timing_steps.csv`、`report.md`、四rank prof、`metadata_microbenchmark.json`、`accuracy_checks.json`、`adoption.json`及源码manifest齐全。默认切换后dryrun已确认flash/prepared、Graph none；shell语法、Ruff与whitespace检查通过。C0/C1显式使用相同配置字段，切换默认没有改变被测算法。

### 0.4 Graph方案及当前边界

详细方案见 [保留Conv3d后的CUDA/HIP Graph方案](docs/models/qwen/qwen38-vision-cuda-graph-plan.md)：G0统计 → G1冻结patch Conv3d（无TP collective）→ G2视觉MLP → G3带varlen/TP的decoder层组 → 可选扩展。保持原dtype/bias舍入和帧边界，精度目标为eager/replay逐位一致。

C1第5步非NCCL kernel并集：rank0约9.20s、rank2/4约6.11s、rank6约13.59s。最后语言阶段工作更重，视觉Graph可能只有局部收益；先做小范围试点，不能预先承诺全step加速。冻结patch Graph原型已实施（G1）；视觉MLP/decoder层Graph与Q/K联合尚未实施。

本轮结果与代码定位见 [Conv3d+metadata验收记录](docs/models/qwen/qwen38-metadata-conv-20261111.md)。下面B0/B1/B2和Linear精度表是**历史实验**，不能作为当前默认配置或当前可运行CLI。

### 0.5 本轮metadata讲解及Graph实际执行（最新）

**22:37核实更新：G0已完成，G1失败，控制器已停止，不再处于等待空闲。** G0在21:13完成20步，退出码0，第10～20步mean=14948.8ms、median=14954.0ms，rank0/2/4/6的prof齐全；整轮contaminated=true。G1于21:32启动、21:34失败，尚无完整训练步、无prof。rank0先成功捕获1632/1728形状，后续Conv3d捕获中出现`miopenStatusUnknownError`，随后`hipErrorStreamCaptureInvalidated`导致退出；另一视觉rank还记录了1680形状成功。失败时外部8卡作业重新出现，但无直接OOM证据，不能断言资源争用就是根因。独立组件/完整ViT精度通过不等于流水训练内多形状捕获稳定。需排查多形状、stream/异步通信环境及MIOpen捕获兼容性，再只重试G1；保留G0及失败原始证据，不重复跑基线。Graph仍默认关闭，不能报告整步加速。


metadata完整流程、示例与精度边界见 [metadata实现详解](docs/models/qwen/qwen38-metadata-explained.md)。缓存几何/位置，不缓存像素、特征或可训练位置权重；每次仍使用当前pos_embed计算插值。缓存命中仍有一次grid整表D2H，不能声称消除了所有同步。

本轮实验目录：`/public/home/yugeng/work/checkpoints/qwen38_graph_20261111`。

- 原Conv3d HIP capture最小探针通过；不同输入两次replay均逐位一致。
- 定向测试：**101 passed、2 skipped**；含输入更新、输出生命周期、下游可训练Linear梯度、参数更新失效、条目/字节/捕获窗口限制、安全回退及metadata/RoPE回归。
- `full_vit_tp2/` 在原27层ViT、TP2、BF16权重上完成，`accuracy_checks.json` **24项比较全部逐位一致**。真实1844-patch双图Graph每rank捕获1次、replay2次；多帧40-patch可训练场景走eager，不声称验证了Graph backward。当前权重为固定随机初始化，不代表预训练任务质量评测。
- Graph仅捕获冻结patch Conv3d，不改卷积算法表达式、bias舍入、帧边界、TP/PP通信。回放含static_input.copy_与static_output.clone，避免后续microbatch覆盖语言反向需要保留的特征。
- **G0_eager** = Conv3d + flash RoPE + prepared metadata + Graph none。
- **G1_patch_graph** = 同G0，仅增加patch Graph；token白名单 `[1680,1728,1696,1632]`，最多4张图、256MiB估计预算、前192次调用允许捕获，每形状预热3次。未知shape、训练视觉参数、非支持dtype/layout、超预算等均eager回退。无padding，不合并帧。
- 组件配对测试：1680 patches时eager约24.7ms、Graph含copy/clone约21.5ms，中位下降约13%，FP32/BF16输入均精确一致；首次capture+warmup有额外启动成本。详见 [Graph本轮实测](docs/models/qwen/qwen38-patch-graph-20261111.md)。
- 完整训练控制器已启动，20:51检测外部8卡作业每卡约33GiB，处于waiting_resources，训练尚未开始；空闲后自动顺序运行。
- 每组一次20步、prof第5步(rank0/2/4/6)、计时10～20。实际进度以实验目录`state.json`为准；尚未完成时不宣布整步加速。组件计时与prof在`patch_benchmark.json`及`patch_*.json`。

本轮代码定位（均在perf worktree）：

| 文件 | 修改内容 |
| --- | --- |
| Bridge `models/qwen_vl/modelling_qwen3_vl/vision_graph.py` | 新增FrozenPatchGraph，有界捕获、静态buffer、输出clone、权重版本失效、统计与回退 |
| 同目录 `vision_model.py` | 初始化可选patch_graph、原forward接入、设备转换清理、训练结束清理接口 |
| 同目录 `transformer_config.py` | 声明并透传vision_graph_*配置，默认mode=none |
| Bridge `models/qwen_vl/qwen35_vl_provider.py` | Dense/MoE provider声明相同配置字段 |
| Bridge `training/train.py` | 正常训练结束输出Graph统计并清理，无通信流程改动 |
| `examples/qwen/audit_vision_accuracy.py` | 新增--patch-graph，完整ViT eager/replay对照；捕获时跳过不能capture的CPU patch快照hook |
| `tests/unit_tests/hcu/models/test_qwen38_patch_graph.py` | 新增上述定向正确性检查 |

启用示例：在单节点train命令后追加 `model.vision_graph_mode=patch_embed 'model.vision_graph_patch_tokens=[1680,1728,1696,1632]' model.vision_graph_max_entries=4 model.vision_graph_max_bytes=268435456 model.vision_graph_capture_calls=192 model.vision_graph_warmup_calls=3`。不设置这些参数时保持Graph none；现有TE Graph开关仍为none，不能与本原型混用。MCore未改，Linear未恢复。

## 0.6 持续空闲监测与HCU迁移建议（23:03更新）

独立监测已启动：`checkpoints/qwen38_graph_20261111/monitor_node.py`，实时`node_monitor_state.json`，历史`node_monitor.jsonl`和`node_monitor.log`。使用文件锁避免重复实例，每15秒加查询耗时采样；成功读取8卡、无外部>64MiB进程、各卡memory<=5%且use<=1%，连续4次才标记idle_confirmed；监控错误不视为空闲。23:03仍有外部8卡作业，各约44GiB，状态busy。该监测不覆盖原训练失败state，不自动重跑已知失败的G1；准备好修复/诊断后可在确认空闲时启动任务。Graph捕获失败仍未解决。

用户提出将优化放在HCU目录、不直接修改官方Bridge。已核查现有代码，推荐 **HCU provider/config + HCU实现 + 最小运行时适配**，本轮尚未执行迁移：

- 已有`hcu_megatron_bridge/patch_utils.py:MegatronPatchesManager`，由HIP入口`run_recipe.py`加载`megatron_adaptor`触发features注册。优化实现可置于`hcu_megatron_bridge/src/hcu_megatron/bridge/models/qwen38_vl/`，patch清单由features接入。
- RoPE/metadata/Graph helper迁到HCU；HCU dataclass provider正式声明配置字段，不向原dataclass临时塞属性冒充可序列化配置；HCU recipe明确构造/转换为该provider，保留AutoBridge映射产生的配置。
- provider.provide构造完成后、DDP/optimizer包装前，只对该模型的vision/decoder/attention实例绑定HCU方法或受控替换；不替换Parameter，不改变state_dict与分片checkpoint路径。仅wrapper中加一个注册子模块可能引入`.inner.`前缀，须避免。
- 当前`Qwen35VLModelProvider.provide()`直接创建`Qwen3VLModel`，后者固定创建`Qwen3VLVisionModel`，不使用provider.build_vision_encoder_spec来建整个训练模型。仅覆写spec不够，需构建后实例适配或最小factory接缝。完整forward涉及原27层、merger、输出接口和可选recompute，需要一起验证。
- 对不能实例适配的入口才用补丁管理器；限制Qwen型号/开关/Bridge版本与签名，幂等、冲突检测、日志明确失败，不静默吞掉本模型必需补丁的注册错误。尽量不全局替换training.train，而由HCU训练生命周期管理Graph清理/统计。
- `env/env.sh`目前还有cp HCU recipes/__init__.py到Bridge的既有行为。如果目标是运行后官方仓库也保持clean，须改为HCU recipe直接导入/注册，并核查runner加载逻辑；仅迁移性能文件不等于整个官方仓库已零修改。
- 迁移先保存当前源码/diff/测试证据，在干净上游版本上验收HCU依赖闭合；不能直接reset整个Bridge而覆盖Qwen适配或其他既有修改。次序metadata+RoPE、关闭Graph的完整ViT精度回归、参数/checkpoint结构核对，再迁Graph；捕获故障排查与迁移分开记录。
- 长期最佳方案是向上游贡献通用vision factory/metadata/rotary扩展接口，HCU仅保留硬件实现。短期纯runtime patch也可行，但更依赖版本和import顺序，不建议复制整套官方模型目录或运行时修改源文件。

本轮未修改训练模型源码；新增的监测脚本位于实验目录，修改此handoff作交接。后续迁移须维持已证实的metadata/RoPE数值一致性，Graph仍默认none。

## 1. 历史Linear实验：B0/B1/B2定义、结果与精度

最新核对：2026-11-11。patch Linear 和 prepared metadata 已实现；定向检查及原 RoPE 回归合计 **108 passed、4 skipped**。三组单节点训练均已完成20步，控制器于 **17:18:18（UTC+8）** 标记 `complete`。Q/K 联合旋转与视觉 Graph 尚未实现；语言 M-RoPE 暂缓。

### 1.1 B0、B1、B2 分别测什么

这三个名称是本轮**逐项叠加的实验组编号**，不是模型版本或 batch size。三组都启用上一轮已经完成的视觉 flash RoPE；**B0 不是最早的未融合 RoPE 基线**。

| 实验组 / 实际目录名 | `vision_rope_backend` | `vision_patch_embed_backend` | `vision_metadata_backend` | 本组对应的优化 |
| --- | --- | --- | --- | --- |
| **B0** / `B0_flash_conv_reference` | `flash` | `conv3d` | `reference` | 本轮起点：已有共享 cos/sin + fused rotary，patch 和 metadata 保持原路径 |
| **B1** / `B1_flash_linear_reference` | `flash` | `linear` | `reference` | 在 B0 上仅增加 **patch Linear**：将独立 patch Conv3d 等价改为批量 Linear，保留原参数和 state_dict |
| **B2** / `B2_flash_linear_prepared` | `flash` | `linear` | `prepared` | 在 B1 上再增加 **metadata 准备**：一次 CPU 整表几何读取、CPU 整数长度/边界、有界几何与 cos/sin 缓存 |

比较方式：

- **B1 对 B0**：观察 patch Linear 的增量效果。
- **B2 对 B1**：观察在 Linear 已启用时，prepared metadata 的增量效果；本轮 B1 有异常长步，不能直接把均值下降当作 metadata 的独立加速。
- **B2 对 B0**：观察 Linear + metadata 两项组合相对于已有 flash RoPE 的累计效果。
- 最早的 `reference RoPE → flash RoPE` 是另一轮实验，不能把它与 B0 混称，也不能直接相加两轮百分比。

三组统一条件：bw45 / yg_qwen38，8卡，TP2/PP4/CP1，GBS64/MBS1、SEQ4096、BF16；**16层语言 + 完整27层ViT**，冻结vision/projection，随机初始化、真实mantis-birds数据，LR decay30，无checkpoint/eval I/O。每组一次20步，第5步采rank0/rank4 prof，第10～20步（含两端11个样本）计时。结果不能称为完整64语言层27B训练。

### 1.2 三组结果与解读

| 组 | mean ms/step | median ms/step | min～max ms | samples/s |
| --- | ---: | ---: | ---: | ---: |
| B0：flash RoPE 基线 | 15182.500 | 15166.000 | 15123.7～15398.2 | 4.2154 |
| B1：+ patch Linear | 17218.973 | 15019.800 | 14955.4～37630.0 | 3.7168 |
| B2：+ prepared metadata | 14919.336 | 14924.200 | 14839.9～15052.2 | 4.2897 |

B2相对B0，观测到mean耗时下降 **1.73%**、吞吐上升 **1.76%**，median下降约 **1.59%**。这是单轮粗测，三组全程均有外部进程或监控异常标记，不能视为严格隔离的因果估计。B0计时窗口13次、B2计时窗口12次采样未发现外部GPU进程；采样不等于连续无干扰证明。

B1第17步 **37630ms** 异常被完整保留，计时窗口中确有外部进程。因此B1平均数不能证明Linear使训练变慢；同样，自动汇总中的 **B2对B1均值下降13.36%不能解释为metadata独立加速13.36%**。B1的median仅作补充，不用它替换事先约定的mean指标。

| rank0 第5步 prof | B0 | B1 | B2 |
| --- | ---: | ---: | ---: |
| GPU kernel总数 | 288587 | 70755 | 63658 |
| im2col kernel | 108916 | 0 | 0 |
| `_local_scalar_dense`（整步CPU调用） | 11659 | 11659 | 2699 |
| 首个视觉 LayerNormLinear 前的scalar调用 | 33 | 33 | 1 |
| 视觉 rotary kernel | 3456 | 3456 | 3456 |
| 视觉 FlashAttnVarlenFunc | 1728 | 1728 | 1728 |

结构结果确认：Linear消除了海量逐patch im2col/小GEMM；prepared metadata显著减少标量读取，保留视觉attention调用结构。B2仍有一次整表GPU→CPU几何copy，不能宣称所有同步均已消除。kernel减少不等于同百分比的step加速。

### 1.3 工件、已知问题与下一步

- 当前实验目录：`/public/home/yugeng/work/checkpoints/qwen38_patch_metadata_20261111`。`state.json=complete`，`summary.json`、`timing_steps.csv`、`report.md` 为完整结果；三个组目录名即上表实际目录名，各含配置、命令、日志、监控与 `torch_profile/rank-0.json.gz`、`rank-4.json.gz`。
- 当前控制器已结束，无待启动的B2；不要重启已完成三组或恢复旧三轮矩阵。
- B1控制器在训练正常结束后遇到 `hy-smi` 超时，未保留Docker退出码；记录 `exit_code=null`，以20步日志、after-training时间戳及正常结束footer确认完成，不伪造退出码0。B0/B2退出码为0。
- 已修复监控处理：查询失败只记录coverage gap，不终止训练；空闲判定要求成功取得8卡数据并连续四次满足阈值。首次B0外部争用导致的未完成尝试保存在 `invalidated/B0_external_1627/`，不纳入结果。
- B2前补充rotary buffer转低精度后仍生成FP32系数的保护；在本轮默认FP32频率下为no-op，B0/B1使用reference metadata不执行该分支。前后源码manifest保留。
- 代码、精度验收和问题记录见 [本轮验证记录](docs/models/qwen/qwen38-patch-metadata-validation-20261111.md)；三组结果见 [训练记录](docs/models/qwen/qwen38-patch-metadata-training-20261111.md)。
- 当前下一步以第0节为准：Linear撤回，metadata用C0/C1单独验收；Graph只制定方案，不改语言M-RoPE。新优化的整步收益仍需实测，不能由kernel数直接外推。

### 1.4 最新精度复核：不要把小模型通过当作完整ViT无损

2026-11-11新增完整27层、真实1152宽度、TP2、BF16的固定权重/输入审计。**RoPE融合与metadata在所测完整模型输出及梯度中逐位一致；当前Linear不是数值无损。**

- 1844个真实图片patch：Linear与Conv3d的patch输出relative L2约0.2824%；完整ViT+merger输出约 **1.0619%**，max abs **0.0087890625**。
- 完整27层、40-token多帧反向：输入梯度relative L2约 **1.4778%**，TP本地参数梯度整体 **1.0753%～1.0997%**，单参数最大约 **1.6443%**。这些结果高于原2层小模型，不能外推原108项通过为完整深度精度无损。
- bias融合改变BF16舍入位置是重要来源；仅在审计脚本实验的分离bias方案将真实patch误差降到0.002823%，但完整ViT仍约0.9910%，没有写入训练实现。
- Linear在patch单算子上更接近FP32算术参考，不意味着与原BF16模型一致或任务质量更好。以上是数值误差，不是任务accuracy下降比例；未加载预训练权重做任务评测。
- 精度保真优先已按最新要求恢复Conv3d；Linear已从训练路径移除并归档，不再是活动opt-in。metadata按第0节单独性能验收。
- 配置中 `batch_p2p_sync=true`、`overlap_p2p_comm=false`；B1 rank0 profile有70次device sync（CPU duration约3.88s），rank4有134次（约11.85s）。局部kernel减少不能直接变成全step收益，具体最慢stage尚需完整rank证据。

详细方法、指标、限制与复现见 [精度与流水线分析](docs/models/qwen/qwen38-accuracy-and-pipeline-20261111.md)。新增工具为 `examples/qwen/audit_vision_accuracy.py`；原始结果在 `/public/home/yugeng/work/checkpoints/qwen38_accuracy_20261111`。审计完成不代表Linear通过了更严格的一致性门槛。

上一轮仅融合RoPE的结果另见第5节：reference 15315.091ms → flash 14949.500ms，观测耗时下降2.387%；该轮也有外部干扰，与本轮B0/B1/B2分开解读。

## 2. 目录、Git 与代码归属

| 内容 | 位置或版本 |
| --- | --- |
| 原项目，保留现状 | `/public/home/yugeng/work/Megatron-bridge-das` |
| 后续性能工作目录 | `/public/home/yugeng/work/Megatron-bridge-das-perf-rope` |
| 根仓库分支/基准 | `perf/qwen38-vision-rope` / `eb51a5e6f354138734bb8400416f06d1104b76d4` |
| Bridge 子仓库 | `3rdparty/Megatron-Bridge`，独立 worktree，同名分支 |
| Bridge 基准 | `51885cf132b2814188b6855c25a8588254274c2a` |
| MCore | `3rdparty/Megatron-Bridge/3rdparty/Megatron-LM`，本地共享对象检出，detached HEAD |
| MCore 基准 | `cd4afffa648426a959dc7cb1e24b5ce7d0c3ff54`，未修改源码 |
| 运行环境 | `bw45` / Docker `yg_qwen38`，8 × BW1000 64G，PyTorch 2.11.0 / HIP 7.2.26365 |

**目前没有提交 commit。** 根仓库和 Bridge 的修改/新增文件须分别检查和保存。`git diff` 不显示未跟踪文件内容；备份/交接不能只保存 diff。原目录已有的 Bridge `recipes/__init__.py` HCU 注册改动被继承，它不是本次 RoPE 新功能；原目录已有两份未跟踪性能报告也保留。

```bash
cd /public/home/yugeng/work/Megatron-bridge-das-perf-rope
git status --short
git -C 3rdparty/Megatron-Bridge status --short
git -C 3rdparty/Megatron-Bridge/3rdparty/Megatron-LM status --short
```

遵守 Bridge 的 `AGENTS.md`：先读实际工件，再读相关 `skills/*/SKILL.md`；不改嵌入的 MCore，不跑全套测试，不引入必需依赖。本次分析使用 `nemo-mbridge-perf-moe-vlm-training` 中真实多模态负载/视觉热点原则，以及 `nemo-mbridge-perf-cuda-graphs`；当前模型是 Dense，不照搬其中的 MoE 路由建议。用户要求：每次改代码均明确指出修改文件和位置。

## 3. 项目与实际模型

项目基于 Megatron Bridge + Megatron-Core，HCU 层提供适配、recipe 和启动环境；`examples/qwen` 包含 Qwen3.8-27B 的训练入口。模型本地名为 Qwen3.8-27B，但本地 HF 配置实际为 `model_type=qwen3_5`、`Qwen3_5ForConditionalGeneration`，Bridge 使用 `Qwen35VLModelProvider`，视觉代码复用 `modelling_qwen3_vl`。

- 模型：`/public/home/yugeng/work/models/Qwen3.8-27B/config.json`。
- 数据：`/public/home/yugeng/work/datasets/mantis-birds-energon-v1`，真实图片，常见每样本两张图；不要以无图 mock 数据替代性能结论。
- 完整语言模型：64 层，每组 3 个 Gated DeltaNet（GDN）线性 attention + 1 个 full attention；hidden 5120，FFN 17408；full attention 为 24 Q heads、4 KV heads、head dim 256。
- 当前单节点性能 recipe：截取前 16 个语言层，即 12 GDN + 4 full attention；MTP 关闭。语言 M-RoPE 仅涉及 full-attention 路径，不能把 16 层都算作旋转层。
- ViT：27 层，hidden 1152、16 heads、head dim **72**、FFN 4304，patch `(2,16,16)`，spatial merge 2，输出 hidden 5120，无 DeepStack。
- TP2 / PP4 / CP1 / DP1，GBS64 / MBS1 / SEQ4096；每 step 64 microbatches。rank0 与 TP 同组的 rank1 承担视觉；rank4 无视觉。
- 语言 sequence parallel 开启，视觉 sequence/context parallel 不启用；当前没有 recompute/offload，视觉与 projection 冻结。优化过的反向已在算子和小型真实 ViT 单测验证，但**没有完成解冻视觉的端到端训练性能对照**。
- 入口 `examples/qwen/run1_qwen38_vl_single_node.sh` → `train_qwen38_vl_pretrain.sh` → `run_recipe.py`；HCU recipe 位于 `hcu_megatron_bridge/src/hcu_megatron/bridge/recipes/qwen38_vl/bw1000/qwen38_vl.py`。
- 为测性能 fresh init、不载入预训练参数、不进行 checkpoint/eval I/O；图片处理的像素范围、seed、数据顺序、LR、TP/PP 等必须随 A/B 固定。当前粗测 flash 虽止于 20 步，仍显式设置 `scheduler.lr_decay_iters=30` 以匹配 reference。

## 4. RoPE / M-RoPE 与当前实现

### 4.1 原理与本模型的区别

RoPE 将一对通道按位置相关角度旋转：`y1=x1*cosθ-x2*sinθ; y2=x2*cosθ+x1*sinθ`。同一 microbatch、同一 token 的位置跨层不变，Q/K 数值每层变化，因此可以跨层共享 cos/sin，不能共享旋转后的 Q/K。

**视觉 2D RoPE**：按图像 H/W 坐标产生角度，按照 spatial-merge 后的真实 patch 顺序排列。72 个 head 通道全部参与旋转，半表为 36 维，H/W 各 18；输入可为 `[T,H,D]` 或 `[T,1,H,D]`。旋转是前后半段配对。多图/多帧 flatten 后，表按 token 直接对齐，旋转时不再根据 `cu_seqlens` 重置索引；attention 仍使用分帧边界。

**语言 M-RoPE**：文本 token 的 T/H/W 位置相同，多模态 token 用时间/高度/宽度坐标。这里每个 head 为 256 维，仅前 64 维旋转，后 192 维原样透传；32 个频率对按 `[11,11,10]` 分配 T/H/W，theta 为 1e7。`apply_interleaved_mrope()` 交错的是频率来源，**不等于 kernel 相邻通道配对的 `INTERLEAVED=True`**；本模型仍使用 half-split 配对。

`rope.py:143` 组织交错频率；`rope.py:161` 附近的 `forward()` 按 position_ids 生成 angle；`rope.py:455` 的 `apply_rotary_pos_emb_absolute()` 直接走 MCore 普通实现，绕过标准 fused dispatcher。其频率是逐 token 的 absolute 表，含 batch/多模态位置，不能把它当普通相对序列频率表。现有 `apply_rope_fusion` 会被禁止，不应强行打开。

### 4.2 已实现视觉三档

| backend | 准备 | 每层旋转 | 状态 |
| --- | --- | --- | --- |
| `reference` | 原始角度 | 原 PyTorch 路径，Q/K 分别算三角函数 | 默认、回退基线 |
| `cached` | microbatch 内生成一次 FP32 cos/sin | PyTorch FP32 half-split | 正确性及隔离表复用贡献 |
| `flash` | 同上 | Q/K 各一次 FA2 Triton kernel | 当前已测试路径 |

关键约束：保留 FP32 旋转计算、BF16 输入输出；out-of-place，支持非连续 Q/K；只在 microbatch 生命周期内持有表，checkpoint/recompute 显式传表；不对输入就地写。自定义 autograd backward 采用共轭旋转。底层 kernel 使用 mask 支持 head_dim72，`enable_fp_fusion=False` 保留参考的分离 FP32 乘加。

容器中 FA 高层 wrapper 要求 Q/K 与 cos/sin 同 dtype，不适合 BF16 Q/K + FP32 表；当前直接复用已有 `rotary_kernel`，检查所需 ABI，失败时报错，**没有静默退回 reference**。这是对目标 FA2 内部接口的依赖，升级 FA 后应重新验收。

### 4.3 已改代码定位

以下路径相对工作树，Bridge 路径从 `3rdparty/Megatron-Bridge/` 起：

| 文件 / 搜索符号 | 修改内容 |
| --- | --- |
| Bridge `src/megatron/bridge/models/qwen_vl/modelling_qwen3_vl/vision_rope.py:56` / `prepare_vision_rotary_cos_sin` | 新增共享表；`:74` 底层 kernel 调用；`:118` 接口及 cached/flash；自定义前反向 |
| 同目录 `vision_model.py:136`、`:364` 附近 | backend 验证、Graph 组合保护；microbatch 表生成并传给 decoder |
| 同目录 `attention.py:114`、`:199` 附近 | 接收共享表、Q/K 分别融合；语言 raw-angle 路径保持原行为 |
| 同目录 `transformer_block.py:102` 附近 / `_checkpointed_forward` | uniform/block 与普通/TE checkpoint 显式传递 cos/sin |
| 同目录 `transformer_config.py:52`、`:97` 附近 | 视觉 backend 字段与 provider 配置传递 |
| Bridge `src/megatron/bridge/models/qwen_vl/qwen35_vl_provider.py:190`、`:380` | Dense/MoE provider 声明 `vision_rope_backend`，默认 reference |
| 根仓库 HCU recipe / `VISION_ROPE_BACKEND` | 读取 backend 环境变量，可由 CLI 覆盖 |
| 根仓库 `examples/qwen/train_qwen38_vl_pretrain.sh` / `VISION_ROPE_BACKEND` | 导出并验证枚举 |
| 根仓库 `tests/unit_tests/hcu/models/test_qwen38_vision_rope.py` | 输出/梯度、传参、checkpoint、真实 GPU/ViT 验证 |
| 根仓库 `examples/qwen/benchmark_vision_rope.py` | 27 层旋转算子计时、正确性和 Chrome traces |
| 根仓库 `examples/qwen/run_qwen38_rope_ab.sh` | 原冻结/解冻 × backend × 重复矩阵，**当前不执行** |

行号会随后续编辑变化，优先按符号搜索。新增未跟踪文件也属于必须交接的实现。

### 4.4 本轮 patch Linear / metadata 修改定位

Bridge文件相对 `3rdparty/Megatron-Bridge/src/megatron/bridge/models/qwen_vl/`：

| 文件 / 符号 | 内容 |
| --- | --- |
| `modelling_qwen3_vl/utils.py` / `Qwen3VLVisionPatchEmbed.forward` | `conv3d|linear`选择、等价前提校验；保留原weight/bias和state_dict |
| `modelling_qwen3_vl/vision_metadata.py` / `PreparedVisionMetadata`、`VisionMetadataCache` | 一次CPU几何读取、空间merge顺序、插值索引/权重、cu_seqlens/int maxlen、compact RoPE与有界LRU |
| `modelling_qwen3_vl/vision_model.py` / `__init__`、`_apply`、`forward` | 选择prepared分支、设备/精度转换时清缓存、每次读取当前learned embedding、保留帧边界并移除prepared路径末端split/cat |
| `modelling_qwen3_vl/transformer_config.py`、`qwen35_vl_provider.py` | 声明并传递patch/metadata backend及cache entries/bytes |
| 根HCU recipe `hcu_megatron_bridge/src/hcu_megatron/bridge/recipes/qwen38_vl/bw1000/qwen38_vl.py` | 环境选择；实验实际使用显式CLI字段 |
| 根 `tests/unit_tests/hcu/models/test_qwen38_patch_metadata.py` | FP32/BF16输出与梯度、位置精确对照、缓存、真实ViT、精度转换边界 |

B0默认保留原算法；B1显式加 `model.vision_patch_embed_backend=linear`；B2再加 `model.vision_metadata_backend=prepared`。三组均显式 `model.vision_rope_backend=flash`。配置选择与精度门槛详见验证记录。

## 5. 上一轮视觉 RoPE 验证与测量（本轮B0/B1/B2见第1节）

### 5.1 正确性

- CPU 常规：58 passed；独立 FP32 Triton interpreter：2 passed。
- 真实 GPU：18 passed，包含 6 项实际 rotary 前反向、12 项小型真实 ViT（cached/flash × none/uniform/block recompute × frozen/trainable）。覆盖 BF16/FP32、T7/65/1568、非连续 view、多图、非方图、多帧、merger 输出与输入/参数梯度。
- 真实 GPU 算子 benchmark 的已测试随机输入输出和梯度 max abs / relative L2 为 0；这不保证所有输入或不同编译器上 bitwise 一致。
- BF16/FP32 × forward/backward 的 4 组 `hcu:gfx936:64` 离线编译通过，产生 HSACO；BF16 IR 检查过 FP32 分离乘加与 RNE 转换。
- 针对改动文件 Ruff、shell `bash -n`、`git diff --check` 通过。未运行全套测试；未修改 MCore；未新增必需训练依赖。
- 未完成：完整 64 层训练、解冻视觉端到端性能、FP8、MTP、CP>1、视觉 Graph 组合。不可把单测覆盖扩大宣称为这些场景均已验证。

### 5.2 算子与训练结果

隔离的算子测量：每份 workload 包括 27 层 Q/K 旋转及共享表准备，30 次 event 计时样本，真实 BF16 输入。GPU event 覆盖 host 供给间隙，不能当作纯 kernel duration。

| 模式 / T | reference ms | cached ms | flash ms |
| --- | ---: | ---: | ---: |
| forward / 1568 | 5.654 | 4.033 | 2.968 |
| forward / 2048 | 6.700 | 4.477 | 2.949 |
| forward / 4096 | 10.915 | 8.399 | 2.943 |
| forward+backward / 1568 | 12.842 | 12.559 | 9.076 |
| forward+backward / 2048 | 15.601 | 12.512 | 8.842 |
| forward+backward / 4096 | 21.108 | 16.889 | 9.077 |

算子 forward kernel 数：648 → 488 → 56；forward+backward：1134 → 1028 → 110。flash forward 为 54 rotary + 2 trig；带 backward 为 108 rotary + 2 trig。T1568 单独 prof 的 GPU kernel sum 约为 forward 5.548 / 3.726 / 0.702 ms，forward+backward 9.718 / 7.577 / 1.420 ms；与 event 计时来自不同采样，不能相减后断言精确 CPU 开销。

训练最终只完成用户指定的单轮粗测，不再补跑三次：

| 同一第 10～20 步窗口（含两端，11 个样本） | reference | flash |
| --- | ---: | ---: |
| 平均 step ms | 15315.091 | 14949.500 |
| samples/s，GBS64 | 4.1789 | 4.2811 |
| 实际总步数 | 30（已在运行，保留完成） | 20 |

flash 第 5 步抓 prof（START5 / END6，END 排他），计时不含 profiler/export 步。两次训练有外部作业干扰，`contaminated=true`；不删记录、不混入“干净多轮平均”。flash 20 步退出码 0，无 NaN/跳步，loss/grad 有限；独立训练 loss 接近不是严格数值一致性依据。

旧 reference rank0 prof：326475 kernels，视觉旋转相关 41472 kernels、GPU sum 381.559 ms。新 flash rank0：288587 kernels，其中 3456 rotary、GPU sum48.571 ms。总数减少37888，恰为 `41472 - 3456 - 128`（64 microbatch × 2 次共享 trig）。旧/新 trace 运行条件不同，**结构减少可信，duration 差值不能直接当严格加速量**。rank4 仍为28143 kernels，没有视觉 rotary。

## 6. 新 rank0 trace：“标量之后到首个 LayerNormLinear”究竟是什么

本节重新读取用户给出的新文件，非沿用旧 trace 推断。新 prof 关闭 Python stack 与 shapes，因此下面用 CPU op 嵌套、External id、runtime 和 GPU kernel 对齐，再与实际源码核对；不声称已取得精确 Python 调用栈。

时间原点为 `ProfilerStep#5` 开始，单位 ms：

| 事件 | 时间/耗时 | 证据与含义 |
| --- | --- | --- |
| `aten::_local_scalar_dense` | start 9.072，dur 0.062 | CPU 读取 GPU 标量，runtime 可见 `hipMemcpyWithStream`；不是一个要单独“融合”的 GPU 算子 |
| `aten::miopen_convolution` | start 9.608，CPU dur25.770 | External id99；上层是 `conv3d/convolution/_convolution` |
| 该 convolution 的 GPU 工作 | start9.713 → end35.786，3400 kernels，duration sum24.802 | 与同一 External id99 关联，主要为1700 im2col +1700 小 GEMM |
| 视觉位置/metadata 准备 | convolution CPU 返回后 → norm，6.727 | 位置插值、旋转坐标、cu_seqlens/maxlen、共享 cos/sin；首个 norm 前最后共享 cos/sin 约41.831/41.848 |
| 首个视觉 `_LayerNormLinear` | CPU start42.106；相关 GPU kernel 起约42.184 | 进入第一层视觉 Transformer |

所以该标量结束到首个 norm 约 **32.972 ms**；其中很大一段不是 GPU 空闲，而是在执行 patch embedding 的大量小 kernel。CPU 调用返回与 GPU 完成可重叠，表中时间不可直接相加得另一套精确分解。

整 step 进一步统计：

- `im2col_fwd_3d_fp16` **108916 次 / 736.512 ms**；配套同名小 GEMM family **108916 次 / 853.228 ms**。合计217832次，约占rank0全部kernel数量 **75.5%**，GPU duration sum约 **1589.740 ms**。内核名带 fp16 不等于本次模型已改为 FP16；训练仍为 BF16。
- 64次 `miopen_convolution` 的 CPU duration 合计1324.954 ms；每次返回到随后第一个 norm 的 CPU窗口合计637.861 ms。
- 这64个后处理窗口中，共2048次 `_local_scalar_dense`，**每 microbatch 32次**，标量调用 host duration合计337.875 ms。含排队/等待，不能与GPU sum相加，也不能把637.861或337.875 ms全部视为必得收益。
- 特别注意，最前面的少数标量读取可能在等待先前 convolution 的尾部；优化读取动作不会自动消掉上游计算。

源码解释：

1. `utils.py:33` 的 `Qwen3VLVisionPatchEmbed`：每个输入已是一个独立 `(C=3,T=2,H=16,W=16)` patch，却走3D convolution，当前 MIOpen 后端呈现逐 patch im2col+小 GEMM 的低效分解。这部分是确定的主要小 kernel 来源。
2. `vision_model.py:237` 的 `fast_pos_embed_interpolate`：遍历 GPU `grid_thw` 标量，把 h/w/t 用作 linspace、split、repeat、view 等 CPU 参数，触发隐式标量提取；索引/权重主要经CPU tensor与Python列表构造后再传GPU。不能笼统地称其全部为GPU插值 kernel。
3. `vision_model.py:197` 的 `rot_pos_emb`：`.max().item()`、`.prod().sum().item()`，以及GPU标量控制 arange/expand/repeat/if，制造同步和小操作。
4. `vision_model.py:395` 的 `build_packed_seq_params`：动态 `repeat_interleave`、cumsum、pad、max。trace看到repeat_interleave内部读取输出长度和检查条件；可在已知CPU几何时避免动态推断。
5. `max_seqlen_q=seqlens.max()` 当前返回GPU标量而不是Python int。已安装TE也含maxlen推导/标量条件路径；完整视觉层内是否每次都发生哪一种标量转换，需额外局部标记确认，不能把当前“首层前32次”全归给TE。

分析证据已保存到实验根目录：`analyze_preamble.py`、`preamble_analysis.log`、`preamble_analysis.json`，以及首个窗口原始事件子集 `preamble_events_compact.json`。脚本只读gzip trace，不使用GPU。

## 7. 分阶段方案与实施边界

P0历史实现已撤回并备份；P1已通过Conv3d下的C0/C1验收并在单节点入口默认开启，当前状态以第0节为准。下列保留原理与历史方案；P2、P4尚未实施，P3语言M-RoPE暂缓。

### P0：patch Linear（历史实现，已撤回并备份）

**为什么优先**：当前GPU sum约1.590 s，比剩余视觉 rotary 48.6 ms大约33倍，并占约75.5%的kernel数量。GPU sum不等于step关键路径收益；仍须实测。

**修改入口**：Bridge `modelling_qwen3_vl/utils.py` 的 `Qwen3VLVisionPatchEmbed.forward`。保留 `self.proj` 的 Conv3d参数与state_dict名字，只在计算时做：

```python
# 历史Linear计算原理；当前活动代码不再包含该路径
x = hidden_states.reshape(-1, self.in_channels * self.temporal_patch_size * self.patch_size**2)
w = self.proj.weight.reshape(self.embed_dim, -1)
out = F.linear(x.to(dtype=w.dtype), w, self.proj.bias)
```

输入的实际flatten顺序必须与原 `view(N,C,T,H,W)` 逐项一致；当前kernel=stride=patch、无padding、每patch输出1个位置，数学上等价。不要重新注册另一份Linear参数，避免权重转换/optimizer/state_dict问题。新增 `vision_patch_embed_backend=conv3d|linear`，默认先保留原路径；非本结构明确拒绝或使用conv路径。autocast和dtype跟随原行为，不预先缓存可训练权重的拷贝。

**验证**：同一权重/输入比较真实 BF16/FP32 输出；解冻时比较输入、weight、bias梯度，检查state_dict装载；测试非连续输入和实际token规模。矩阵计算归约顺序不同，BF16不能预设bitwise一致，用事先确定的误差与下游完整ViT对照验收。随后跑真实数据20步，prof5、比较10～20。trace中海量im2col及小GEMM应消失并被少量大GEMM取代；检查显存、loss和吞吐。通过后再考虑加入默认路径。

### P1：metadata 准备与缓存（C1验收，单节点默认开启）

当前通过vision入口一次整表 `.cpu().tolist()` 取得真实重排后的几何，以完整有序网格为key缓存几何张量，默认16条/64MiB双上限。尚未增加dataloader CPU几何透传或每图几何的两层缓存；不缓存最终learned位置embedding。实现入口为新增 `vision_metadata.py` 和 `vision_model.py` 的prepared分支。

**修改入口**：`vision_model.py` 的 `fast_pos_embed_interpolate`、`rot_pos_emb`、`build_packed_seq_params`、`forward`；必要时在数据batch构造保留CPU `grid_thw`，不改数据语义。

实施拆分：

1. 尽量从dataloader/processor保留CPU整数几何，在进入GPU前生成不可变 `((t,h,w),...)`；短期fallback每microbatch仅做一次 `grid_thw.detach().cpu().tolist()`，之后不再逐标量读取GPU。一次copy仍可能等待GPU，不是完全无同步。
2. CPU一次算总patch数、每帧长度、cu_seqlens、Python int `max_seqlen_q/kv`、split_sizes、spatial-merge token排列。把int32 cu_seqlens一次传GPU并跨27层复用；保留每帧边界。
3. 将二维插值的4组indices、weights与merge重排索引按 `(h,w,merge_size,num_grid_per_side,device,dtype)` 缓存；时间重复只由t控制。使用有界LRU，控制不同分辨率导致的CPU/GPU缓存增长。
4. 若视觉pos_embed权重可训练，只缓存indices/weights，**每次仍要查当前embedding权重并保留梯度**。冻结状态下可另行缓存最终位置embedding，但须有权重更新/加载/设备迁移时失效机制；不要把frozen专用缓存默认用于可训练路径。
5. 视觉RoPE可直接缓存几何对应的compact cos/sin或坐标/基础频率，避免构造重复的72维角度后再切回36维；表对齐真正token顺序，并明确缓存上限和dtype/device失效条件。先去标量同步，再增加缓存，便于定位收益。
6. 删除末端无必要的split后立刻cat，前提是确认输出顺序和接口未依赖split；使用已保存split_sizes避免又一次 `.tolist()`。

**验证**：单/多图、非方形、多帧、不同网格连续microbatch和缓存命中/失效，检查位置embedding、cos/sin、cu_seqlens逐项与基线对齐；比较冻结/解冻ViT输出和梯度。trace检查首层前32次标量读取显著减少、动态repeat_interleave不再读GPU长度、maxlen跨层是int且不重复提取。保留packed attention的独立图片/帧语义；不要用“maxlen=T总长”改变attention划分。

### P2：继续优化视觉 RoPE

可以继续，但优先级低于P0/P1。当前3456次rotary =64×27×Q/K两次，GPU sum48.571ms，相对约14.95s的训练step只是约0.3%的量级；该比例不是host开销上限。继续优化应同时测host调度与实际step，不宜只追求更低kernel duration。

1. **Q/K合成一次launch**：新增可选 `flash_qk` adapter/custom Triton kernel，一次dispatch传入Q/K各自pointer、stride、head数，复用同份cos/sin；V不变。预期1728个旋转kernel，不能声称完整step加速一倍。先保持out-of-place，反向联合处理dQ/dK。测试各自非连续stride、dtype/device、梯度和多microbatch生命周期；不要因QKV共享底层存储就安全地原地旋转。
2. **减少Python包装/调度**：把跨层不变的验证和contiguous准备放到入口，保留逐调用安全所需检查；联合Q/K一次autograd入口，减少view/context/dispatch次数。Triton JIT已缓存并不意味着每次launch零CPU成本。用CPU时间/CPU-GPU间隙证明收益。
3. **按真实72维调kernel tile**：对实际T、heads、stride测BLOCK_M、warp与寄存器/带宽；当前非2次幂维度有padding/mask。所有候选保留FP32中间计算、RNE输出和正确mask；不要未测就打开FMA或改BF16系数。AITER为另一候选，先验证目标gfx936构建、72维mask、layout和前反向，不新增必需依赖。
4. **与位置准备或相邻算子融合**：消掉重复角度/repeat/cos/sin中间量；若进一步融合QKV epilogue/RoPE或attention读入旋转，需重新处理TP布局、TE接口和反向，工作量较高，当前不作为首选。
5. **Graph降低host调度成本**：见P4。Graph复用和kernel融合可以叠加，但收益不能简单相加。

### P3：语言 M-RoPE 的独立适配

可做。建议单独 `language_rope_backend=reference|cached|fused`，不借用视觉backend开关。当前 `attention.py` 只要收到cos/sin就进入视觉专用分支，直接从语言入口传表会误路由，必须显式区分或设计共享的partial-rotary接口。

实施步骤：

1. 保留 `Qwen3VLMultimodalRotaryEmbedding.forward` 的position与interleaving逻辑，先在 `text_model.py` 的 `_preprocess` 之后、decoder之前生成每token32维cos/sin。每个pipeline stage在自身microbatch上准备一次，跨本stage的full-attention层和Q/K复用；无需为此新增跨PP通信。
2. 当前16层/PP4通常每stage只有1个full attention，所以跨层缓存收益远小于ViT27层，但Q/K仍可共享；完整64层时stage内复用价值更大。GDN不加入此路径。
3. 增加partial rotary kernel：逻辑Q/K `[S,B,H,256]` 或packedTHD，**前64维分成两个32维半段旋转，后192维精确拷贝**。支持不同Q/K heads（TP分片后的GQA），不能硬编码全局24/4。表按实际 `(sequence,batch)` token顺序、CP本地分片和attention实际QKV布局对齐；SP的输入分片不等于attention QKV一定已分片，需在真实QKV处核实。先支持当前batch size=1、CP1，其他模式明确校验，再扩展。
4. 关键数值差别：当前语言 `apply_rotary_pos_emb_in_fp32=False`。MCore先FP32算cos/sin后转Q/K dtype；BF16乘法结果及加法存在中间舍入。视觉路径则是FP32全程后最后一次转BF16。**直接套视觉FP32 fused kernel会改变语言数值语义。** 先实现兼容模式，显式匹配系数dtype和中间乘法/加法舍入；或将更高精度模式列为独立选项，用明确误差门槛及训练验证接受变化，不能宣称零误差。
5. backward按现有低精度语义独立对照，不仅凭数学上的“sin取负”保证bitwise梯度一致；后192维梯度透传。保留非连续输入、packed absolute token定位、recompute显式传表、MTP未适配时的保护。
6. 然后考虑Q/K联合dispatch；若该模型实际启用QK norm，可另评估norm+partial rotary融合，但先检查norm的参数/反向及TP语义。

**验证**：逐token构造T/H/W位置的纯文本、图片、视频、混合batch，检查 `[11,11,10]` 交错来源；长位置/非零offset、非方图、多个样本、64/256维透传、不同Q/K heads和stride。对齐state_dict的full-attention单层输出与所有输入/参数梯度；再做trainable语言模块集成与20步训练。通过前不支持的CP>1、MTP、FP8等应明确拒绝。按每个PP stage采prof；rank4是分离语言开销的重要对照。旧rank4语言旋转kernel sum约30.451ms/step，仅作量级参考，新profile未做完整语言旋转duration归因，不预测大幅整步收益。

### P4：视觉 CUDA/HIP Graph

Graph将一串GPU任务的依赖和地址捕获后重放，降低Python/driver逐个发射的host成本；它不自动减少乘加量、不能替代低效Conv3d算法修复。当前是HIP设备，PyTorch仍暴露 `torch.cuda.CUDAGraph` 等API；实际捕获兼容性和收益须在当前DTK验证，不能仅因NVIDIA实现存在就认定可用。

**仓库已有入口但不能直接开**：

- MCore有 `VisionTECudaGraphHelper`，Bridge training有视觉graph设置；但当前 `Qwen35VLModelProvider` 未声明Qwen3VL相应的完整 `vision_cuda_graph_impl/scope/max_seq_length` 字段。
- 现有视觉padding路径会丢掉PackedSeqParams，只处理padding mask，无法保证不同图片/帧相互不可见。在每样本两图的数据上直接启用可能改变attention结果。
- helper的sample inputs未完整适配实际视觉RoPE；默认最大长度 `2304/4=576` 是位置表/merge推导，不是ViT输入总patch数上限。必须按merge前token数选bucket。
- helper训练反向假设、microbatch索引、共享buffer、frozen视觉与输出生命周期需要专项检查。当前flash视觉RoPE与Graph组合是显式拒绝的，不能只删除保护就上线。

**实现顺序**：以已完成的P0/P1为基础，让CPU几何准备、同步与动态分配留在capture外；从冻结ViT decoder的forward-only单层/层组graph开始，不先捕获全部训练迭代。固定dtype、shape、buffer地址，按总patch数/序列边界形状建立有界bucket；静态buffer包括hidden、cos/sin和cu_seqlens，maxlen为CPU int且同bucket稳定。保留真实varlen帧边界；无法静态支持的形状eager fallback。若用padding，额外构造有效独立序列边界，不能只加尾部padding mask。

下一阶段可先graph MLP而attention保持eager，随后扩到带varlen attention的layer组。训练视觉时再加backward/recompute、RNG、梯度累积与PP microbatch索引管理。在Bridge层实现wrapper/适配；不直接修改嵌入MCore。

**特别风险**：即使vision冻结，language backward也可能保存vision输出用于参数梯度；下一microbatch replay覆盖输出地址会破坏之前microbatch。需要明确输出buffer所有权、多buffer/ring或clone到消费者拥有的存储，测算额外copy/显存是否抵消收益。Graph不能缓存image内容；同形状的新输入必须每次更新。

**验收**：至少两组不同图片/位置但同bucket输入，以及跨bucket/尾部padding/多图多帧，eager与replay输出、下游语言梯度对比；确认graph确实重放，无每步重新capture。检查显存峰值、buffer生命周期、recompute、设备错误、fallback次数和host launch下降。捕获/warmup不纳入10～20稳态窗口；若capture未在10步前完成，延后计时窗口而不硬算。仅在实际更快后扩大scope。

## 8. 之前的问题与解决方法

| 问题 | 已采取方案 / 后续注意 |
| --- | --- |
| 容器PyTorch识别0GPU，HIP100；监控却有设备 | 旧DTK `libhsa-runtime64.so.1.5.0` 优先于主机1.11.0。仅对本实验进程prepend `/opt/hyhal/lib:/opt/hyhal/lib64`，恢复8卡；未改全局库或重启容器 |
| 主脚本再次source `/opt/dtk/env.sh` 覆盖环境 | 通过已有 `DTK_ENV` 指向实验目录 `dtk_compat_env.sh`，先source DTK，再prepend正确HSA库 |
| 主机canonical路径 `/public2/...` 容器无此挂载 | 传给Docker的路径统一 `/public/home/yugeng/work/...`，临时host脚本用abspath而非resolve；容器只挂载work范围 |
| editable install导入原repo | `source env/env.sh` 后核实实际 `vision_rope.__file__`；PYTHONPATH须指向优化worktree的根/HCU/Bridge/MCore |
| CPU测试import TE会探测设备 | 单测用实际源码文件加载及AST隔离纯方法/MCore参考函数；不伪造真实GPU测试结论 |
| FA公共rotary wrapper不接受BF16输入+FP32表 | 使用已有低层kernel并检查ABI；无额外Q/K cast、无静默fallback |
| 非2次幂head_dim72 | power-of-two tile配mask；真实设备前反向覆盖，不能把FlashAttention kernel名dim96当成真实head_dim |
| BF16 interpreter约一半结果相差1ULP | interpreter store截断，GPU编译实际RNE；保留FP32 interpreter，BF16以真实GPU检查和IR为准，不放宽GPU精度掩盖问题 |
| reentrant checkpoint不支持autograd.grad | 测试改用backward和.grad；重计算detach后对象身份不同，用共享storage指针验证表复用 |
| 真实ViT单测12项最初PG初始化失败 | fixture显式传 `ProcessGroupCollection.use_mpu_process_groups()`；生产构建本已显式传PG，未改生产PG逻辑 |
| 主机Python3.6，容器无uv/ruff | host分析脚本兼容3.6；uv/ruff仅装实验目录target，不改变训练依赖；容器执行GPU/现代Python代码 |
| 其他容器反复中途启动8卡作业 | 记录显存/利用率/PID/cgroup，不杀别人作业；被干扰测量标记contaminated。早期诊断/重试结果保留但不充当干净计时 |
| profiler导出严重污染后续step | prof5，计时10～20；旧prof中step6曾约166s，不能纳入训练收益。算子prof单独于event计时采集 |
| 原矩阵耗时过长，需求改为粗测 | reference已运行则让30步结束；停止旧调度器；flash一次20步，不再cached/解冻/三遍。取消的prof18启动保存在archive，不当完整结果 |
| 旧自动summary会覆盖新文档为等待状态 | 不重跑旧 `summarize_experiment.py` 生成流程；旧RoPE看quick_compare，本轮patch/metadata看qwen38_patch_metadata_20261111的summary与本handoff |

## 9. 历史RoPE工件与命令（当前C0/C1及备份见第0节）

### 9.1 已有工件

实验根目录（下称 `$EXP`）：
`/public/home/yugeng/work/checkpoints/qwen38_rope_gpu_20261111_1236`

- 最终结论：`$EXP/quick_compare/summary.json`、`timing_steps.csv`、`report.md`。
- flash最终训练：`$EXP/quick_compare/flash_20steps/`，包含command.json、训练YAML、launcher.log、gpu_monitor.jsonl、result.json；训练日志 `train_logs/mpi_2026-11-11-1456.log`。
- 用户当前prof：`$EXP/quick_compare/flash_20steps/torch_profile/rank-0.json.gz`（约28MB）、`rank-4.json.gz`（约8.2MB）。第5步；stack/shapes关闭。
- 最终reference：`$EXP/timing/frozen/reference/r1/train_logs/mpi_2026-11-11-1441.log`，30步，存在干扰。
- 隔离算子结果：`$EXP/microbenchmark/fwd/benchmark.json`、`fwd_bwd/benchmark.json` 及各backend/T的Chrome JSON；查看result.json和gpu_monitor确认有效性。
- 不可当干净计时：`$EXP/diagnostic_contended_fwd/`、`diagnostic_contended_bwd/`、`invalidated/`。诊断仅用于kernel结构；无效旧结果未删除。
- GPU单测：`$EXP/gpu_tests_retry.xml`、对应log；首次fixture失败记录保留。
- 本次追加分析：`$EXP/preamble_analysis.json`、`.log`、`analyze_preamble.py`、`preamble_events_compact.json`。
- 复现实验输入：`$EXP/inputs_and_source_sha256.json`、`source_snapshot/`、根/Bridge diff和status；继续修改源码后旧hash校验会失败，这是设计行为，不要跳过后混称相同源码。
- 旧图文所指reference prof：`/public/home/yugeng/work/checkpoints/mantis-birds-one-node-16layers-20261110_202000`。
- 初期正确性、编译、源状态备份：`/public/home/yugeng/work/checkpoints/qwen38_rope_20261111`，含 `baseline.json`、`inherited_bridge.patch`、CPU/interpreter结果及离线编译IR/HSACO。
- 工作树相关历史文档：`docs/models/qwen/qwen38-vision-rope-{optimization,gpu-20261111,performance-20261111,quick-20261111}.md`；其中旧状态以本文为准。

### 9.2 后续单组训练命令模板

先在节点监控确认资源，不假设“节点之前空闲”就一直无其他作业。进入 `yg_qwen38` 容器后，使用现成compat脚本；不运行旧矩阵脚本。以下命令只是复现模板，本轮未启动：

```bash
cd /public/home/yugeng/work/Megatron-bridge-das-perf-rope
export DTK_ENV=/public/home/yugeng/work/checkpoints/qwen38_rope_gpu_20261111_1236/dtk_compat_env.sh
export TRAIN_ITERS=20
export PROFILE=true PROFILE_STEP_START=5 PROFILE_STEP_END=6
export PROFILE_RANKS='[0,4]' PROFILE_RECORD_SHAPES=false PROFILE_WITH_STACK=false
export MASTER_PORT=29775
export VISION_ROPE_BACKEND=flash
export VISION_METADATA_BACKEND=prepared
export SAVE_DIR=/public/home/yugeng/work/checkpoints/qwen38_next_flash_$(date +%Y%m%d_%H%M%S)
bash examples/qwen/run1_qwen38_vl_single_node.sh train \
  model.vision_rope_backend=flash scheduler.lr_decay_iters=30
```

对照组改backend与独立SAVE_DIR，其他配置不变。新优化应对照“当前flash基线”，不要只与最早reference比较后把已有收益也归给新改动。默认单组脚本冻结vision/projection；解冻须显式CLI覆盖。CPU/GPU targeted测试示例（容器内，工具PATH按实际实验目录设置）：

```bash
export PATH=/public/home/yugeng/work/checkpoints/qwen38_rope_20261111/test_tools/bin:$PATH
source "$DTK_ENV"
source env/env.sh
uv run --no-project --python /usr/local/bin/python python -m pytest \
  tests/unit_tests/hcu/models/test_qwen38_vision_rope.py \
  -q -o 'markers=unit: isolated unit checks'
```

### 9.3 每个后续优化的最小交付

1. 先列方案、文件/符号、语义不变量、开关与回退方式，再实现；仍在当前worktree继续即可，大改Graph可单独再建分支/worktree，但未提交改动须先可靠保存。
2. 针对真实形状比较同权重输出和梯度；低精度语义变化单独说明。只跑相关测试。
3. 一次baseline、一次新方案、各20步即可满足当前粗测偏好；prof5、10～20取平均，确认初始化/capture/export影响已结束。不自动恢复三轮矩阵。
4. 保存完整命令、解析后配置、环境版本、源码hash、数据/processor配置、设备监控、原始日志/prof、步级CSV和结论。
5. 同时报“结构变化、纯kernel时间、host窗口、端到端step”，明确外部干扰与关键路径；不要将rank4的NCCL等待时间直接诊断为网络带宽不足。
6. 对应更新本handoff的进度与改动定位；P0已撤回并备份；P1已在Conv3d下完成C0/C1和独立组件验收并默认开启。历史B0/B1/B2保留；P2（Q/K联合）与P4（Graph）尚未实现，P3（语言M-RoPE）暂缓。
