# Qwen3.8-27B 全参 SFT 长步数稳态方案

> 2026-11-16存储更新：旧runs内测试checkpoint及参数比较副本已按用户要求清理，历史日志/曲线/审计JSON保留；本文涉及旧checkpoint的步骤仅作历史方案，不再可直接恢复。当前启动与状态见[handoff](qwen38-text-sft-handoff.md)。

本轮顺序：n05+n07，先关闭MTP完成500步及门槛审计，通过后独立开启MTP完成500步。沿用现有ROCm容器和已验收TP4/PP2/DP2，不安装依赖、不修改MCore、不使用Socket数据通信。

| 配置 | 取值 |
| --- | --- |
| 模型 | 完整64层全参；无MTP约26.896B，MTP1约27.321B |
| 拓扑 | 2×8卡，TP4/PP2/DP2、CP1，BF16参数/FP32梯度和Adam状态，distributed optimizer |
| 数据 | 官方UltraChat train_sft/test_sft，实际tokenizer筛选16384条训练、256条验证，去重并排除跨split重叠；SEQ1024、thinking disabled、assistant-only、无packing |
| 优化 | GBS32/MBS1，LR1e-5，warmup20、cosine至500步，seed1234，full/uniform/1重计算；历史完整设置见当时的运行配置与审计文件 |
| 阶段A | 无MTP从官方纯语言权重训练1–250，保存退出，恢复251–500 |
| 阶段B | 仅A通过后，从语言部分相同的官方MTP1权重独立训练1–250，保存退出，恢复251–500；MTP系数0.1 |
| 验证/保存 | 每100步完整验证256条（EVAL_ITERS8）；500末尾也验证；250步只保存退出（Bridge提前退出路径不执行末尾额外验证）；250/500保存模型、optimizer、scheduler和RNG |
| 监控 | 日志每步；自动设备采样每30秒；MTP梯度/更新在首次step、每100步和250/500边界抽样，避免逐步审计干扰稳态计时 |

## 预先固定的验收门槛

1. 所有500步主loss/grad norm/LR有限，MTP阶段辅助loss也有限，无NaN/跳步、OOM、IB报错，16rank成功退出；64层覆盖、全参数可训练、无视觉塔。
2. 主训练loss第451–500步均值≤第21–70步均值的97%，50步分块均值整体线性趋势为负。loss不要求逐步单调，完整曲线与窗口统计必须保留。
3. 最终验证loss低于首次第100步验证loss。验证集全覆盖且固定，训练下降而验证持续升高时不判通过，先分析过拟合或数据/优化设置。
4. 保存250/500状态完整：optimizer step250/500、scheduler samples8000/16000、RNG8份。恢复不重新导入HF，首个训练step251、样本计数8032，LR与500步统一schedule连续。
5. MTP真实权重导入抽样匹配、8个MTP rank的代表参数梯度有限非零并更新。当前短测已证明同配置恢复/连续精确一致，本轮增加长期恢复连续性，不额外重复500步连续对照。
6. 记录稳态iteration时间（排除warmup、每次验证/保存之后及恢复首步）、PyTorch峰值、设备采样及后半程显存趋势。设备采样不视为连续绝对峰值；同一次恢复进程内比较step301–350和451–500的每GPU设备采样中位数，增加超过2048MiB则不通过，先分析缓存或泄漏。

预计每种模式2–2.5小时，整轮含数据准备约4.5–6小时；实际启动后按既有13–14秒/step及初始化/保存耗时给出结束窗口。已完成的历史长测仅在计划250/500退出或异常事件检查日志/状态；后续新启动的阶段改为每10分钟进行健康检查（见下节），不得回填为历史长测已执行。A门槛不通过则暂停B，保留证据并定位原因，不改阈值凑通过。

## 实现与证据位置

- `runs/qwen38_text_sft_steady_20261116/run_phase.sh`：本轮配置、IB/cache及独立SAVE_DIR。
- `run_workflow.py`：四段顺序执行、审计和A→B门槛。
- `audit_phase.py`：复用已验收checkpoint/模型/IB审计，适配250/500和稀疏MTP审计。
- `assess_run.py`：固定训练窗口、验证趋势、step时间和设备采样分析；未通过返回非0阻止后续阶段。
- `examples/qwen/qwen38_mtp_audit.py`：新增MTP_AUDIT_INTERVAL（默认1保持旧行为），独立计算真实step，首次/保存/末步必采样。
- `tests/unit_tests/qwen38_text_sft/test_mtp.py`：稀疏审计250步及恢复后251–500步编号回归。

本轮未改动现有`env/env.sh`中的用户注释，不覆盖历史验收目录，不提交Git。

分析产物同时保存逐step CSV、主/辅助训练loss的50步移动均值及固定验证loss曲线（PNG/SVG）；使用容器已有Matplotlib，配置缓存置于证据目录，未安装依赖。


## 第二轮：仅调整学习率的受控对照

第一轮LR1e-5完成500步、250步保存恢复和全部稳定性检查；训练窗口下降2.0489%，未达固定3%，block slope负、验证100/200/300/400/500步均持续下降（0.8849659→0.8578868）、后半程显存中位数增长0MiB，因此未启动MTP。证据`runs/qwen38_text_sft_steady_20261116/nomtp_assessment.json`，保留原始未通过结论，不能把稳定运行等同于已满足全部门槛。

第二轮仅LR由1e-5增加为2e-5；固定500步、warmup20/cosine、16384/256条数据、TP4/PP2/DP2、250恢复和所有验收门槛不变。重新从同一官方权重启动，不从首轮500步checkpoint偷偷延长、重置scheduler，也不缩小数据集制造拟合。假设是增加学习速率能在同样训练步数内得到更明显且仍能泛化的loss下降，需要实际结果验证；若验证loss反转或数值异常则仍不通过。第二轮无MTP通过后才执行同学习率MTP500步。预计追加约4.5–5.5小时，启动后记录实际结束窗口。

第二轮入口与证据`runs/qwen38_text_sft_steady_lr2e5_20261116/`；相对第一轮代码变动仅run_phase.sh显式LR、独立SAVE_DIR/端口和assess_run.py对应的预期LR曲线，未改变门槛。

## 最终执行状态

2026-11-16 07:50完成。首轮LR1e-5无MTP500步稳定、验证持续下降，但训练窗口2.05%未达固定3%，保留未通过结论。第二轮仅改LR2e-5，先无MTP500步全部门槛通过，再执行MTP500步全部通过；两模式250保存恢复正常，训练窗口均下降约3.37%，验证loss持续下降、无NaN/跳步/IB错误、后半程显存中位数增长0。首轮最终验证loss最低，因此不据通过幅度门槛推荐更高LR或宣称MTP效果提升。完整表格、限制、产物和修改位置见handoff“长步数最终结果”。本轮所有训练及采样进程已退出。


## 后续运行的10分钟健康检查

根据用户最新要求，后续通过第二轮run_workflow.py启动的阶段每600秒检查一次：真实launcher进程退出码、最新optimizer step、主/辅助loss和grad norm等有限性、跳步/NaN计数、日志错误。初始化单独显示starting；训练step连续20分钟未推进或初始化超过30分钟时记录attention_required。异常不自动重启任务；launcher非0退出仍由原编排停止本轮peer，不使用Socket回退。

实现位置：`examples/qwen/qwen38_training_health.py::TrainingHealthMonitor`；接入`runs/qwen38_text_sft_steady_lr2e5_20261116/run_workflow.py::run_phase`，日志`health_checks.jsonl`。原实际执行的controller已保留为`run_workflow.executed.py`，SHA256与原request.json一致，历史长测证据不变。本项为测试完成后的后续运行增强，不声称已结束的GPU训练按该频率检查。

CPU进程生命周期验证结果保存在`monitor_validation_container/validation.json`：检查正常推进、-inf、跳步、停滞、正常/异常退出；未为验证监控器重启GPU训练。首个沙箱异步CPU验证的子进程退出观察曾卡住，已仅终止本轮已确认PID并改在现有容器完成验证，原临时记录保留，不作通过证据。
