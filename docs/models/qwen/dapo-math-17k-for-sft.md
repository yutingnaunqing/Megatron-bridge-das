# DAPO-Math-17k：数据结构与本项目使用建议

核查日期：2026-11-13。

## 是什么数据

DAPO 是字节跳动 Seed 与清华 AIR 等团队公开的大模型强化学习项目。其训练集 DAPO-Math-17k 面向英文竞赛数学，题目来自 AoPS 和竞赛官网，并经过筛选、改写，使目标答案成为便于规则校验的整数。它覆盖的例题包括代数、数论、几何、计数和概率。[DAPO 论文，第3.5节](https://dapo-sia.github.io/static/pdf/dapo_paper.pdf)

“17k”指约1.7万道题的规模，不代表1.7万条完整推理过程。发布文件采用 Parquet，Hub 元数据显示 Apache-2.0。当前发布页显示1,791,700行、约299 MB，因此下载后必须检查实际行数和唯一题目数，不能把发布文件行数当作独立题目数量。[官方数据集](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k)

## 每行有什么

官方 viewer 中展示的字段如下，以下为结构示意，不是逐字复制的原始样本：

```json
{
  "data_source": "math_dapo",
  "prompt": [{"role": "user", "content": "数学题目及分步回答、最终答案格式要求"}],
  "ability": "MATH",
  "reward_model": {"ground_truth": "42", "style": "rule-lighteval/MATH_v2"},
  "extra_info": {"index": "题目标识"}
}
```

`prompt` 是输入题目；`ground_truth` 是供规则奖励比较的最终答案；`style` 标记奖励评分规则，**不是完整的奖励模型权重，也不是解题过程**。`extra_info.index` 可用于追踪样本。官方发布结构没有可直接当成 assistant 目标的完整参考推理回答。[字段与样本](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k/viewer)

## 为什么适合 RL，但不能直接做推理 SFT

RL 可以让模型对同一题生成多个回答，再用最终答案是否正确给奖励，不需要每个 token 的参考答案。SFT 则需要指定模型应该模仿的 assistant 输出。

| 使用方式 | 是否适合 | 实际训练信号 |
| --- | --- | --- |
| 原始 prompt + ground_truth 用于 DAPO/GRPO | 适合其原始用途 | 模型生成答案后，通过规则评分 |
| 把 ground_truth 包装为 `Answer: 42` 做 SFT | 技术上可以 | 只监督最终答案格式和数字，不教授分步推理 |
| 给每题补充经验证的教师解答后做 SFT | 适合推理 SFT | 监督解题过程及最终答案 |
| 把 prompt 本身当作 assistant 回答 | 不适合 | 会学习复述题目，监督目标错误 |

论文/仓库的训练成绩来自完整 RL 系统，不能理解为“下载这个数据集做普通 SFT 就能得到同样结果”。[官方项目](https://github.com/BytedTsinghua-SIA/DAPO)

## 数据质量与划分

发布仓库的讨论中，社区用户报告过大量重复记录，以及同题对应不同奖励答案的问题；“意外重复100次”和清洗后17,391条是讨论中的报告，**本项目尚未下载全量并独立复现这些统计**，不作为已验证的官方保证。[相关讨论](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k/discussions/3)

使用前应固定数据 revision，按规范化题目去重，核对同题答案是否冲突；先按题目划分训练/验证，再生成多条教师解答。同一题的不同推理轨迹不能跨集合，否则验证会泄漏。`extra_info.index` 相同或题目相同的行也不能按独立样本随机拆分。

若补充教师解答，应检查最终答案与 ground_truth 一致，并抽查推导过程。答案一致并不能保证每一步推理正确。教师输出的许可证、thinking 标记、最终答案格式和 tokenizer 模板都要匹配实际训练目标。几千到上万 token 的解答可能超过当前 SEQ1024/2048测试窗口；第一轮应筛选完整短样本，不能截掉全部有效监督后继续训练。

## 本项目支持范围

建议先用少量现有 messages 格式的 Tulu/UltraChat 数据验证16层训练链路，再决定是否制作 DAPO 教师解答数据。当前不包含 RL trainer、rollout 或奖励函数，也不会自动调用外部模型生成教师答案。

新增的 `data_preparation/prepare_qwen38_text_sft.py`：

- 接受标准纯文本 messages JSONL/Parquet，并按最后一轮 prompt 去重后划分数据。
- 默认拒绝原始 `prompt/reward_model` 数据，避免误把整数奖励标签当作解题过程。
- 若用户已经提供经验证的完整教师回答列，可通过 `--solution-field verified_solution` 转成 messages；该参数只声明选取哪一列，工具不声称已验证数学正确性。
- 第一版每个 prompt 保留一条回答，适合 smoke test；全量多轨迹数据需要另做按题分组和答案冲突审计。

命令示例：

```bash
python3 data_preparation/prepare_qwen38_text_sft.py \
  --source /path/to/dapo-with-verified-solutions.parquet \
  --solution-field verified_solution \
  --output /path/to/prepared-dapo-sft \
  --limit 1024 --validation-samples 64 \
  --check-tokens --model /path/to/Qwen3.8-27B-text-16layers \
  --seq-length 1024 --thinking-mode default
```

数据选择结论：**可以把 DAPO 作为数学题来源；原版不宜直接作为当前推理过程 SFT 的训练集。若只想测试最终答案 SFT，应另行明确采用这个监督目标，不与完整推理 SFT 的效果混用。**
