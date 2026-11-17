# Parquet 数据及 Qwen3.8 SFT 接入说明

## 是否还需要预处理

**Parquet 可以被数据加载器直接读取，但文件格式本身不会完成 SFT 预处理。** UltraChat 的 Parquet 保存的是结构化对话文本，不是当前 Qwen 模型的 token、label 或 loss mask。无论从 Parquet 还是 JSONL 读取，训练前或加载时都必须应用模型的 chat template、tokenizer、长度策略及监督范围。

当前项目使用已经实测的路径：

```text
官方 train_sft / test_sft Parquet 分片
  → PyArrow 按 batch 读取原生 messages
  → schema / 角色 / 文本检查、按首个用户问题去重、长度与有效监督检查
  → training.jsonl / validation.jsonl + 数据与 token 审计
  → Bridge GPTSFTDataset 加载时套用 Qwen 模板、tokenize
  → next-token labels、assistant-only loss mask、padding
  → 模型训练
```

这里转换 JSONL 是为了适配当前 recipe 和保留可审核样本，**不是 Parquet 必须先转成文本才能训练**。当前启动脚本的 `DATA_PATH` 必须指向已准备的 JSONL 目录，不能直接替换为原始 Parquet 目录。现有准备入口为 `data_preparation/prepare_ultrachat_sft.py`，训练入口沿用 `GPTSFTDatasetConfig(dataset_root=...)`。

具体处理：保留完整多轮 messages，不将整段对话拼成普通字符串；仅 assistant token 参与 loss，user/system/padding 被 mask；拒绝超出目标序列长度或没有监督 token 的完整会话，避免截掉答案。训练取官方 `train_sft`，验证取 `test_sft`，不混入 `train_gen/test_gen`。同一 thinking 模式与序列长度的16层/64层模型共用 tokenizer，因此已准备的1024/64条短样本可以复用。

目前1024/64条是链路测试子集，不是全量训练集。准备脚本设置 `TRAIN_SAMPLES=0 VALIDATION_SAMPLES=0` 可处理全部可用样本；仍会去重、检查和过滤。SEQ1024 会过滤较多长对话，后续正式训练应先统计长度分布，再决定增大序列长度或增加合理切分/packing；不要将短样本上的链路成功解释为正式训练效果。

## 原理和文件结构

Parquet 是二进制列式存储格式。同一列的数据存放在一起，适合对大数据集进行列选择、编码和压缩。例如只读取 `messages` 列时，无需同时解码 `prompt_id` 等无关列。它可以表示整数、字符串，也可以表示 UltraChat 中的 `list<struct<role:string, content:string>>` 嵌套对话，读取后得到列表和字典，不需要对字符串做 `eval`。

文件从逻辑上分为：文件 → row groups（行组）→ 每列的 column chunks（列块）→ pages（页）。页保存编码/压缩后的数据，文件尾部的 metadata/footer 记录 schema、块位置和可用统计信息；标准文件首尾有 `PAR1` 标记。具体压缩算法由文件元数据决定，不能仅凭 `.parquet` 后缀认定为某一种压缩。

行组方便分批读取。支持的读取引擎可借助列投影以及统计信息跳过无关列/行组，但过滤是否能下推取决于表达式、嵌套字段和文件统计信息；不是任意条件都能免读数据。分片文件名中的 `00000-of-00003` 表示多个独立文件，不是必须二进制拼接的分卷；应把全部对应分片交给加载器。

与 JSONL 相比，Parquet 通常更紧凑、保留类型、按列读取更有效；JSONL 更容易逐行查看和编辑。两种格式都不能保证样本正确、无重复或适合某个模型。Parquet 也不规定 `train/validation` 语义、tokenizer 或训练目标，这些来自数据集和训练配置。

参考：[Apache Parquet 概览](https://parquet.apache.org/docs/overview/)、[文件格式](https://parquet.apache.org/docs/file-format/)。

## 如何读取

在已有 `yg_qwen38` 环境中，先检查 schema/行数，然后分批读取，避免将全量嵌套对话一次性转成 pandas DataFrame：

```python
from pathlib import Path
import pyarrow.parquet as pq

root = Path('/public/home/yugeng/work/datasets/HuggingfaceH4_ultrachat_200k')
for path in sorted((root / 'data').glob('train_sft-*.parquet')):
    parquet = pq.ParquetFile(path)
    print(path.name, parquet.metadata.num_rows, parquet.schema_arrow)
    for batch in parquet.iter_batches(columns=['messages'], batch_size=128):
        for row in batch.to_pylist():
            messages = row['messages']  # list[dict]，尚未 tokenize
            # 交给模型相关的模板/监督预处理。
```

Hugging Face Datasets 也支持读取本地 Parquet，无需上传或重新下载：

```python
from datasets import load_dataset

data = load_dataset('parquet', data_files={
    'train': str(root / 'data/train_sft-*.parquet'),
    'validation': str(root / 'data/test_sft-*.parquet'),
})
print(data['train'].features)
print(data['train'][0]['messages'])
```

具体 API 见 [Datasets 本地 Parquet 加载文档](https://huggingface.co/docs/datasets/loading)。如果以后希望 recipe 接收 Parquet，本版本 Bridge 已有 `HFDatasetSourceConfig(path_or_dataset='parquet', split='train', load_kwargs={'data_files': ...})` 配合 `GPTSFTDatasetConfig.hf_dataset` 的声明式入口；其 builder 仍会在内部规范化并物化 JSONL。该替代入口尚未接入本项目启动器，当前继续使用已验证、带过滤审计的准备脚本。
