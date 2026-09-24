# Qwen3.8-Flash-Next QSA 算子需求

本文以 `config.official.json` 的文本配置为准。它描述的是供 HCU kernel 实现和验收使用的计算契约；
不是当前 PyTorch 参考实现的性能宣称。

## 固定结构参数

| 项目 | 值 |
|---|---:|
| hidden size | 2560 |
| query heads / KV heads | 24 / 2 |
| GQA group size | 12 query heads / KV head |
| attention head dim | 256 |
| partial RoPE dim | 64 |
| indexer heads / dim | 4 / 128 |
| index block size | 4 tokens |
| token budget | 2048 tokens = 512 blocks |
| indexer KV heads | 1 |

输入、线性权重和 attention 输出是 BF16。indexer score、attention logit、softmax 概率以及 Stage-2 KL 的规约须保持 FP32。
模型的 `rope_theta=10,000,000`；RoPE 只作用于每个 256 维 attention head 的前 64 维。

## 逐层的计算契约

对于 batch `B`、序列长度 `S`，令 `N=floor(S/4)`。

| 阶段 | 输入与输出 | kernel 需要完成的工作 |
|---|---|---|
| Indexer 投影 | `[B,S,2560] -> [B,S,640]` | 得到 4 个 128 维 query 和 1 个 128 维 token key |
| 压缩与打分 | query `[B,S,4,128]`；block key `[B,N,128]` | 每 4 token 对 key 求 FP32 平均、RMSNorm、RoPE；QK 点积、ReLU、对 4 个 index head 求和，产出 `[B,S,N]` FP32 score |
| 块路由 | score `[B,S,N]` | 每 query 在可见完整块中选 `min(512, floor((t+1)/4))` 个块；分数完全相等时按 block ID 升序，输出 block ID 与有效计数 |
| 主 GQA | Q/gate `[B,24,S,256]`，K/V `[B,2,S,256]` | 以 12:1 GQA 映射执行部分 RoPE、缩放点积、causal softmax、V 加权和和 sigmoid gate；不要物化重复后的 K/V |
| Stage-2 KL | attention 概率与 indexer score | 对 24 个 attention head 求和，在每个 4-token block 上取 max，仅保留所选块、归一化；将其作为停止梯度的 teacher，与 indexer `log_softmax` 计算 KL |

当前训练路径固定使用未打包、无 KV cache 的因果文本。首个完整 QSA 层是第 4 层（从 1 开始计数）。

## 稀疏语义

第 `t`（零开始）个 query 的可见完整块数是 `floor((t+1)/4)`。它保留：

```text
min(2048, 4 * floor((t+1)/4)) 个来自 TopK 完整块的 token
+ 当前未凑满 4 token 的因果尾部
```

所以 `S <= 2048` 时，全部可见 token 都被保留，QSA 没有稀疏收益。`S=4096` 的最后一个 query 保留
2048 个历史 token，约为全因果历史的一半。kernel 的验收长度至少应包括 `128`、`2048` 和 `4096`。

选择出的块可以按 score 顺序存储；attention 对 key 的排列不敏感。若需要与本参考实现精确复现，
同分时应选较小 block ID。当前 PyTorch 参考实现为了检查数学语义，仍创建 `[B,24,S,S]` 的完整 FP32
logit 和概率张量以及 `[B,S,S]` mask；正式 kernel 必须避免这些张量。

## 建议的算子边界

1. `qsa_indexer_route`：投影、压缩、RMSNorm、部分 RoPE、score、稳定 TopK，输出每 query 的 selected block IDs/count。
2. `qsa_sparse_gqa`：Q/K/V/gate 投影与部分 RoPE，按 block IDs gather K/V，完成 causal GQA attention 和 output 投影。
3. `qsa_stage2_kl`：由 sparse attention 内部概率直接做 head/block reduction，避免保存完整 attention probability；teacher 必须停止梯度。

第一版可将线性投影留给 GEMM，将 route、sparse GQA 和 KL 分成三个 kernel。性能版再融合中间结果，
但需要保持 BF16 输入输出、FP32 规约、4-token 分块和同分确定性。

## 基准和验收

容器中执行：

```bash
cd /public/home/yugeng/work/Megatron-bridge-das
bash examples/qwen38/benchmark_qsa.sh smoke \
  --sequence-lengths 128,2048 --warmup 2 --iterations 5 --profile-ops

# 只做一次长序列前向，验证超过 budget 后的路由形状和显存。
bash examples/qwen38/benchmark_qsa.sh smoke \
  --sequence-lengths 4096 --forward-only --warmup 1 --iterations 1
```

每次会产出 `runs/qwen38/qsa_*/qsa_benchmark.json`，其中包含实际设备、总耗时、峰值 PyTorch 显存，
以及每个长度的张量契约和选中 token 比例。`--profile-ops` 还记录当前 Torch/HIP 容器能提供的算子摘要；
如果该版本的 Kineto 不支持 HIP，结果会明确写出不可用原因。

替换为新 kernel 前后，使用相同随机种子比较 QSA output、输入梯度和 indexer 权重梯度。短序列可使用
`tests/qwen38/test_reference.py` 的上游对齐测试；长序列还应检查 selected block IDs、每 query 的有效数量、
KL 和最终输出。性能比较只可在相同 `B/S/dtype/训练或仅前向` 条件下进行。

## BW1000 参考基线（2026-09-15）

在 `yg_mbridge`（PyTorch 2.10.0、HIP 6.3.26113、1 × BW1000 64G）上，以 `smoke` 配置、batch=1 测得：

| S | 模式 | 稳态时间 | PyTorch 峰值已分配显存 | 最后 query 选中 token | 所有 query 的选中/因果 token 比 |
|---:|---|---:|---:|---:|---:|
| 128 | 前向+反向 | 8.13 ms/iter | 0.258 GiB | 128 | 1.000000 |
| 2048 | 前向+反向 | 24.70 ms/iter | 1.987 GiB | 2048 | 1.000000 |
| 4096 | 仅前向 | 29.48 ms/iter | 5.010 GiB | 2048 | 0.750305 |

`S=4096` 的前向只跑了 2 次，不应与 3 次前反向测量作速度比较；它只用来验证超过 budget 的路由分支和参考实现的显存增长。
2048 的单次 HIP profiler 摘要中，`aten::bmm` 占 21.41% 自身设备时间，`aten::mul` 占 17.72%，`aten::copy_`
占 17.28%，`aten::mm` 占 9.57%，`aten::masked_fill_` 占 6.61%。这是 PyTorch 运算拆分和 profiler 开销下的诊断，
不是 kernel 的性能指标。完整原始结果在本机被忽略的 `runs/qwen38/qsa_smoke_*` 目录。
