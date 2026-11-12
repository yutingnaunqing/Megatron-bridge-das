# 冻结patch Conv3d Graph：精度与首轮性能测试

> 22:37最新核实：G0完成20步，mean14948.8ms、median14954.0ms，四rank prof齐全；G1在首步前因MIOpen Graph capture失败退出，无训练prof。控制器已停止，旧waiting描述属于历史。外部作业同期重启，根因未确定；尚无整步加速结论。

日期：2026-11-11。原Conv3d保留，flash RoPE与prepared metadata保留。Graph默认关闭；本轮实现仅冻结patch embedding，不含decoder、collective或Graph backward。

## 已完成的验证

- HIP capture探针通过；101项定向测试通过，2项跳过。
- 完整27层ViT、TP2、BF16权重，24项比较全部逐位一致：metadata前后、Graph前后、重复控制、逐层/最终输出及可训练eager回退的输入/参数梯度。
- 真实1844-patch双图，每rank实际capture1次/replay2次；不同输入与输出所有权由独立单元测试覆盖。
- 测试用固定随机初始化，无optimizer更新；不代替预训练任务准确率测试。
- 训练入口dryrun及Ruff检查通过。

## patch组件配对计时

1680 patches、1536输入宽度、1152输出宽度；每种dtype交替顺序30对。包含输入copy、输出clone及每次同步；捕获和预热不计入稳态。独立组件结果不能当作整步加速，节点共享且没有为该短组件测量采集完整资源监控。

| 输入dtype | eager wall中位ms | Graph wall中位ms | 下降 | eager GPU中位ms | Graph GPU中位ms | 不同输入精确相同 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| torch.float32 | 24.6970 | 21.5132 | 12.89% | 24.6370 | 21.4605 | [True, True] |
| torch.bfloat16 | 24.6566 | 21.4649 | 12.94% | 24.6115 | 21.4270 | [True, True] |

prof为`patch_float32.json`/`patch_bfloat16.json`。每份包含一次eager与一次replay，可见一次hipGraphLaunch；eager有3360次hipExtModuleLaunchKernel调用。这是在复用launch DAG，不能声称卷积数学计算或内部kernel都被融合成一个。

FP32输入首次capture+warmup约2947ms，BF16约98ms（受首次库初始化影响）；不可把捕获成本混入稳态，也不能忽略训练启动代价。每形状预算为估计owned显存，非进程总reserved上限。

## 完整单节点对照

G0_eager：原Conv3d + flash + prepared + Graph none。
G1_patch_graph：同G0，仅patch Graph；精确token白名单1680/1728/1696/1632、4条/256MiB估计预算、前192次调用可capture、预热3次；无padding，不合并帧。

各一次20步，第5步prof（rank0/2/4/6），第10～20步计时，不重复三遍。控制器已启动并持续检查空闲；20:51记录外部8卡作业各约33GiB，当前waiting_resources，训练尚未开始。以实验目录`state.json`的实时状态为准；结束后`summary.json`、`timing_steps.csv`及G0/G1的`torch_profile`为原始证据。尚不能宣布整步加速。

工件目录：`/public/home/yugeng/work/checkpoints/qwen38_graph_20261111`。主handoff包含代码逐文件定位；实现原理见metadata详解与Graph方案。
