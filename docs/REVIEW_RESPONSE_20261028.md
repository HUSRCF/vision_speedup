# 对外部审阅建议的实测核查

审阅基于 `9f88358`。其后已采用双卡 world 完整空间块＋时间 FFN／卷积，以及四个非 V2 分项的 VAE 改进。当前正式材料的 V2 为 301.930663 秒、五项合计 850.836587 秒，已通过官方 103 视频校验；各项独立运行后汇总。

## 主干与 head 跨步流水线

先测量其收益预算，再决定是否拆分图。使用保留的 209 文件单卡基线、比赛权重和真实首步输入；在第二份诊断 CUDA Graph 中插入显式 CUDA driver event-record 节点。没有替换、禁用或删除计算 kernel。

| 每步中位数，ms | Decision | World |
|---|---:|---:|
| 完整 denoiser | 222.510078 | 292.879349 |
| Head 之前的视频主干 | 214.258690 | 284.682236 |
| 完整 head | 8.231936 | 8.164352 |
| 9 次 SpatialSoftmax | 2.833408 | 2.821632 |
| 剩余 head | 5.401600 | 5.342720 |

每阶段先预热 12 次，再测 18 次。诊断图的三个 FP32 输出在计时前后均与原始图逐位一致，包含正负零；恢复原图后再次一致。RNG、权重版本、decision 的 138＋43＋8 个 native cache 和 world 的 23 SGEMM＋45 head cache 均保持。

SpatialSoftmax 列表中有共享 module 对象；诊断按对象去重安装 hook，再按真实调用计时，避免将嵌套 hook 重复计入。每阶段实测为 9 次调用，输出合计 **1,064,960 字节**，与审阅中的通信量推导一致。

按该首步样本计算 `11×49×min(主干+SpatialSoftmax, 剩余head)`，两阶段理想节省分别为 **2.911462 秒、2.879726 秒**，合计 **5.791189 秒**。这个条件性上限忽略通信、helper 初始化、图重构和流水线排空；不是完整推理的实测加速，也不证明后续所有步耗时完全相同。主干/head 存在可利用的独立性，但当前 head 所占时间不足以解释百秒差距，暂不优先投入大规模流水线重构。

代码见 [`profile_head_pipeline209.py`](../experiments/head-pipeline-209/profile_head_pipeline209.py)，全部样本、形状和验证证明见 [`head-pipeline-209`](../results/aggressive-20261028/head-pipeline-209)。脚本从完整单卡源码目录运行，需要比赛机既有依赖、原始生产运行参数及授权资源。它只生成测量，不生成正式成绩。

## Kahan 静态 high/low 预打包

这条建议指出的重复工作确实存在。此前已实现独立 packed-Kahan kernel，并完成七种实际几何的合成输入、原始与改动权重、清零/恢复、NaN canary、原始 workspace 污染、缓存刷新及三/四指针图参数见证检查。普通随机权重和 Half 可精确表示权重的矩阵结果均与当前 Kahan 路径逐位一致。这里的逐位等价对象是当前 Kahan kernel，不能写成对原始 native 卷积逐位等价。

相同 tile 下预打包可能更快，也可能更慢；额外 Half 访存、资源占用和调度会影响结果。历史逐形状证据见 [`kahan-prepack-existing-evidence.json`](../results/aggressive-20261028/kahan-prepack-existing-evidence.json)。其中更宽 tile 的 packed 组合在合成完整卷积中有正收益，但尚未形成当前 209 文件模型的完整组合验证。

另一个只改 BK32 tile、不预打包的整模型候选曾逐位通过，完整视频质量也通过，但总耗时较慢，未采用。不能据此推断 packed-Kahan 组合无效，也不能把单算子收益当成完整模型收益。后续组合实验应沿用 BK32、三个乘积、舍入边界、Kahan 累加和权重版本刷新，先做当前整模型配对，再决定是否正式跑视频。

## 后续优先级

视频主干占每步约 96%–97%。保留完整逻辑形状、只在底层分配空间块工作的版本现已完成正式运行，比上个双卡版本快 10.532237 秒，23 个视频与原始单卡基线逐文件一致；代码见 [`nccl-spatial-temporal-209`](../experiments/nccl-spatial-temporal-209)。直接输入 batch 8+8 的版本仍保留为负结果。

下一步是 decision 阶段的同类分片：先清点原始算子，保留 vendor GEMM 的完整矩阵尺寸，只拆独立行的原始 Triton kernel。原始 decision 的 277 次空间块 JIT 调用元数据已发布；尚无新 decision 加速或质量结论。Decision 的 norm/layout/half-pack、动作有效前缀 attention 和 Kahan 预打包组合也仍待验证。
