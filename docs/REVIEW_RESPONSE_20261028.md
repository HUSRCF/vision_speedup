# 对外部审阅建议的实测核查

审阅基于 `9f88358`。其后已采用双卡 decision/world 完整空间块＋world 时间 FFN／卷积，以及四个非 V2 分项的 VAE 改进。当前正式材料的 V2 为 289.486308 秒、五项合计 838.392232 秒，已通过官方 103 视频校验；各项独立运行后汇总。下面的主干／head 和 Kahan 算子测量仍对应原始 209 文件基线。

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

相同 tile 下预打包可能更快，也可能更慢；额外 Half 访存、资源占用和调度会影响结果。历史逐形状证据见 [`kahan-prepack-existing-evidence.json`](../results/aggressive-20261028/kahan-prepack-existing-evidence.json)。其中更宽 tile 的 packed 组合随后已接入当前 209 文件模型，并完成完整 V2。

八处静态 high/low 预打包保留原 BK32、三个 Half 乘积、FP32 舍入边界和 Kahan 累加；原三指针 cubin 检查完整保留，新增四指针 ABI 经过实际图参数见证。增加的 Half 权重缓冲共 **825,753,600 字节**，随原始权重版本刷新。六组输入与全部 570 个 Linear、782 个 norm、192 个 Conv2d 参数修改／清零／恢复检查中，三个输出全部逐位一致；每个 owner 刷新五次，RNG 与原基准恢复检查通过。六对完整 world 前向均更快，中位省 **1.576950 ms**；按 550 次前向条件性外推约 **0.867 秒**，并非完整进程收益。

随后完整单卡 V2 为 **324.052568 秒 / PSNR 29.675324786 dB**，11 轮、23 个视频均完成，全部视频 SHA256 与原始单卡基线相同。但它比历史单卡 316.774396 秒慢 **7.278172 秒**，因此没有替换当前双卡正式结果。两次完整进程不是同次配对测量；本次未改动的 decision 阶段 CUDA 时间也增加约 3.254 秒，不能把全部差值归因于权重预打包。新增初始化、打包和退出成本均在原 process timer 内。

本次 GPU 2 共 565 个遥测样本，最高 **76°C**、显存峰值 **73,968 MiB**；利用率至少 90% 时 SM 频率 **1245–1410 MHz**、中位 **1395 MHz**。没有固定频率或调整功耗。精确代码与别名生成器见 [`kahan-packed-current209`](../experiments/kahan-packed-current209)，完整单项 log／JSON、参数与视频一致性证明见 [结果](../results/aggressive-20261028/kahan-packed-current209)。另一个只改 tile、不预打包的历史候选同样未被采用，两者是不同实验。

## 后续优先级

视频主干占每步约 96%–97%。当前 decision/world 双阶段完整空间块分片已完成正式运行，为 **289.486308 秒**，比此前仅 world 空间块分片的 301.930663 秒快 **12.444355 秒**，23 个视频与原始单卡基线及另一 rank 逐文件一致；代码见 [`nccl-both-spatial-temporal-209`](../experiments/nccl-both-spatial-temporal-209)。直接输入 batch 8+8 的历史版本仍保留为负结果。

Decision 分片已通过六组完整模型检查，两个 rank 的前向节省中位数为 19.006966 / 15.908356 ms。Decision 的 norm/layout/half-pack、动作有效前缀 attention、扩大卷积并行范围，以及带精度保护的量化补偿仍是后续方向。Kahan 预打包的单卡完整结论如上，尚未证明它能改善当前双卡进程时间。
