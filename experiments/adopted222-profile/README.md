# 当前222双卡图热点：2026-10-29

对当前已采用222源码分别采集decision和world，未修改数学、路由、配置或正式成绩。Decision在GPU0/1，world在GPU2/3，各自有整段占用守护和遥测。两个阶段在不同GPU对上运行，不能把跨阶段时间差解释为某一改动的收益。

| 阶段 | rank0 无 profiler 前向中位 ms | rank1 中位 ms | rank0 profiler 内核合计 ms |
|---|---:|---:|---:|
| Decision | 191.479294 | 191.394310 | 190.500200 |
| World | 229.395454 | 229.362175 | 227.707800 |

无profiler计时采用20次预热、6组各7次CUDA Event样本。每rank的6次相同输入回放，三路输出逐位一致，RNG不变。随后各采集5次完整图回放，仅导出算子名称、计数和耗时等标量汇总；不导出原始trace、指针、权重或输入。Profiler耗时受采集扰动，内核时间相加也不能代替完整process timer。

主要算子族预算（rank0，ms／前向）：

| 阶段 | 算子族 | 时间 |
|---|---|---:|
| Decision | `_zero_low_gemm`＋`inline_zero_low_gemm` | 39.100 |
| Decision | 原生FP32 CUTLASS attention | 7.922 |
| Decision | 38次NCCL all-gather | 7.066 |
| Decision | FP32时间attention | 6.034 |
| World | Winograd packed＋Kahan GEMM | 23.697 |
| World | presplit＋zero-low时间卷积 | 19.358 |
| World | 121次NCCL all-gather | 18.284 |
| World | 两类GEGLU subtile32 producer | 17.102 |
| World | 原生FP32 CUTLASS attention | 7.986 |

这些是整个阶段的算子族合计；不能将其全部视作尚可减半的工作量。World的121次汇合包含初始块，通信内核时间包含同步等待，不等于纯链路传输时间。两个rank的耗时不能相加。

源码核对发现，当前world时间块已扩展，但decision时间块仍重复计算。新的[decision时间路由实验](../decision-temporal-222)据此保留完整vendor GEMM描述符，尝试仅分片独立Triton工作，在完整5D GroupNorm之后沿空间位置划分，保留16时间步；其新增汇合必须计入实测。

精确诊断脚本、守护脚本、222文件清单和阶段计数在本目录；数值与遥测记录在 `results/aggressive-20261028/adopted222-profile`。执行 `python3 reproduce/verify_adopted222_profile.py` 可检查本快照。这是首个真实输入的诊断样本，不是整个闭环所有轮次的加权profile，也不是新正式成绩。
