# 2026-10-29：优先补齐现有分片范围

对 `69d6006` 的外部审阅建议已用于调整顺序。新的完整模型证据是 world 全 12 残差块分帧在两个 rank 上均逐位通过，但每次前向慢约 2.9 ms，因此不进行该全量候选的正式 V2。正式结果保持 V2 **287.745486 秒／29.675325 dB**，五项 **836.651410 秒**；完整单项、视频质量和进程时间不能由局部算子计时替代。

当前先实现空间块内原生 xFormers attention 的局部分片。它原先绕过 Triton JIT hook；新包装器只在已有空间 scope 内，将 Q/K/V 和原全局 tensor bias 的对应八帧交给原生算子，再复制到 full16 缓冲的对应部分，复用原块末尾 all-gather。外层形状、GEMM dispatch、补偿 self-attention、KV cache、时间 attention 和全部旧缓存／CUBIN 门槛保持原逻辑。新增复制进入完整前向计时，没有额外 collective。

两个配对条件都显式安装**实际正式 218 文件树中的九个 helper**，不以较旧的无 ResBlock 基线代替当前算法。原 209 源码及正式 218 manifest 全部逐文件校验；decision 的既有 22 个 ResBlock／44 原生消费者也在两边启用。24 项 CPU 检查覆盖两 rank、四种 K 长度、无 mask／全局前缀 mask／广播 mask、完整输出形状与 RNG。

两个阶段的 GPU0/1 完整模型配对测试均已完成。每个 rank 的六组动态状态／18 个带符号零输出均逐位通过，570 Linear、782 norm、192 Conv2d master 的变化、清零及恢复检查，原始缓存／CUBIN、RNG 和基线恢复门槛全部通过。每个阶段、每个 rank 的六对计时均更快：

| 阶段 | rank 0 中位配对节省 | rank 1 中位配对节省 |
|---|---:|---:|
| Decision r2 | 4.919807 ms | 4.903427 ms |
| World r1 | 4.868607 ms | 4.843002 ms |

两个 rank 的节省不能相加。完整前向计时包含局部输出复制。独立的后置 profiler 重放确认两条件都执行同一 FP32 CUTLASS 64×64 kernel 家族，每条件 70 次原生 attention；只导出名称、次数和耗时汇总，没有原始 trace 或设备地址。这些结果支持接入正式入口，但尚不能证明完整进程加速或视频质量。

冻结的 [decision r2 源码](../experiments/native-attention-218/model-decision-r2)、[world r1 源码](../experiments/native-attention-218/model-world-r1)，以及 [decision 完整证据](../results/aggressive-20261028/native-attention-218/model-decision-r2)、[world 完整证据](../results/aggressive-20261028/native-attention-218/model-world-r1)可独立审阅。decision r1 在 CUDA 初始化前取消并迁移到空闲 GPU0/1，由 r2 替代，不属于数值失败。它们是诊断脚本，不是正式计时入口；当前尚无新提交成绩。

后续顺序是补齐 `init_attn` 的 512 通道 FFN，再研究将已有 world 时间 FFN 的汇合边界扩展到完整时间块，随后检查空间 norm/layout pack。world 卷积分帧仅在按完整块净收益筛选后再测，CO 分片和相同流水级数／实际形状的 Half chained 对照随后安排。非 V2 双卡适配需要单独处理 FP16 路径。这些尚未实施的方向不计作已取得收益，旧 profile 的预算不作为当前双卡实测。
