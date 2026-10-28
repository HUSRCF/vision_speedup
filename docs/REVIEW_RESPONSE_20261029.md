# 2026-10-29：优先补齐现有分片范围

对 `69d6006` 的外部审阅建议已用于调整顺序。新的完整模型证据是 world 全 12 残差块分帧在两个 rank 上均逐位通过，但每次前向慢约 2.9 ms，因此不进行该全量候选的正式 V2。审阅起点为 V2 **287.745486 秒／29.675325 dB**，五项 **836.651410 秒**；完整单项、视频质量和进程时间不能由局部算子计时替代。

当前先实现空间块内原生 xFormers attention 的局部分片。它原先绕过 Triton JIT hook；新包装器只在已有空间 scope 内，将 Q/K/V 和原全局 tensor bias 的对应八帧交给原生算子，再复制到 full16 缓冲的对应部分，复用原块末尾 all-gather。外层形状、GEMM dispatch、补偿 self-attention、KV cache、时间 attention 和全部旧缓存／CUBIN 门槛保持原逻辑。新增复制进入完整前向计时，没有额外 collective。

两个配对条件都显式安装**实际正式 218 文件树中的九个 helper**，不以较旧的无 ResBlock 基线代替当前算法。原 209 源码及正式 218 manifest 全部逐文件校验；decision 的既有 22 个 ResBlock／44 原生消费者也在两边启用。24 项 CPU 检查覆盖两 rank、四种 K 长度、无 mask／全局前缀 mask／广播 mask、完整输出形状与 RNG。

两个阶段的 GPU0/1 完整模型配对测试均已完成。每个 rank 的六组动态状态／18 个带符号零输出均逐位通过，570 Linear、782 norm、192 Conv2d master 的变化、清零及恢复检查，原始缓存／CUBIN、RNG 和基线恢复门槛全部通过。每个阶段、每个 rank 的六对计时均更快：

| 阶段 | rank 0 中位配对节省 | rank 1 中位配对节省 |
|---|---:|---:|
| Decision r2 | 4.919807 ms | 4.903427 ms |
| World r1 | 4.868607 ms | 4.843002 ms |

两个 rank 的节省不能相加。完整前向计时包含局部输出复制。独立的后置 profiler 重放确认两条件都执行同一 FP32 CUTLASS 64×64 kernel 家族，每条件 70 次原生 attention；只导出名称、次数和耗时汇总，没有原始 trace 或设备地址。这些结果支持接入正式入口，但尚不能证明完整进程加速或视频质量。

冻结的 [decision r2 源码](../experiments/native-attention-218/model-decision-r2)、[world r1 源码](../experiments/native-attention-218/model-world-r1)，以及 [decision 完整证据](../results/aggressive-20261028/native-attention-218/model-decision-r2)、[world 完整证据](../results/aggressive-20261028/native-attention-218/model-world-r1)可独立审阅。decision r1 在 CUDA 初始化前取消并迁移到空闲 GPU0/1，由 r2 替代，不属于数值失败。它们是诊断脚本；正式入口与最新结果见下一段。

`init_attn` 的 512 通道 FFN 已完成正式验证，接下来研究将已有 world 时间 FFN 的汇合边界扩展到完整时间块，随后检查空间 norm/layout pack。world 卷积分帧仅在按完整块净收益筛选后再测，CO 分片和相同流水级数／实际形状的 Half chained 对照随后安排。非 V2 双卡适配需要单独处理 FP16 路径。这些尚未实施的方向不计作已取得收益，旧 profile 的预算不作为当前双卡实测。

## 正式入口与完整验证

[219 文件生产增量](../experiments/native-attention-218/production219)保留原 209 文件、110 个 JIT 函数和四个计时接点，仅新增 attention helper、扩展 runtime 路由及覆盖断言。`local_native` 和 `attention` 两函数的 AST 与已验证模型脚本完全一致。每阶段捕获 70 个原生调用，原 spatial／FFN／temporal／ResBlock 覆盖数保持一致。

同一 GPU1/2、同一私有 RAM 文件系统输出的完整 11 轮对照：

| 条件 | 完整进程耗时 | PSNR | 23 视频 SHA256 |
|---|---:|---:|---|
| 当前 218 基线 | 286.724677 s | 29.675325 dB | 与历史有效基线一致 |
| 新 219 候选 | 280.850431 s | 29.675325 dB | 与基线及另一 rank 一致 |

本轮完整进程省 **5.874246 秒**；相对历史 287.745486 秒省 6.895055 秒，两种比较不可混用。完整进程只有一对 AB，六对交替测试是模型级证据。决策／世界生成 CUDA 时间分别为 120.354629／148.676148 秒。Home 仅剩约 15 MiB，因此两个条件均将源码和输出放在私有 RAM 文件系统；native `.so` 链接至原可执行 home 挂载。保留原 warm compiler cache，两条件均禁用 Python bytecode 写入。视频写入、全部初始化、通信、捕获及退出仍在原计时内。

守护程序全程未发现外来 context。两卡峰值显存均 74,170 MiB，最高温度 74／73°C；繁忙频率中位均 1410 MHz，范围分别 1335–1410／1260–1410 MHz。未改频率或结束他人作业。原 GPU0/1 排队在 CUDA 前取消后迁移到空闲 GPU1/2，旧排队证据未用作成绩。

[完整对照证据](../results/aggressive-20261028/native-attention-218/full-candidate-r2)包含原 log／JSON、阶段计时、控制器覆盖、视频哈希比较、整段 telemetry 和源码证明。重新校验全部五项 **49 轮／103 视频**通过，五项合计 **829.756355 秒**；是各次有效单项汇总，不是一次连续五项墙钟。公开 `results/runtime.log`、`results/result.json` 与本机 `GPU/` 同字节，尚未上传网站。

## 第二项：init_attn 的 512 通道 FFN

[模型 r2](../experiments/initial-ffn-219/model-r2)已将原 producer／consumer 一起按行分片，保持 full40960 外层形状、BK32、group 对齐、原权重版本缓存和全部 JIT 函数体；整个 FFN 最后汇合一次，每 rank 发送 40 MiB。以实际已采用 219 文件为两条件共同基线，六组动态 master／输入状态及每 rank18个带符号零输出均逐位通过；570 Linear／782 norm／192 Conv2d、RNG、基线恢复和原23+45 owner／CUBIN门槛保留。两个 rank 的六对完整 world 前向均更快，中位净节省 **3.940857／3.897850 ms**，含汇合，不能将两 rank 相加。

首轮在图构建后遇到测试脚本读取旧 `captured` 字段的 KeyError；r2 仅修正覆盖断言，算子代码 SHA256 未变。首轮没有数值／速度结论。新增 M20480 变体使用独立的每 rank 标准 Triton 缓存，立即恢复旧缓存目录；不替换 cache manager、不删除旧缓存。

[220 文件版本](../experiments/initial-ffn-219/production220)已完成正式验证：同一 GPU0/3、同一 RAM 输出存储，219 基线 **278.896881 秒**，220 候选 **277.436969 秒**，完整进程净省 **1.459912 秒**。World 生成 CUDA 时间为148.341296→146.375799秒，减少1.965497秒；模型传输为2.948446→3.506492秒，增加0.558046秒。相对历史219正式280.850431秒减少3.413462秒，包含跨轮波动，不能全部归因于新增 FFN 分片。

23个视频与已采用219及另一rank的SHA256相同，PSNR仍为29.675324786dB。每个阶段550次调用、decision/world各16个空间块、16个原world时间FFN、88个时间卷积、decision22残差块／44原生消费者和每阶段70个原生attention捕获均核对；新增world初始FFN恰好捕获1次。峰值显存两卡均74,172MiB，最高温度71／68°C，繁忙频率范围均1335–1410MHz、中位1410MHz；整段无外来context。

重新使用官方工具验证49轮／103视频通过，五项汇总 **826.342893 秒**；本机GPU与公开results双文件同字节。验证集合复用了此前已验证MP4的普通文件硬链接，逐文件校验SHA且原生成输出保持不变，全部103视频仍重新解码并与官方参考计算PSNR；这只发生在分项进程计时完成后的材料汇总中，不作为推理缓存。保留未改动的原log／JSON和完整源码证明，见[正式对照](../results/aggressive-20261028/initial-ffn-219/full-candidate-r1)。

下一项扩展时间块分片边界。[实际配置的meta构造清单](../results/aggressive-20261028/temporal-block-219/meta-r3)确认17个时间模块，其中16个常规块使用Linear投影，init_attn的320→512→320投影使用Conv1d。该清单未加载checkpoint、输入或GPU，不是性能证据。
