# Reviewer 阅读说明

## 执行结构

入口为 `unifolm/scripts/evaluation/world_model_interaction.py`。V2 每轮先生成动作，再模拟世界；反馈帧和状态进入下一轮。因此不能将 11 轮直接当作独立 batch 改写。DDIM 仍为 50 步，视频为 16 帧、320×512，轮数固定。

`src/unifolm_wma/modules/networks/wma_model.py` 是主干与 action/state heads，`modules/attention.py` 包含空间、时间与多条件 cross-attention，`utils/inference_runtime.py` 负责安装精度和优化策略。`utils/cuda_graph.py` 捕获与重放完整 denoiser，并保留新输入、条件和权重版本的处理。

## 当前方案

当前正式/default为[225文件CO分片](WORLD_CO_SPLIT_20261029.md)：V2 **257.945205秒 / PSNR29.675324786dB**，该次采用时五项 **771.314275秒**。在223上筛选9个world卷积，按输出通道分片，保留原CUBIN／constexpr／batch stride／K顺序，新增9次world汇合。请检查tile对齐、原cache检查先于替换、完整输出stride恢复、关闭恢复及权重版本刷新。完整同卡AB省2.480175秒，23视频逐文件SHA相同，新103视频官方验收通过。默认复现入口、CPU验证、精确源码和历史223材料均保留。

当前五项已进一步降至 **735.397241秒**，全部双卡；非V2中的Z1／双臂采用65文件预测，清理铅笔采用[66文件末端5块分帧＋预测](NONV2_RESBLOCK_TAIL73_20261029.md)，G1采用68文件精确权重广播。新分帧必须检查8／8帧的嵌入切片、完整16帧时间卷积／GroupNorm边界及四张图的模块清单；103视频与上一版SHA全同。以下FP32补偿细节针对V2，非V2精度按各自方案独立验证。

- 主要公开激活、master weights、偏置和残差保持 FP32；内部采用缩放 FP16 的高低分解及补偿乘积。不能将此描述成全 FP32 算术或全 FP16 推理。
- 缓存条件编码与静态权重变换，按权重 `_version` 刷新；输入激活每次重新计算。
- 采用 GroupNorm/布局转换/投影与残差等融合，以及预打包 QKV、GEGLU、时间卷积。
- 决策图的原生权重缓存为 138+43+8。世界图保留 23 个 Winograd owner，其中 15 个预打包 GEMM、8 个 Kahan；另缓存 45 个状态头原生滤波器变换。
- 世界模型非反馈帧允许批量、较低精度解码；反馈帧 14、15 保留原始 FP32 路径。

重点文件为 `native_graph_sgemm.py`、`world_head_filter_cache.py`、`native_graph_weight_cache_probe.py`、`triton_geglu_subtile32.py`、`triton_geglu_both_subtile32.py`、`triton_zero_weight_low_linear.py`、`world_persistent_zero_low_runtime.py`、`vae_protected_tf32_runtime.py`。

## 正确性重点

请检查 CUDA Graph 的指针所有权、被禁用节点恢复、权重原地修改和零化后的缓存刷新、跨 stream 生命周期、条件张量 storage 变化、Count 的动态分支、FP32 舍入边界，以及反馈帧是否保持完整计算。静态缓存不能回放上一次输入或参考视频。

官方五项均要求 pooled RGB PSNR **严格大于 25 dB**。单步相对误差很小，也可能在 11 轮闭环后失败。复核最终 MP4、所有分项证据及官方质量结果，而不仅检查张量误差。

## 剖析与改进方向

`results/aggressive-20261028/temporal-block-220/full-split-r3` 保留历史221双卡V2的阶段时间及存储包装复核记录；旧两阶段空间块版本保留在 `nccl-both-spatial-temporal-full-r1`，更早版本亦保留。`results/*telemetry-phase-evidence.json` 和 `results/profiles-209` 保留单卡 209 源码基线的计时与 CUPTI 归因；两阶段三个 FP32 输出逐位一致，RNG 与 master versions 未改变。较早的 207 版本结果亦保留用于对比。

209 世界模型单次诊断内，时间卷积约 20.47 ms，单分支 GEGLU 约 20.12 ms，持久化 zero-low GEMM 约 19.21 ms，双分支 GEGLU 约 13.57 ms，原生 FP32 cross-attention 70 次合计约 13.93 ms。这些是带剖析开销的归因数字，不是正式成绩。明确按阶段过滤的 r3 元数据现已发布到 `results/profiles-209-r3`；真实操作数不发布。

`profiles-209/decision/cross-context-audit.json` 记录决策阶段代表性的 13 路 cross-attention 结构，包括 K 长度 77、16、2、256，以及动作分支 mask。采集限制为每阶段 256 MiB，9 个超出预算的代表路径未存储。r2 世界阶段的采集器也先遇到决策阶段，因此其 operand 快照不作为世界模型证据发布；世界阶段 CUPTI 与重放检查仍有效。r3 两阶段已重新采集、逐位重放检查并完成独占复核，元数据见 `results/profiles-209-r3`。

建议审查多路 cross-attention 能否共享 Q 访存与调度，同时保留文本、图像、状态、动作各自的 softmax 和动作 mask。主干 GEMM 可研究批量 vendor GEMM 或并行归约，但应保留补偿项并验证舍入误差。历史上仅将前 5/10 步 GEGLU 改为单乘积也未通过质量，相关失败不能当作待采用方案。

## 未集成实验

最新的[时间注意力实验](TEMPORAL_ATTENTION_EXPERIMENTS.md)包括原始 FP32 launch 的逐位一致优化，以及有原生回退的补偿 Tensor Core 候选。前者配对模型计时更快，但完整 V2 为 317.385042 秒，略慢于当前最佳，因此未采用。请保留这种微测、完整模型和完整视频结果之间的区别。

`results/head-scaled-half-209-report.json`：隐式 gather 卷积采用独立分块/Kahan 后数值门槛通过，但比缓存原生基线更慢，已关闭。

`results/head-prepacked-half-209-report.json`：显式 im2col、split-K GEMM、归并均进入计时。5 组真实形状中 4 组存在通过全部检查且六对计时均加速的配置，最快每组节省约 18–81 微秒。第 5 组未通过既定数值门槛。独立 cuDNN 基线含原生 filter 缓存，但其算法选择与完整模型不完全一致；目前没有实际模型集成或完整视频质量结论。

最新的[双卡实验](AGGRESSIVE_PARALLEL_EXPERIMENTS.md)覆盖 decision/world 各 16 个完整空间 Transformer、world 的 16 个时间 FFN 与 88 个时间卷积，以及 decision 22 个 ResBlock 的 44 个原生空间卷积，完整 V2 为 287.745486 秒，23 个视频与单卡生产基线相同。请重点检查 class-level JIT hook 是否在生产 owner 安装前建立、分片指针和全局帧编号、展平 K/V pack 的 batch 切片、分组输出的回退、空间块内 FFN 通信的暂停/恢复、每块末尾只通信一次，以及原始计时和退出流程。新增原生网格应保留完整 N16 描述符、已验证的 cached filter 参数和原网格关闭恢复，帧汇合必须在时间卷积之前。模型有六组动态逐位证明，并通过完整视频及官方 103 视频复核；此前缩小 decision vendor GEMM 的失败候选仍不可直接复用。

`experiments` 中的诊断探针部分依赖未发布的本地调度状态文件。当前双卡 V2 使用 `reproduce/run_v2_parallel.py`，单卡历史基线使用 `reproduce/run_v2.py`；已采用的四个非 V2 解码优化使用 `experiments/nonv2-vae-all-display/run_case.py`，其 59 个源码文件、入口脚本和分项参数均已固定。

2026-10-29 已采用原生 xFormers attention 的八帧局部边界，当时完整 V2 为 **280.850431 秒**。应连同 [219 生产增量](../experiments/native-attention-218/production219)检查原全局 mask 的切片、full16 未归属帧在既有 gather 前不被消费、实际 FP32 CUTLASS 分派及局部输出复制成本。完整同卡／同存储对照省 5.874246 秒，103 视频重新验证通过；详见 [最新审阅回复](REVIEW_RESPONSE_20261029.md)。下一个重点是遗漏的 `init_attn` FFN，随后将已有时间 FFN 的汇合边界外移，保持全局 GroupNorm 统计和每个空间位置的完整时间序列。

初始 512 通道 FFN 现已进入 [220 版本](../experiments/initial-ffn-219/production220)，完整 V2 **277.436969 秒**；同卡完整对照净省 1.459912 秒，六对模型前向省3.940857／3.897850ms。注意初始 FFN 按整齐 GROUP32×BM64 边界切行，两个原 JIT 的 K 累加不变，旧权重版本缓存继续刷新；每 rank40MiB输出在一个 gather 处汇合。额外 M20480 编译使用私有每 rank 缓存，随后恢复原目录。原103视频复核、原计时器和完整工作量均保持。[实际 meta 构造清单](../results/aggressive-20261028/temporal-block-219/meta-r3)确认16个常规时间块使用 Linear 投影，init_attn 使用320→512→320 Conv1d投影，扩展边界时应分开处理。

历史正式V2曾采用222（268.844229秒），新增初始时间块范围共用既有汇合，不改原209文件／110 JIT算术。完整同GPU0/1、同私有tmpfs对照272.043547→268.844229秒，23视频SHA及PSNR一致，103视频官方汇总accepted。源码与证据见`experiments/initial-boundary-221/production222`及`results/aggressive-20261028/initial-boundary-221/private-tmpfs-full`。存储适配器只影响本次进程树的新NCCL临时文件；CPU跨进程、GPU图内all-gather和实际控制文件映射均有独立证据。完整差值包含运行波动，未变decision阶段也更快；不能将3.199318秒全部归因于新增范围。
