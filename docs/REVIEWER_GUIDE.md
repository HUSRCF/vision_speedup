# Reviewer 阅读说明

## 执行结构

入口为 `unifolm/scripts/evaluation/world_model_interaction.py`。V2 每轮先生成动作，再模拟世界；反馈帧和状态进入下一轮。因此不能将 11 轮直接当作独立 batch 改写。DDIM 仍为 50 步，视频为 16 帧、320×512，轮数固定。

`src/unifolm_wma/modules/networks/wma_model.py` 是主干与 action/state heads，`modules/attention.py` 包含空间、时间与多条件 cross-attention，`utils/inference_runtime.py` 负责安装精度和优化策略。`utils/cuda_graph.py` 捕获与重放完整 denoiser，并保留新输入、条件和权重版本的处理。

## 当前方案

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

`results/*telemetry-phase-evidence.json` 给出当前 V2 的正式阶段时间。`results/profiles-209` 是当前 209 源码版本的 CUPTI 归因；两阶段三个 FP32 输出逐位一致，RNG 与 master versions 未改变。较早的 207 版本结果亦保留用于对比。

209 世界模型单次诊断内，时间卷积约 20.47 ms，单分支 GEGLU 约 20.12 ms，持久化 zero-low GEMM 约 19.21 ms，双分支 GEGLU 约 13.57 ms，原生 FP32 cross-attention 70 次合计约 13.93 ms。这些是带剖析开销的归因数字，不是正式成绩。

`profiles-209/decision/cross-context-audit.json` 记录决策阶段代表性的 13 路 cross-attention 结构，包括 K 长度 77、16、2、256，以及动作分支 mask。采集限制为每阶段 256 MiB，9 个超出预算的代表路径未存储。r2 世界阶段的采集器也先遇到决策阶段，因此其 operand 快照不作为世界模型证据发布；世界阶段 CUPTI 与重放检查仍有效。`experiments` 中的 r3 代码已增加明确的阶段过滤，尚无新的 r3 运行结果。

建议审查多路 cross-attention 能否共享 Q 访存与调度，同时保留文本、图像、状态、动作各自的 softmax 和动作 mask。主干 GEMM 可研究批量 vendor GEMM 或并行归约，但应保留补偿项并验证舍入误差。历史上仅将前 5/10 步 GEGLU 改为单乘积也未通过质量，相关失败不能当作待采用方案。

## 未集成实验

`results/head-scaled-half-209-report.json`：隐式 gather 卷积采用独立分块/Kahan 后数值门槛通过，但比缓存原生基线更慢，已关闭。

`results/head-prepacked-half-209-report.json`：显式 im2col、split-K GEMM、归并均进入计时。5 组真实形状中 4 组存在通过全部检查且六对计时均加速的配置，最快每组节省约 18–81 微秒。第 5 组未通过既定数值门槛。独立 cuDNN 基线含原生 filter 缓存，但其算法选择与完整模型不完全一致；目前没有实际模型集成或完整视频质量结论。

`experiments` 中的探针来自本机调度环境，部分依赖未发布的本地实验状态文件；它们用于说明方法与证据，不能直接视为生产启动入口。生产复现请使用 `reproduce/run_v2.py`。
