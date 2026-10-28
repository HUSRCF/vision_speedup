# UnifoLM 视频推理优化审阅包

NEWBIE 2.0 · team07 · NVIDIA A800 80 GB；当前 V2 使用两张卡，其余四项使用单卡。此仓库用于外部代码审阅。

当前已验证的 V2 为 **312.462900 秒 / PSNR 29.675324786 dB**，五项有效耗时合计 **867.455206 秒**。完整原始材料为 [`results/runtime.log`](results/runtime.log) 和 [`results/result.json`](results/result.json)。这是官方工具在比赛机上的本地验证结果；不表示本仓库代码已经经过主办方独立复跑。

| 分项 | process_wall_s | PSNR dB |
|---|---:|---:|
| Z1 堆箱 | 167.819341 | 40.074914 |
| 双臂堆箱 | 109.197566 | 31.993287 |
| 双臂堆箱 V2 | 312.462900 | 29.675325 |
| 清理铅笔 | 117.077862 | 36.537127 |
| G1 相机装箱 | 160.897537 | 26.107156 |

五项材料由各次已完成运行汇总后重新校验，共 49 轮、103 个视频；不是一次连续从头跑完五项的墙钟计时。V2 来自 2026-10-28 的双卡 213 源码版本；Z1、铅笔、G1 来自同日的非反馈 VAE 优化版本；双臂堆箱保留 2026-10-25 的历史有效运行，来源见 `results/case-sources.json`。不能把五项历史总分宣称为当前代码同一次完整复跑的实测结果。

## 审阅入口

- [`docs/REVIEWER_GUIDE.md`](docs/REVIEWER_GUIDE.md)：代码结构、现有瓶颈、需要重点检查的正确性。
- [`docs/REPRODUCE.md`](docs/REPRODUCE.md)：版本、外部资源和 V2 运行命令。
- [`problem/PARTICIPANT_HANDBOOK.md`](problem/PARTICIPANT_HANDBOOK.md)、[`problem/QUICKSTART.md`](problem/QUICKSTART.md)：比赛机共享的赛题与规则原文。
- [`unifolm/SUBMISSION.md`](unifolm/SUBMISSION.md)：提交契约；完整工作量及 PSNR 门槛。
- [`unifolm/src`](unifolm/src)：保留的单卡基线精确源码，209 个 Python 文件，110 个 Triton JIT 函数。
- [`unifolm/world-head-filter-cache-production-sources.json`](unifolm/world-head-filter-cache-production-sources.json)：生产源码 SHA256 清单。
- [`experiments/nonv2-vae-all-display`](experiments/nonv2-vae-all-display)：已采用的 Z1、铅笔、G1 精确源码与分项启动参数。
- [`experiments/nccl-ffn-209/timed-entry`](experiments/nccl-ffn-209/timed-entry)：已采用双卡 V2 的精确增量、计时证明与 213 文件清单；使用 `reproduce/run_v2_parallel.py`。

源码从正式运行的不可变目录取回，逐文件核对生产清单。保留上游 [`LICENSE`](unifolm/LICENSE) 和原始说明；上游赛题入口为 <https://github.com/HKUSTGZ-SCC/Newbie2.0>。

本仓库不包含 SSH 私钥、网站登录 token、模型权重、输入缓存、参考视频或生成视频。模型和输入需要使用比赛授权的资源。随代码附带的 `.so` 是自有 CUDA Graph 句柄访问扩展，不是模型权重；其 C++ 源码与构建脚本一并提供。

## 最新优化重点

2026-10-28 的[双卡实验](docs/AGGRESSIVE_PARALLEL_EXPERIMENTS.md)覆盖 32 个 world FFN 与 88 个时间卷积。完整 11 轮为 **312.462900 秒**，比单卡基线 316.774396 秒快 **4.311496 秒**；23 个视频与基线及另一 rank 全部逐文件相同。正式计时包含初始化、通信、双模型传输和退出；微测的 22.7–25.5 ms 前向收益没有等比例转化为总耗时下降。

[时间注意力实验](docs/TEMPORAL_ATTENTION_EXPERIMENTS.md)的补偿 Tensor Core 路线已完成完整 V2，但 PSNR 仅 22.76 dB；空间混合精度候选也未达质量要求。这两条路线未被采用。

用户提供的领先成绩为 V2 **205.67 秒**、五项 **653.75 秒**；这只是用户提供的排行榜快照，尚未独立验证。当前双卡 V2 的决策生成约 137.97 秒，世界模型生成约 161.96 秒。后续重点是扩大有效并行覆盖，同时保留已有快速算子的形状和调度。直接将完整空间 Transformer 改成半批输入反而变慢，未采用。

历史中全 FP16、宽 GEGLU 单乘积和部分扩散步降低精度均有质量失败记录，不能仅凭单步误差或算子加速判定有效。每次候选必须重新跑完整 V2 视频质量检查。
