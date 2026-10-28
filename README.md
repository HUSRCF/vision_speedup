# UnifoLM 视频推理优化审阅包

NEWBIE 2.0 · team07 · NVIDIA A800 80 GB 单卡。此仓库用于外部代码审阅。

当前已验证的 V2 为 **316.774396 秒 / PSNR 29.675324786 dB**，五项有效耗时合计 **889.570845 秒**。完整原始材料为 [`results/runtime.log`](results/runtime.log) 和 [`results/result.json`](results/result.json)。这是官方工具在比赛机上的本地验证结果；不表示本仓库代码已经经过主办方独立复跑。

| 分项 | process_wall_s | PSNR dB |
|---|---:|---:|
| Z1 堆箱 | 175.139262 | 40.066380 |
| 双臂堆箱 | 109.197566 | 31.993287 |
| 双臂堆箱 V2 | 316.774396 | 29.675325 |
| 清理铅笔 | 122.080670 | 36.539782 |
| G1 相机装箱 | 166.378951 | 26.108330 |

五项材料由各次已完成运行汇总后重新校验，共 49 轮、103 个视频；不是一次连续从头跑完五项的墙钟计时。V2 来自 2026-10-28 的 209 源码版本，其他四项来自 2026-10-25 的历史有效运行，来源见 `results/case-sources.json`。不能把五项历史总分宣称为当前代码同一次完整复跑的实测结果。

## 审阅入口

- [`docs/REVIEWER_GUIDE.md`](docs/REVIEWER_GUIDE.md)：代码结构、现有瓶颈、需要重点检查的正确性。
- [`docs/REPRODUCE.md`](docs/REPRODUCE.md)：版本、外部资源和 V2 运行命令。
- [`problem/PARTICIPANT_HANDBOOK.md`](problem/PARTICIPANT_HANDBOOK.md)、[`problem/QUICKSTART.md`](problem/QUICKSTART.md)：比赛机共享的赛题与规则原文。
- [`unifolm/SUBMISSION.md`](unifolm/SUBMISSION.md)：提交契约；完整工作量及 PSNR 门槛。
- [`unifolm/src`](unifolm/src)：正式 V2 的精确源码，209 个 Python 文件，110 个 Triton JIT 函数。
- [`unifolm/world-head-filter-cache-production-sources.json`](unifolm/world-head-filter-cache-production-sources.json)：生产源码 SHA256 清单。
- [`experiments`](experiments)：后续实验代码，**尚未集成到正式成绩**。

源码从正式运行的不可变目录取回，逐文件核对生产清单。保留上游 [`LICENSE`](unifolm/LICENSE) 和原始说明；上游赛题入口为 <https://github.com/HKUSTGZ-SCC/Newbie2.0>。

本仓库不包含 SSH 私钥、网站登录 token、模型权重、输入缓存、参考视频或生成视频。模型和输入需要使用比赛授权的资源。随代码附带的 `.so` 是自有 CUDA Graph 句柄访问扩展，不是模型权重；其 C++ 源码与构建脚本一并提供。

## 最新优化重点

2026-10-28 的[时间注意力实验](docs/TEMPORAL_ATTENTION_EXPERIMENTS.md)已补充原始证据。逐位一致的 warp 调整在完整 V2 中未超过当前最佳，因此正式代码和成绩保持不变；补偿 Tensor Core 路线仍在做完整视频验证。

用户提供的领先成绩为 V2 **205.67 秒**、五项 **653.75 秒**；这只是用户提供的排行榜快照，尚未独立验证。当前 V2 的决策生成约 134.88 秒，世界模型生成约 172.91 秒。重心转向主干补偿 GEMM、多路 cross-attention 的批处理与融合，而非仅累加微秒级卷积优化。

历史中全 FP16、宽 GEGLU 单乘积和部分扩散步降低精度均有质量失败记录，不能仅凭单步误差或算子加速判定有效。每次候选必须重新跑完整 V2 视频质量检查。
