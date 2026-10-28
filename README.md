# UnifoLM 视频推理优化审阅包

NEWBIE 2.0 · team07 · NVIDIA A800 80 GB；当前 V2 使用两张卡，其余四项使用单卡。此仓库用于外部代码审阅。

当前已验证的 V2 为 **280.850431 秒 / PSNR 29.675324786 dB**，五项有效耗时合计 **829.756355 秒**。完整原始材料为 [`results/runtime.log`](results/runtime.log) 和 [`results/result.json`](results/result.json)。这是官方工具在比赛机上的本地验证结果；不表示本仓库代码已经经过主办方独立复跑。

| 分项 | process_wall_s | PSNR dB |
|---|---:|---:|
| Z1 堆箱 | 167.819341 | 40.074914 |
| 双臂堆箱 | 103.111184 | 31.991494 |
| 双臂堆箱 V2 | 280.850431 | 29.675325 |
| 清理铅笔 | 117.077862 | 36.537127 |
| G1 相机装箱 | 160.897537 | 26.107156 |

五项材料由各次已完成运行汇总后重新校验，共 49 轮、103 个视频；不是一次连续从头跑完五项的墙钟计时。V2 来自 2026-10-29 的双卡 219 源码版本；其余四项来自同日的非反馈 VAE 优化版本，来源见 `results/case-sources.json`。不能把五项汇总宣称为当前代码同一次完整复跑的实测结果。

## 审阅入口

- [`docs/REVIEWER_GUIDE.md`](docs/REVIEWER_GUIDE.md)：代码结构、现有瓶颈、需要重点检查的正确性。
- [`docs/REPRODUCE.md`](docs/REPRODUCE.md)：版本、外部资源和 V2 运行命令。
- [`problem/PARTICIPANT_HANDBOOK.md`](problem/PARTICIPANT_HANDBOOK.md)、[`problem/QUICKSTART.md`](problem/QUICKSTART.md)：比赛机共享的赛题与规则原文。
- [`unifolm/SUBMISSION.md`](unifolm/SUBMISSION.md)：提交契约；完整工作量及 PSNR 门槛。
- [`unifolm/src`](unifolm/src)：保留的单卡基线精确源码，209 个 Python 文件，110 个 Triton JIT 函数。
- [`unifolm/world-head-filter-cache-production-sources.json`](unifolm/world-head-filter-cache-production-sources.json)：生产源码 SHA256 清单。
- [`experiments/nonv2-vae-all-display`](experiments/nonv2-vae-all-display)：已采用的四个非 V2 分项精确源码与启动参数。
- [`experiments/native-attention-218/production219`](experiments/native-attention-218/production219)：当前双卡 V2 的精确增量、计时证明与 219 文件清单；使用 `reproduce/run_v2_parallel.py`。此前 218、215 和 213 文件版本均保留。
- [`docs/REVIEW_RESPONSE_20261029.md`](docs/REVIEW_RESPONSE_20261029.md)：最新审阅回复、原生 attention 分片的六对完整模型测试与正式同卡对照。
- [`docs/REVIEW_RESPONSE_20261028.md`](docs/REVIEW_RESPONSE_20261028.md)：外部审阅建议的核查结果、主干/head 新测量与 Kahan 既有证据。
- [`docs/FP16_SCALAR_EXPERIMENTS_20261028.md`](docs/FP16_SCALAR_EXPERIMENTS_20261028.md)：FP16 全局／分组／通道缩放与 INT8 残差补偿的实际输入测量；新算术仍未替换正式成绩，decision 分片随后已完成正式验证。
- [`experiments/fp16-balanced-residual-209`](experiments/fp16-balanced-residual-209)：仅缩放 INT8 残差使中间 FFN 算子约省 21.5%，完整 world 前向每次省 0.975 ms；但完整 V2 的 PSNR 仅 24.786 dB，未通过质量门槛，未采用。失败的精确代码与完整单项 log／JSON 均保留。
- [`experiments/fp16-outlier-residual-209`](experiments/fp16-outlier-residual-209)：缩放 INT8 残差加 256/5120 个 Half 通道保护，融合连续打包后，完整 V2 为 **321.720411 秒 / PSNR 25.743337 dB**，质量通过但未证明整项加速。另有 40 个微基准配置、六组整网与十层误差／求和顺序对照，均未替换正式成绩。
- [`experiments/fp16-cached-outlier-residual-209`](experiments/fp16-cached-outlier-residual-209)：按原 FP32 master 版本缓存量化权重，完整 world 前向比动态打包中位省 0.322 ms。完整 V2 为 **322.518989 秒 / PSNR 25.743337 dB**，23 个视频与动态版本相同，尚未证明整项加速，未采用。
- [`docs/NATIVE_FRAME_GRID_EXPERIMENTS_20261028.md`](docs/NATIVE_FRAME_GRID_EXPERIMENTS_20261028.md)：保留 N16 描述符，对 22 个 ResBlock 的 44 个空间卷积分帧。完整 decision 前向从约 207.5 ms 降至 198.2 ms；完整 V2 为 287.745486 秒，23 个视频与原基线及另一 rank 相同，官方 103 视频复核通过。
- [`experiments/world-native-frame-interleaved-209`](experiments/world-native-frame-interleaved-209)：world 原生 Winograd 的交错帧映射，两轮各完成 336 次卷积和 84 次矩阵逐位检查。11/14 种几何局部更快，三种 packed 几何退化；完整 12 残差块模型测试逐位通过，但每次前向慢约 2.9 ms，未采用；局部卷积更快不能代替整块净收益。
- [`experiments/kahan-packed-current209`](experiments/kahan-packed-current209)：八处 Kahan 权重 high/low 预打包，完整 world 前向中位省 1.577 ms；完整单卡 V2 为 324.052568 秒，23 个视频逐位一致，但未改善完整进程耗时，未采用。精确增量与完整 log／JSON 已保留。

源码从正式运行的不可变目录取回，逐文件核对生产清单。保留上游 [`LICENSE`](unifolm/LICENSE) 和原始说明；上游赛题入口为 <https://github.com/HKUSTGZ-SCC/Newbie2.0>。

本仓库不包含 SSH 私钥、网站登录 token、模型权重、输入缓存、参考视频或生成视频。模型和输入需要使用比赛授权的资源。随代码附带的 `.so` 是自有 CUDA Graph 句柄访问扩展，不是模型权重；其 C++ 源码与构建脚本一并提供。

## 最新优化重点

2026-10-29 已补上空间块中绕过 Triton hook 的原生 xFormers attention：每 rank 只处理自己的八帧 Q/K/V 和全局 mask 切片，保持外层 full16 形状与 GEMM 分派，复用原块末尾汇合。两阶段每次捕获各 70 个原生调用，没有新增 collective。六组动态状态逐位通过，两个阶段、两个 rank 的六对完整模型计时均更快。

正式 V2 为 **280.850431 秒**，比历史已采用 218 版本快 **6.895055 秒**。另做同 GPU1/2、同 RAM 文件系统输出的完整对照：218 基线 **286.724677 秒**，219 候选 **280.850431 秒**，本轮省 **5.874246 秒**。这是一次完整进程 AB 对照；六对交替证据来自独立模型测试。系统盘余量很小，因此本轮输出放在私有 RAM 文件系统，原扩展仍位于可执行的 home 挂载；不能将历史差值全部归因于内核。初始化、通信、模型传输、视频写入、helper 和图退出均计入原进程时间。23 个视频与历史基线及另一 rank 的 SHA256 相同；五项 103 视频官方复核通过，网站尚未上传。

[时间注意力实验](docs/TEMPORAL_ATTENTION_EXPERIMENTS.md)的补偿 Tensor Core 路线已完成完整 V2，但 PSNR 仅 22.76 dB；空间混合精度候选也未达质量要求。这两条路线未被采用。

用户提供的领先成绩为 V2 **205.67 秒**、五项 **653.75 秒**；这只是用户提供的排行榜快照，尚未独立验证。当前仍分别相差约 **75.18 秒、176.01 秒**。本次双卡 V2 的决策生成 CUDA 时间约 120.35 秒，世界模型生成约 148.68 秒。后续优先补齐 `init_attn` 的 512 通道 FFN，再扩展已有时间 FFN 汇合边界，保留完整五维 GroupNorm 和时间轴。直接缩小空间 Transformer 的 Python 输入 batch、以及缩小 decision vendor GEMM，已有变慢或数值差异记录，不能直接复用为有效优化。

新增原始单卡图计时测得：两个阶段 head 各约 8.2 ms/步，其中约 2.8 ms 为 SpatialSoftmax。将其留在主卡后，余下 head 约 5.4 ms/步；按该样本外推，跨步流水线两阶段合计的理想节省上限约 **5.79 秒**，尚未扣通信、启动和图重构成本。它不是百秒级收益的证据。完整计时方法、18 次样本和逐位一致性证明见上述审阅回复。

历史中全 FP16、宽 GEGLU 单乘积和部分扩散步降低精度均有质量失败记录，不能仅凭单步误差或算子加速判定有效。每次候选必须重新跑完整 V2 视频质量检查。
