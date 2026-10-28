# UnifoLM 视频推理优化审阅包

NEWBIE 2.0 · team07 · NVIDIA A800 80 GB；当前 V2 使用两张卡，其余四项使用单卡。此仓库用于外部代码审阅。

当前已验证的 V2 为 **268.844229 秒 / PSNR 29.675324786 dB**，五项有效耗时合计 **816.711804 秒**。完整原始材料为 [`results/runtime.log`](results/runtime.log) 和 [`results/result.json`](results/result.json)。这是官方工具在比赛机上的本地验证结果；不表示本仓库代码已经经过主办方独立复跑。

| 分项 | process_wall_s | PSNR dB |
|---|---:|---:|
| Z1 堆箱 | 166.780992 | 40.074914 |
| 双臂堆箱 | 103.111184 | 31.991494 |
| 双臂堆箱 V2 | 268.844229 | 29.675325 |
| 清理铅笔 | 117.077862 | 36.537127 |
| G1 相机装箱 | 160.897537 | 26.107156 |

五项材料由各次已完成运行汇总后重新校验，共 49 轮、103 个视频；不是一次连续从头跑完五项的墙钟计时。V2 来自 2026-10-29 的双卡 222 源码版本；Z1来自同日的单卡60文件Half projection／GEGLU融合版本，其余三项来自非反馈VAE优化版本，来源见 `results/case-sources.json`。不能把五项汇总宣称为当前代码同一次完整复跑的实测结果。

## 审阅入口

- [`docs/REVIEWER_GUIDE.md`](docs/REVIEWER_GUIDE.md)：代码结构、现有瓶颈、需要重点检查的正确性。
- [`docs/REPRODUCE.md`](docs/REPRODUCE.md)：版本、外部资源和 V2 运行命令。
- [`problem/PARTICIPANT_HANDBOOK.md`](problem/PARTICIPANT_HANDBOOK.md)、[`problem/QUICKSTART.md`](problem/QUICKSTART.md)：比赛机共享的赛题与规则原文。
- [`unifolm/SUBMISSION.md`](unifolm/SUBMISSION.md)：提交契约；完整工作量及 PSNR 门槛。
- [`unifolm/src`](unifolm/src)：保留的单卡基线精确源码，209 个 Python 文件，110 个 Triton JIT 函数。
- [`unifolm/world-head-filter-cache-production-sources.json`](unifolm/world-head-filter-cache-production-sources.json)：生产源码 SHA256 清单。
- [`experiments/nonv2-vae-all-display`](experiments/nonv2-vae-all-display)：其余三个非 V2 分项及历史 Z1 的59文件精确源码与启动参数；当前 Z1 默认转到下述60文件版本。
- [`experiments/nonv2-fused-half-geglu`](experiments/nonv2-fused-half-geglu)：当前 Z1 的60文件增量、完整复现入口与正式103视频验收证据。
- [`experiments/temporal-block-220/production221`](experiments/temporal-block-220/production221)：历史双卡221的精确增量、计时证明与文件清单。此前220、219、218、215和213文件版本均保留。
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

[NCCL通信调优](experiments/nccl-tuning222)已完成13个独立进程配置的逐位检查。采用实际222汇合尺寸的通信替代图，32channels＋P2Pwrite合计约24.08ms，默认约29.27ms；该图按尺寸分组，不包含模型计算/实际调度，不能当作完整提速。复用同一222源码、只改三项NCCL环境参数的完整V2正在验证，正式材料仍为下述268.844229秒。

当前正式/default已采用[初始时间块222](experiments/initial-boundary-221/production222)：将既有初始FFN汇合移到残差之后，覆盖前方QKV、FP32时间attention与残差norm，包含2个QKV回写节点，不增加collective，保留完整5D GroupNorm、16时间步、Conv1d边界投影与原始209文件／110个JIT算术。此前六状态逐位和六对模型计时通过；新的同GPU0/1、同私有tmpfs完整对照为 **272.043547 → 268.844229秒**，本次少 **3.199318秒**。这是一次完整进程AB，模型六对计时另有证据。本次world生成CUDA少1.388971秒，未改动的decision也少0.907554秒，因此不把全部进程差值归因于新路由。相对历史221正式成绩少1.980970秒。23视频与221及另一rank的SHA一致、PSNR29.675324786dB；新的103视频官方汇总accepted，五项 **816.711804秒**，本机GPU双文件已更新。

本轮通过[私有tmpfs适配器](experiments/private-nccl-tmpfs)解除NCCL控制缓冲空间限制：仅本次进程树的新建NCCL临时文件放到私有/var/tmp，CPU跨进程mmap与双卡图内all-gather均逐位通过；两rank的实际控制文件映射设备也已核实。两个完整对照使用相同适配器、guard、缓存及输出挂载。没有清理或迁移原历史证据/缓存，安装的NCCL库和GPU内核保持不变。完整源码、逐项log/JSON及同配置对照见[private-tmpfs-full](results/aggressive-20261028/initial-boundary-221/private-tmpfs-full)。

此前221时间块扩展已完成正式验证：[221源码](experiments/temporal-block-220/production221)、[完整证据](results/aggressive-20261028/temporal-block-220/full-split-r3)。16个常规world时间块沿空间位置分片，保留完整5D GroupNorm、16时间步及既有FFN汇合；QKV、FP32时间attention、投影及残差norm共享该边界，无新增通信，包含54个QKV回写节点。此前六状态逐位检查及六对交替模型计时均通过，两卡world前向中位省11.475960／11.577858ms。完整11轮为 **270.825199秒**，比历史220的277.436969秒少 **6.611770秒**；完整进程比较不是同期AB配对。23个视频与220及另一rank一致，PSNR29.675324786dB，官方103视频复核通过，当时切换221，当前由222继承。

本次推理使用私有/tmp和RAM分盘输出，未清理历史证据。原视频根目录的符号链接被官方校验器拒绝，原失败log/JSON保留。随后将同次rank1视频逐文件核对rank0及220的SHA，硬链接为普通目录，再执行官方视频验证与五项汇总；没有重跑GPU或改写进程耗时。最终可提交材料只有仓库results下的accepted双文件。分盘守护脚本和包装证明见[full-split-r3](experiments/temporal-block-220/full-split-r3)。

[完整尝试的存储失败记录](results/aggressive-20261028/temporal-block-220/full-failed-r2/summary.json)与模型检查分开保留：该次未计分；两卡峰值显存74,192MiB、最高73／72°C。[空间norm的world模型验证](experiments/spatial-norm-221/world-model-r2)现已通过六状态逐位检查与六对交替计时：38个启动分片，两卡每次前向中位省 **0.488449／0.484352 ms**，约0.27秒／550次，属于小增量；尚无该增量的完整V2成绩。此前72项CPU布局和16项时间GroupNorm拒绝检查亦保留。

[初始时间块边界扩展](experiments/initial-boundary-221/world-model-r3)在同一221基线上也通过六状态逐位检查与六对交替计时：将已有初始FFN汇合移到残差完成之后，覆盖其前方QKV、FP32时间attention及残差norm，不增加collective，保留完整5D GroupNorm与Conv1d投影。两卡world前向中位省 **2.127869／2.097145 ms**，六对均更快，含两次QKV回写；550次约1.15–1.17秒，仅为模型外推。此结果与空间norm尚未组合验证，也没有替换正式221成绩。

该边界现已集成为[冻结222候选](experiments/initial-boundary-221/production222)，源码／计时接点和CPU检查核对通过；首次完整尝试因RAM盘耗尽中止，当时没有222的完整耗时或PSNR。并行启动的第二个NCCL实验在申请5,767,520字节共享控制缓冲时失败，完整任务的遥测写入也随之ENOSPC；[失败证据](results/aggressive-20261028/initial-boundary-221/full-storage-failed-r1)单独保留。修正后的未来guard在CUDA分配前要求RAM盘24MiB、/tmp11MiB，并将官方validator直接指向普通/tmp目录。小型双卡诊断也要预留14MiB RAM。NCCL日志已显示P2P/CUMEM/read连接，不能因共享控制缓冲失败而推断数据绕行主机内存；关闭SHM传输的诊断同样未完成初始化，无带宽结果。该次失败后已完成上述私有tmpfs复核，正式/default现为222。

[非V2 Half projection／GEGLU融合](experiments/nonv2-fused-half-geglu)已完成单卡整网与完整Z1对照：只启用M40960／C320，保留Half舍入边界，10处producer／阶段，原权重直接读取。Decision／world六状态三路输出均逐位一致，六对完整前向分别省0.729084／0.690689ms。同GPU2、同/tmp输出的完整进程为 **171.304477 → 170.633517秒**，省 **0.670960秒**；25视频SHA完全一致、PSNR40.074914dB。640／1280通道已测配置更慢，未启用。GPU0完整复核为 **166.780992秒**，比上一正式Z1少 **1.038349秒**；该差值包含跨历史运行波动，因果收益另见同GPU2对照。与上一正式25视频SHA相同，新的103视频官方汇总accepted，Z1已采用60文件版本，当时五项合计818.692774秒；随后222更新为816.711804秒。源码60文件增量、模型测试、完整log／JSON和微基准失败边界均公开。

[Z1 全帧 Half VAE 候选](experiments/nonv2-all-feedback-vae)已完成完整12轮／25视频：**167.555672秒，PSNR29.290234dB**，独立GPU运行和官方单项检查通过。原来两张FP32反馈帧也改成Half，world合成一次16帧解码；原58个其余源码、FP32 master、DDIM及计时入口保持不变。实际TF32配置的六对解码微基准均更快，中位省39.146ms。相比历史已采用Z1的167.819341秒，仅少0.263669秒，且PSNR从40.074914降低；没有同期完整进程AB或官方103视频新汇总，因此**未替换正式材料**。关闭TF32的97.003ms微基准不代表生产收益。首次尝试在推理前因精简源码视图的完整git diff写满RAM中止；重试仅限制诊断补丁范围，完整执行源码仍归档并核对SHA，失败记录单独保留。

[非V2的FP16 FFN双卡模型实验](experiments/nonv2-ffn-59)覆盖33个FFN，保留其余计算：decision六对均更快，两卡中位省 **3.876354／5.427204 ms**；world为 **5.751297／4.246525 ms**。权重修改／清零／恢复和RNG检查通过，但输出存在Half舍入差异，最大相对L2约0.0525%。这只是模型误差粗筛，尚无完整Z1视频PSNR或正式加速成绩。

首版把`init_attn`一并纳入后未通过原图finite校验；其原始FFN输出先汇合、随后才加残差，不能套用普通块边界。r2保留`init_attn`已采用220路径。模型检查完成后，外层复制重复日志曾因RAM盘满失败；完整GPU报告和原始日志已取回并核对SHA，再压缩仅已结束任务的重复证据。该归档失败单独记录，不掩饰为成功退出，也未改动GPU测量或正式材料。


此前220版本补齐 world `init_attn` 的 512 通道 FFN：原 producer／consumer 一起按行分片，保持 BK32、full40960 外层形状和权重版本缓存，结束时汇合一次，每 rank 40 MiB。完整模型六对计时均更快，中位净省 3.940857／3.897850 ms，六组动态状态逐位一致。正式同 GPU0/3、同 RAM 输出对照为 **278.896881 → 277.436969 秒**，整项省 **1.459912 秒**。World 生成 CUDA 时间减少 1.965497 秒，模型传输增加约 0.558 秒，说明不能把前向收益直接当作进程收益。相对上一正式记录少 3.413462 秒，其中包含跨轮波动。23 视频与原基线及另一 rank 相同，103 视频官方复核通过；[完整证据](results/aggressive-20261028/initial-ffn-219/full-candidate-r1)。

2026-10-29 已补上空间块中绕过 Triton hook 的原生 xFormers attention：每 rank 只处理自己的八帧 Q/K/V 和全局 mask 切片，保持外层 full16 形状与 GEMM 分派，复用原块末尾汇合。两阶段每次捕获各 70 个原生调用，没有新增 collective。六组动态状态逐位通过，两个阶段、两个 rank 的六对完整模型计时均更快。

上一轮 219 正式 V2 为 **280.850431 秒**，比历史已采用 218 版本快 **6.895055 秒**。另做同 GPU1/2、同 RAM 文件系统输出的完整对照：218 基线 **286.724677 秒**，219 候选 **280.850431 秒**，本轮省 **5.874246 秒**。这是一次完整进程 AB 对照；六对交替证据来自独立模型测试。系统盘余量很小，因此本轮输出放在私有 RAM 文件系统，原扩展仍位于可执行的 home 挂载；不能将历史差值全部归因于内核。初始化、通信、模型传输、视频写入、helper 和图退出均计入原进程时间。23 个视频与历史基线及另一 rank 的 SHA256 相同；五项 103 视频官方复核通过，网站尚未上传。

[时间注意力实验](docs/TEMPORAL_ATTENTION_EXPERIMENTS.md)的补偿 Tensor Core 路线已完成完整 V2，但 PSNR 仅 22.76 dB；空间混合精度候选也未达质量要求。这两条路线未被采用。

用户提供的领先成绩为 V2 **205.67 秒**、五项 **653.75 秒**；这只是用户提供的排行榜快照，尚未独立验证。当前仍分别相差约 **63.17 秒、162.96 秒**。此前221的决策生成CUDA时间约119.87秒，世界模型生成约139.73秒。`init_attn` 的 512 通道 FFN 已补齐；常规world时间块已扩展；初始时间块新边界已验证；下一步验证非V2分片的完整视频质量，并实测双卡通信参数，继续保留完整五维GroupNorm和时间轴。直接缩小空间 Transformer 的 Python 输入 batch、以及缩小 decision vendor GEMM，已有变慢或数值差异记录，不能直接复用为有效优化。

新增原始单卡图计时测得：两个阶段 head 各约 8.2 ms/步，其中约 2.8 ms 为 SpatialSoftmax。将其留在主卡后，余下 head 约 5.4 ms/步；按该样本外推，跨步流水线两阶段合计的理想节省上限约 **5.79 秒**，尚未扣通信、启动和图重构成本。它不是百秒级收益的证据。完整计时方法、18 次样本和逐位一致性证明见上述审阅回复。

历史中全 FP16、宽 GEGLU 单乘积和部分扩散步降低精度均有质量失败记录，不能仅凭单步误差或算子加速判定有效。每次候选必须重新跑完整 V2 视频质量检查。
