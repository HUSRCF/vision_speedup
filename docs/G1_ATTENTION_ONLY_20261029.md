# G1 原生 attention 分片：数值通过，速度未通过

当前 G1 正式成绩仍为 **160.897537 秒 / PSNR 26.107156332 dB**，五项 **771.314275 秒**。此前 G1 完整内部块63版本虽然为142.834192秒，但PSNR仅19.081776215dB，未采用。本轮检验保留原 vendor GEMM 形状的 attention-only 方案，**未更新正式双文件，也未运行完整 G1 视频候选**。

基线为 G1 正式运行的59文件源码与原G1 flags，FP32反馈头开关保持1，帧步长为官方的6。只在16个 SpatialTransformer 的原生 xFormers 调用边界，将 Q/K/V 从16帧切成各rank的8帧，并切取原全局 action mask。块内归一化、FFN、QKV及输出投影继续使用完整形状；每个空间块末尾新增一次汇合。未归属帧可在各自独立的行计算中被处理，随后丢弃；有效帧在进入跨帧计算前完整恢复。整个时间轴与5D GroupNorm保持原路径。没有新量化或精度策略。

每阶段80次原生attention调用、16次新增汇合，local attention结果复制与汇合均进入完整前向计时。它与V2在既有空间分片内补原生attention不同：G1单卡59基线没有可直接复用的汇合，因此必须支付新通信成本。

## 实测

两个阶段各检查512个空间块参数的六种动态状态：初始、输入变化、全部相关参数修改、恢复、清零、再恢复。三路输出均逐位相同，baseline恢复、条件缓存、RNG检查通过。该结论限于模型测试，不能当作完整闭环视频PSNR结论。

每阶段六对交替前向计时，每个条件7次样本、取中位数。Decision使用GPU0/1，world使用GPU2/3，两阶段在不重叠的GPU对上并行；这是模型诊断，不是独占整机的完整跑分。

| 阶段 | rank0 中位净省 ms | rank1 中位净省 ms | 较快单卡 baseline − 较慢 candidate，中位 ms |
|---|---:|---:|---:|
| decision | -0.440834 | +1.038334 | -0.440834 |
| world | +1.111549 | -0.311806 | -0.538109 |

两个阶段的原始“所有rank、所有pair都更快”门槛均未通过。不能把两个rank的净省相加。还逐pair计算了 `min(baseline_rank0, baseline_rank1) - max(candidate_rank0, candidate_rank1)`，两阶段六对均为负。这是与较快单卡前向比较的保守筛选，不是新的官方评分公式；若仅比较较慢rank的baseline，会显示约1ms的正收益，不能据此宣称双卡比原单卡更快。

完整逐pair数据见[汇总](../results/aggressive-20261028/nonv2-native-attention59/decision-world-outcome.json)，源代码与失败记录见[实验目录](../experiments/nonv2-native-attention59)。当前净收益没有支持完整G1跑分，因此未继续占卡。已确认的结构成本是重复执行的完整GEMM和新增的16次汇合、局部结果复制；本轮未分别剖析这些成本，不能给出各自占比。

## 探针失败边界

r1误沿用了Z1的帧步长4，在加载 `unitree_g1_pack_camera-fs4.pt` 前失败，未到denoiser。修正为官方步长6后生成新r2目录，未改候选算子。两次失败源码、日志和遥测独立保留。冻结 harness 的部分注释仍沿用“FFN”字样；实际导入为 `nonv2_native_attention59_mode`，计数为16个空间块／80个attention，源码哈希及捕获报告共同定义本次范围。

可运行 `python3 reproduce/verify_nonv2_native_attention59_evidence.py` 核对源码、官方G1 flags、步长、六状态逐位结果和速度拒绝结论。下一步需要减少更大范围的重复计算，例如验证保留原生Half GEMM描述符与内核、只缩小合法行网格的可行性；这尚未实现，不能算作已有收益。
