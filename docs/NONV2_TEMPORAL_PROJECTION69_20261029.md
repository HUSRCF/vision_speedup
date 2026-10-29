# 清理铅笔：把已有汇合移到时间输出投影之后

当前清理铅笔正式版本已经把16个空间内部块、17个时间内部块分配到两张GPU，每次前向33次汇合。时间块输出先汇合，再在两张卡上各自执行完整输出投影。

本实验保留16个普通时间Transformer的GroupNorm、输入投影和完整16帧时间序列，只把原有汇合推迟至输出Linear之后。每张卡对自己的空间位置执行原输出投影，然后汇合；初始时间块的Conv1d路径保持原样。保留原33次通信，不增加一次collective。原有三个数学分片辅助模块与已发布版本逐字节一致；新增代码只使用普通Torch模块和张量，没有新原生CUDA参数扫描。

对照双方都启用原63文件对应的完整双卡内部块分片。`mode(False)`代表原63双卡路径，`mode(True)`在此基础上推迟输出投影汇合；原59文件源码是两者共同的未改动模型代码，不能把对照写成原单卡59。诊断沿用旧日志标记，但报告明确记录`baseline_is_original63_nonv2_FP16`。

Decision、world各自准备6组输入／权重动态状态与6对交替完整前向计时，动态权重范围包括新增覆盖的输出投影参数。数值粗检查通过也不能推导完整闭环视频PSNR。每组双图应捕获66条原分片记录，其中候选16次输出投影在原汇合之前。

两个阶段已在比赛机禁用CUDA，通过公开入口逐文件重建原59代码、所有诊断脚本和入口资源，并运行四项CPU检查。新检查覆盖两种dtype、两个宽度、两rank及权重变化，共16个情况；不支持的batch／时间长度被拒绝。

GPU0／1实测两个阶段均通过788个参数的六组动态状态检查，RNG不变、恢复原权重后baseline逐位还原。候选与baseline有少量数值差异，因此继续独立验证完整视频。每rank的六对完整前向均更快；按每对最快baseline减最慢candidate保守计算，decision／world中位分别省0.289280／0.274944ms，12对均为正。按本项每阶段400次调用外推约0.225690秒，只是条件性预算。

已集成为64源码文件版本：只新增本模块，原63文件中只改运行控制器的导入和新增覆盖断言；计时入口、三个数学分片模块、全部50步及8轮不变。完整候选在GPU1／3为 **105.976138秒／PSNR36.122616450dB**，17个视频与另一rank一致，每阶段400次调用、33条捕获记录，两图共32次投影后汇合核对通过。

随后在同GPU1／3重跑原63：**106.348409秒／34.881926888dB**，17个视频与原正式SHA一致。候选该对完整进程少 **0.372271秒**；只有一组完整对照，不能当作稳定收益分布。此前GPU0／1的106.613350秒基线另存，未混入配对比较。候选仍略高于历史正式105.949243秒，未采用；完整结果没有替换正式材料。

候选169次遥测中，两卡峰值显存均29234MiB、最高温71／58°C；利用率至少50%的样本中SM频率范围1245–1410／1290–1410MHz。保留原始遥测，不推定频率是耗时差异原因。

```bash
python reproduce/verify_nonv2_temporal_projection69.py
python reproduce/run_nonv2_temporal_projection69.py --stage decision --gpus 0,1 --workdir /path/on/private-tmpfs/decision --storage-config /path/to/qualified-storage.json --cache-dir /path/to/qualified-FFN-cache
```

把`--stage`改为`world`测另一阶段。公开资料位于[`experiments/nonv2-temporal-projection69`](../experiments/nonv2-temporal-projection69)，CPU证据位于[`results/aggressive-20261029/nonv2-temporal-projection69`](../results/aggressive-20261029/nonv2-temporal-projection69)。正式五项仍为755.093960秒，`GPU`双文件未替换。

完整候选入口：

```bash
python reproduce/run_nonv2_projection64.py --gpus 1,3 --workdir /path/on/private-tmpfs/candidate --storage-config /path/to/qualified-storage.json --cache-dir /path/to/qualified-FFN-cache
python reproduce/run_nonv2_parallel.py --case pencils --gpus 1,3 --workdir /path/on/private-tmpfs/baseline --storage-config /path/to/qualified-storage.json --cache-dir /path/to/qualified-FFN-cache
```

64文件公开入口也已关闭CUDA逐文件重建并运行打包后的CPU检查，哈希与实际计时源码和资源一致。GPU0被其他任务占用后，原等待进程在入场前结束；同一候选改到GPU1／3，未修改源码或守护。原计划元数据另存，并记录了基线标签修正，避免把初始GPU0／1基线误标为实际同卡组对照。
