# 非V2：双卡与时间步复用组合候选

当前正式五项已经全部使用两张GPU，合计765.731042秒；本实验没有修改GPU下的正式log／JSON。

新候选在Z1正式63文件数学路径上接入时间步缓存，形成65文件版本。复用策略沿用单卡已完成12轮／25视频、PSNR30.254356344dB的7块方案：`input_blocks[9:]`、middle、`output_blocks[:3]`，两阶段位置11、13、…、39读取缓存，其余35步刷新。全部轮数和每阶段50个DDIM更新保留。**单卡的质量结果不代表组合双卡后通过。**

与原63相比，三个分片数学helper逐字保持一致。只改并行runtime的接线／验证和CUDA Graph的可选共享池，增加两个已冻结复用模块。复用runtime仍调用并行执行器，不能绕过其分片上下文和阶段计数。

每张图分别核对捕获的普通Torch模块名，要求名单无重复、顺序归一后严格一致：

| 图角色 | 汇合节点 |
|---|---:|
| decision完整 | 33 |
| decision复用 | 31 |
| world完整 | 33 |
| world复用 | 31 |

7块中只有middle包含一对空间／时间Transformer，因此只减少2个汇合。合计四张图128个捕获记录；每阶段仍执行600次，其中180次使用复用图。完整结束前先写出时间步／失效／图角色证据，再重置四张图和关闭NCCL。仍保留原输出head、完整时间归一化、16帧时间序列、全局action mask及完整输出视频检查。

CPU验证已检查1200次调用均经过并行执行器、24段完整50步序列、每阶段180次复用、浅层与head持续执行、RNG不变、四种范围的模块名单和重复名拒绝。公开入口在比赛机禁用CUDA重建65文件与全部计时资源，SHA匹配并再次通过CPU接线验证。

完整双卡任务已在GPU1／2排队，守护程序要求两卡同时空闲，保留原双文件锁、全程外来上下文检查、16次失败尝试上限及频率／温度遥测。目前没有完整GPU结果，不宣称提速或质量通过，也没有加入新的103视频汇总。

- [冻结源码、原guard和CPU证据](../experiments/nonv2-crossstep65/full-r1)
- [公开重建CPU结果](../results/aggressive-20261029/nonv2-crossstep65/public-repro-cpu-r1/summary.json)
- [复现入口](../reproduce/run_nonv2_crossstep65.py)

```bash
python reproduce/verify_nonv2_crossstep65.py
python reproduce/run_nonv2_crossstep65.py --gpus 1,2 \
  --workdir /run/lock/team07-public-crossstep65 --prepare-only
```

完整GPU运行去掉`--prepare-only`，传入已验证的私有tmpfs适配器`--storage-config`和可执行FFN缓存`--cache-dir`。完整运行的source与NCCL backing必须位于同一私有tmpfs，保留原适配器的文件系统检查；CPU重建可使用`/run/lock`。不使用单卡试验耗时与历史双卡成绩的差值推断因果。
