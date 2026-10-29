# 已采用：双卡13块时间步预测

Z1正式记录更新为 **145.327923秒／28.992932876dB**，五项合计 **758.855686秒**。完整12轮、25视频、两rank视频SHA一致；新103视频官方汇总accepted，其他78视频SHA不变。本机 `GPU/runtime.log` 和 `GPU/result.json` 已备份更新，网站未上传。五项仍全部使用两张A800，汇总来自分别完成的分项运行，不是一次连续五项墙钟计时。

相比历史正式7块缓存的150.838389秒，当前记录少5.510466秒；**没有同期配对完整进程计时**，不把历史差值全部当作稳定因果收益。单卡13块预测此前为161.216743秒／29.507847903dB，本次双卡重新验证质量，没有继承单卡PSNR。

预测算法见[单卡完整记录](TEMPORAL_PREDICTION61_20261029.md)：保存前两次完整刷新的结果，按实际位置间隔线性外推。范围为 `input_blocks[6:]`、middle、`output_blocks[:6]` 共13个父块；位置11、13、…、39预测，其余35步完整刷新。每轮每阶段原有50次DDIM更新保留，浅层、head、条件与sampler更新继续执行。两阶段各600次调用、180次预测，24段完整50步序列；系数1共24次，系数0.5共336次。

与正式63文件分片路径相比，三个数学helper保持一致。新增两个预测／复用模块，改变并行runtime接线及CUDA Graph可选共享池，仍为65个src文件。预测kernel、Controller、GPU自检与单卡版本的数学源码一致；唯一新增上下文是预测kernel的私有编译缓存作用域。

| CUDA Graph角色 | 汇合次数 |
|---|---:|
| decision完整 | 33 |
| decision预测 | 21 |
| world完整 | 33 |
| world预测 | 21 |

每张图对实际捕获的普通Torch模块名单进行严格检查，两阶段合计108个捕获记录。预测版使用四张图，复用时仍经过双卡执行器。原FFN的缓存断言不放宽；新增预测kernel临时使用每rank私有GPU／可执行launcher缓存，退出作用域同时恢复三个环境变量与Triton2.3的两个缓存类全局变量。该逻辑沿用已验证的精确时间嵌入缓存r4实现。

CPU验证同时覆盖1200次调用通过并行执行器、模块范围、3600个仿射完整输出、失效与warm-up语义，以及新kernel缓存作用域退出后原FFN仍使用标准缓存类。公开入口在比赛机禁用CUDA精确重建65文件及全部计时资源，再执行同一CPU检查。完整GPU运行包含Half／FP32、直接／残差四组257元素尾块的捕获后修改重放检查；这些局部检查与闭环质量独立。

首个GPU2／3预约在等待期间发现GPU2被其他进程占用，未创建benchmark或推理输出。核实该等待进程仍存活、GPU1／3空闲后，只终止我们自己的等待进程，保留原记录，再以r2在GPU1／3完成整项。r1和r2的数学源码、计时资源、guard及flags相同；没有放宽空卡准入和全程foreign-context检查。

两卡显存峰值均31174MiB，最高72／58°C；利用率至少50%的样本中，SM频率中位1380／1395MHz，范围1260–1410／1275–1410MHz。完整遥测221个采样均保留。

- [冻结65文件增量、计时资源与CPU证据](../experiments/nonv2-prediction65/full-r2)
- [完整145.327923秒log／JSON、两rank、遥测](../results/aggressive-20261029/nonv2-prediction65/full-r2)
- [新103视频官方验收](../results/aggressive-20261029/nonv2-prediction65/official103)
- [上一正式双文件](../results/history/official-before-prediction65)
- [公开CPU重建结果](../results/aggressive-20261029/nonv2-prediction65/public-repro-cpu-r1/summary.json)

```bash
python reproduce/verify_nonv2_prediction65.py
python reproduce/run_nonv2_prediction65.py --revision 2 --gpus 1,3 \
  --workdir /run/lock/team07-review-prediction65 --prepare-only
```

完整GPU运行去掉 `--prepare-only`，传入已验证私有tmpfs `--storage-config` 和可执行FFN `--cache-dir`；可选 `--prediction-cache-parent` 指定新预测缓存的可执行父目录。完整运行的source与NCCL backing仍须位于同一私有tmpfs，CPU重建可使用 `/run/lock`。初始化、双模型搬运、通信、预测自检、图捕获、视频写出和helper退出都在原process timer内。
