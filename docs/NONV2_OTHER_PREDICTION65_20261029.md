# 双臂采用13块双卡预测，铅笔质量失败

双臂堆箱的新正式记录为 **90.557285秒／26.513549583dB**，完整7轮、15视频、双rank视频SHA一致。新103视频官方汇总accepted，其余88视频SHA不变，五项正式合计 **755.093960秒**。本机 `GPU/runtime.log` 与 `GPU/result.json` 已备份更新，网站未上传。

| 任务 | 本轮 process_wall_s | PSNR dB | 前一正式双卡时间 | 结论 |
|---|---:|---:|---:|---|
| 双臂堆箱 | 90.557285 | 26.513549583 | 94.319011 | 质量通过，采用 |
| 清理铅笔 | 100.661865 | 19.202957201 | 105.949243 | 质量失败，排除 |

双臂比历史正式记录少3.761726秒；没有同期配对完整进程计时，不把该历史差值当作稳定因果收益。铅笔的完整8轮／17视频虽更快，但未达到25dB，正式材料继续保留105.949243秒／34.881926888dB的63文件版本。两个任务都没有减少轮数或每阶段50个DDIM更新。

预测模块与[已采用Z1的65文件版本](NONV2_PREDICTION65_20261029.md)逐字相同：缓存两次完整刷新，按实际位置间隔外推13个深层父块，位置11、13、…、39预测，其他35步刷新；浅层、head、条件和sampler继续执行。新kernel的私有缓存作用域恢复原环境与Triton缓存类，原FFN断言保留。没有继承Z1的PSNR结论。

每项只专门化三个数据集断言和该项原有flags；三个分片数学helper与原63文件版本保持一致。两个src模块增加及CUDA Graph／并行接线修改形成65文件，源码、计时资源、guard及flags均固定SHA。每个任务独立在GPU1／3通过全程foreign-context guard，第一次尝试完成，四张图均与相应即时执行一致。

| 验证项 | 双臂堆箱 | 清理铅笔 |
|---|---:|---:|
| 正式轮数 | 7 | 8 |
| 完整视频数 | 15 | 17 |
| 每阶段denoiser调用 | 350 | 400 |
| 每阶段预测次数 | 105 | 120 |
| 完整图／预测图汇合 | 33／21 | 33／21 |
| 两阶段图捕获记录 | 108 | 108 |

公开入口分别在比赛机禁用CUDA精确重建65文件及计时资源，匹配实际运行SHA，并重做1200次并行执行器接线检查、3600个仿射完整输出及缓存作用域恢复检查。GPU预测自检覆盖Half／FP32、直接／残差四组257元素尾块，捕获后修改历史与系数再重放。这些局部证据没有代替独立完整视频质量。

双臂两卡各峰值31174MiB、最高70／57°C；铅笔各峰值31174MiB、最高71／58°C。利用率至少50%的样本中，两项SM频率中位均约1380／1395MHz；原遥测全部保留。

官方汇总只选择本轮双臂15个真实计时视频，与上一已接受汇总中SHA相同的其余88个视频组成普通文件目录，再重新执行官方PSNR与result工具。未重跑GPU、未改写分项时间，日志每项证据唯一。新旧正式双文件均保留；不能把该汇总描述为一次连续五项推理墙钟。

- [冻结源码、guard与CPU证据](../experiments/nonv2-other-prediction65)
- [两项完整log／JSON、双rank与遥测](../results/aggressive-20261029/nonv2-other-prediction65)
- [新103视频官方验收](../results/aggressive-20261029/nonv2-other-prediction65/official103)
- [前一正式双文件](../results/history/official-before-other-prediction65)
- [双臂公开CPU重建](../results/aggressive-20261029/nonv2-other-prediction65/dual-public-repro-cpu-r1/summary.json)
- [铅笔公开CPU重建](../results/aggressive-20261029/nonv2-other-prediction65/pencils-public-repro-cpu-r1/summary.json)

```bash
python reproduce/verify_nonv2_other_prediction65.py
python reproduce/run_nonv2_other_prediction65.py --case dual --gpus 1,3 \
  --workdir /run/lock/team07-review-dual-prediction65 --prepare-only
```

完整GPU运行去掉 `--prepare-only`，使用同一私有tmpfs的source与NCCL backing，传入已验证 `--storage-config`、可执行FFN `--cache-dir`，可选 `--prediction-cache-parent` 指定新缓存父目录。初始化、双模型搬运、通信、图捕获、预测自检、视频写出与helper退出都保留在原process timer内。`--case pencils`复现的是上述质量失败候选，当前正式铅笔仍用 `run_nonv2_parallel.py --case pencils`。
