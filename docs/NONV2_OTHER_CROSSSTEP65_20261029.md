# 双臂堆箱、清理铅笔：双卡时间步复用完整结果

五项正式成绩已经全部使用两张 A800，当前有效合计为 **764.366152 秒**。本轮进一步把 Z1 已采用的深层时间步缓存接到双臂堆箱和清理铅笔的现有双卡路径，两项独立跑完整视频并检查质量。

| 分项 | 本轮双卡 process_wall_s | PSNR dB | 当前正式双卡时间 | 结论 |
|---|---:|---:|---:|---|
| 双臂堆箱 | 94.595307 | 26.460281025 | 94.319011 | 质量通过，未改善现有记录 |
| 清理铅笔 | 105.993354 | 33.050495011 | 105.949243 | 质量通过，未改善现有记录 |

两项均在 GPU1／3 上通过全程 foreign-context guard，第一次尝试完成；两 rank 视频 SHA 一致。比较列是历史正式成绩，**没有同期配对完整进程计时**，不足以证明该缓存策略存在稳定回退或收益。没有替换正式 log／JSON，没有生成新的五项103视频汇总，也没有上传网站。

双臂完整7轮／15视频，两阶段分别350次 denoiser 调用、105次复用；清理铅笔完整8轮／17视频，两阶段分别400次调用、120次复用。每轮、每阶段原有50次 DDIM 更新全部保留，浅层、输出 head、动态条件与 sampler 更新继续执行。

策略与 Z1 一致：缓存 `input_blocks[9:]`、middle、`output_blocks[:3]` 的7个父块，两阶段位置11、13、…、39使用缓存，另外35步刷新。与各项原63文件数学路径相比，三个分片 helper、原参数和计时入口保持一致；新增两个复用模块，修改并行 runtime 接线以及 CUDA Graph 可选共享池，形成65文件。复用时仍调用原并行执行器，每个角色严格核对其实际捕获的普通 Torch 模块名：完整图33次汇合，复用图31次，两阶段共四张图、128个捕获记录。

CPU 接线验证检查1200次模拟调用全部经过并行执行器、RNG不变、浅层和head持续执行、模块名单及重复名拒绝；它不推断实际模型视频质量。两个公开入口分别在比赛机禁用 CUDA 重建65文件和全部计时资源，SHA 与实际运行一致。完整 GPU 运行再独立验证各项工作量、四种图角色、PSNR和双 rank 视频一致性。

双臂两张卡显存峰值均31048MiB，最高温度71／58°C；利用率至少50%的样本中，SM频率中位1380／1395MHz，并非恒定1410MHz。原始遥测及各项阶段计时均保留于完整证据目录。

- [双臂冻结源码、guard及CPU证据](../experiments/nonv2-other-crossstep65/dual-full-r1)
- [清理铅笔冻结源码、guard及CPU证据](../experiments/nonv2-other-crossstep65/pencils-full-r1)
- [完整log／JSON、工作量、两rank与遥测证据](../results/aggressive-20261029/nonv2-other-crossstep65)
- [公开重建与运行入口](../reproduce/run_nonv2_other_crossstep65.py)

```bash
python reproduce/verify_nonv2_other_crossstep65.py
python reproduce/run_nonv2_other_crossstep65.py --case dual --gpus 1,3 \
  --workdir /run/lock/team07-review-dual-crossstep65 --prepare-only
python reproduce/run_nonv2_other_crossstep65.py --case pencils --gpus 1,3 \
  --workdir /run/lock/team07-review-pencils-crossstep65 --prepare-only
```

完整 GPU 运行去掉 `--prepare-only`，传入已验证的私有 tmpfs `--storage-config` 和可执行 FFN 缓存 `--cache-dir`。完整运行的 source 与 NCCL backing 仍须位于同一私有 tmpfs；CPU重建可使用 `/run/lock`。所有初始化、通信、图捕获、视频写出及 helper 退出保留在原进程计时内。

G1正式双卡仍为155.314304秒／26.107156332dB，保留完整 Half GEMM 形状的68文件路径。其63文件数学路径有完整质量失败记录，因此没有把这次依赖63路径的缓存组合直接套到G1。
