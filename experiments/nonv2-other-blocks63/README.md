# 非 V2 完整内部块分片：逐项验收

沿用已采用 Z1 的三个数学路由 helper，分别为双臂堆箱、清理铅笔、G1 建立63文件源码。原59个数学源码逐字节保留；仅生命周期 runtime 的三个数据集断言按分项替换，原 flags、DDIM 步数、完整轮数及 frame stride 保留。16个空间和17个时间内部块共享原33次输出汇合，完整5D GroupNorm及16时间步保留。原生 Half GEMM 的形状会变化，因此必须逐项检查完整闭环视频质量。

| 分项 | GPU 数 | 完整轮数／视频 | process_wall_s | 官方 PSNR dB | 处理 |
|---|---:|---:|---:|---:|---|
| 双臂堆箱 | 2 | 7／15 | 94.319011 | 31.972206776 | 已采用 |
| 清理铅笔 | 2 | 8／17 | 105.949243 | 34.881926888 | 已采用 |
| G1 相机装箱 | 2 | 11／23 | 142.834192 | 19.081776215 | 质量失败，未采用 |

双臂堆箱相对历史单卡103.111184秒少8.792173秒，清理铅笔相对117.077862秒少11.128619秒。两项在GPU0/1及GPU2/3上并行完成；这些是不同卡数的历史完整运行比较，没有同期同GPU完整AB对照，不能把全部时间差归因于单个kernel。两rank的分项视频SHA一致，初始化、双模型加载、通信、首次图捕获、写出、helper退出和rank视频核对均在原process timer内。

G1保留 `UNIFOLM_FP32_FEEDBACK_HEADS=1` 和 `UNIFOLM_SIM_HEAD_TF32=0`；实际日志确认862个受保护模块。两rank视频一致并不意味着与参考视频一致。完整闭环PSNR失败原因尚未定位，Half GEMM形状／舍入变化是待检查方向，不能认定为已经证明的原因。当前正式G1仍为单卡160.897537秒／26.107156332dB。

采用的32个新视频与其余71个既有已验收视频重新通过官方工具的103视频校验，当时五项合计 **782.213299秒**，其双文件保留在 `results/history/official-other63-before-decision223`；随后V2更新为223后，当前根双文件为 **778.585395秒**。这是一份分别计时的完整分项汇总，不是连续五项进程墙钟时间。网站未上传。当前可提交双文件为仓库根 `results/runtime.log` 和 `results/result.json`；原802.134091秒双文件保留在 `results/history/official-z1-blocks63-before-other63`。

`results/aggressive-20261028/nonv2-other-blocks63` 保留各项完整日志、JSON、profile、遥测、rank视频哈希以及新 `official103` 汇总。三个单项shell都因其余四项缺席退出1；所选分项的推理进程退出0，工作量完整。G1的质量失败完整保留，不能拿任何单项原始JSON冒充五项有效提交。

从仓库根运行：

```bash
python3 reproduce/verify_snapshot.py
~/.venv/bin/python reproduce/run_nonv2_parallel.py --case dual --gpus 0,1 \
  --workdir /var/tmp/team07-review-dual-63 \
  --storage-config /var/tmp/team07-review-storage/storage-config.json \
  --cache-dir /path/to/executable/FFN-shape-cache
```

`--case pencils` 为另一个已采用分项；`--case g1` 复现的是失败实验。存储配置用 `experiments/private-nccl-tmpfs/build_and_check.py` 构建，其host `.so`及编译缓存必须位于可执行挂载。`--workdir` 必须是新目录。`--prepare-only` 不使用Torch/GPU，三个分项的精确源码与计时资源已在比赛机通过检查。冷缓存编译仍计入原进程时间，可能更慢；不清理或放宽原warm缓存／CUBIN校验。
