# 实际 FFN 操作数上的 FP16 / scalar / INT8 残差原型

这些目录是固定实验源码，不是当前正式推理入口。`r5` 是最新测量：匹配原 N64 kernel 的流水深度 3，增加三个 alpha 的通道平衡；`r4` 移出循环内权重 predicate；`r3` 保留第一版可运行实现。

完整结论与误差定义见 [实验记录](../../docs/FP16_SCALAR_EXPERIMENTS_20261028.md)，逐项原始数据见 [结果目录](../../results/aggressive-20261028/fp16-scalar-residual-209)。没有新的算术候选同时通过速度与局部精度筛选，没有完整视频 PSNR 资格。

复现需要原始 209 文件生产树、授权 checkpoint、预处理输入、原 CUDA 扩展及其工作缓存环境，以及独占 A800。将选定 revision 的 `scripts/` 覆盖到生产树的临时副本，保留原 `scripts/evaluation`；不要覆盖已采用的正式目录。

在临时副本中加载原 `scripts/team07_env.sh`，设置 `CURRENT_EXPERIMENT_FLAGS_FILE` 为该 revision 的 `current-experiment-effective-flags.json`，设置 `KERNEL_PROFILE_SOURCE_RUN` 为被比较的原始运行目录，保留 `CKPT_PATH` 和 `INPUT_CACHE_DIR`，然后运行：

```bash
python scripts/run_fp16_scalar_residual209.py
```

原始测量由比赛机的整段 GPU 占用 guard 包裹；上述命令本身只启动诊断，不提供跨用户 GPU 互斥。每次执行从模型当前调用重新采集内存操作数，输出聚合 JSON，随后退出；不要求原始操作数文件。
