# 只对 INT8 残差做通道缩放

保持原 `XH @ WH`、K64 FP32 主项求和、bias 与 residual 输出，只对 `XL @ WH` 的 INT8 近似应用互逆通道乘数。从第一份实际 world 激活校准通道乘数，未做跨 timestep 校准。非零权重低位由设备 predicate 回退到原三次 Half 路径。

在当前双卡的 `[1280,1280,5120]` 形状上，alpha=0.5、完整 K 分组、BM64/BN128、流水深度 3：

- 六对计时均更快，每次中位省 **0.077210 ms**，约 **1.274 倍**加速。
- 去掉 bias/residual 后相对 L2 **4.98374e-6**，通过原预设的 6e-6 局部筛选；FP64 样本相对误差 **3.57889e-6**。
- 动态输入打包和两个设备 predicate kernel 的启动计入每次重放；静态权重准备与首样本校准未计入重放。
- 输入变化、非零低位回退、原模型输出、RNG 和全部 master 版本恢复检查通过。

24 个新配置，另有三种形状的原补偿对照，共 27 项。本目录记录算子原型；后续 [model-r1](model-r1) 已通过六组完整 world 前向检查，并将权重打包也放入每次重放，整网每次中位省 0.975357 ms。完整视频 PSNR 尚未验证，正式成绩不变。静态权重和校准的生命周期仍需接入，初始化成本必须计入正式进程时间。

复现：将这里的 `scripts/*.py` 放入原始 209 文件生产树的独立临时副本，加载 `current-experiment-effective-flags.json`，提供授权的模型、输入、原 evaluator 与运行环境，运行 `scripts/run_fp16_scalar_residual209.py`；调用方提供 GPU 独占和整段监控。该诊断只写聚合 JSON。

完整[结果与源码 SHA](../../results/aggressive-20261028/fp16-balanced-residual-209)和[方法及限制](../../docs/FP16_SCALAR_EXPERIMENTS_20261028.md)均保留。
