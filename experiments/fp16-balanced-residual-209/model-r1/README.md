# 完整 world 模型的残差量化检查

单张 A800、原始 209 生产源码和全部原 flags。只替换十次 `[2560,1280,5120]` world FFN 最后投影；其他形状、decision 与 sampler 不变。原 Half 主乘积保留，残差通道乘数固定为该 owner 第一次 eager 激活校准的 alpha=0.5 方案。

权重与输入 INT8 打包均在每次图重放内执行，包含全部成本，保证修改参数时不会复用旧量化权重。非零权重低位仍使用设备 predicate 进入原三次 Half 回退。

六组输入及 570 Linear／782 norm master 修改、清零、恢复检查通过；三个 FP32 输出记录近似误差，原基准和 RNG 恢复正常。原 23 个 SGEMM、45 个 head cache owner 未缩减。初始视频与 state 输出相对 L2 为 2.13673e-5、3.55792e-5。六对完整前向计时均更快，每次中位省 **0.975357 ms**。完整 V2 PSNR 尚待独立验证，未采用到生产实现。

复现：将本目录 `scripts/*.py` 放入原始 209 文件生产树的临时副本，加载这里的 flags 并设置 `CURRENT_EXPERIMENT_FLAGS_FILE`、`KERNEL_PROFILE_SOURCE_RUN` 及授权模型／输入／运行环境，执行 `scripts/run_fp16_balanced_residual209_model.py`。调用方需在整个测量期间独占监控 GPU。CPU 导入及 hook 还原检查在 runner 中执行；统计结果不包含原始操作数。

全部[结果与源码 SHA](../../../results/aggressive-20261028/fp16-balanced-residual-209/model-r1)均保留；本诊断的 0.002 gross gate 不能代替官方视频 PSNR 门槛。
