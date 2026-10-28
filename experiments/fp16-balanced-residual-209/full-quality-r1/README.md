# 完整 V2 残差量化试验：质量失败

**316.602474 秒 / PSNR 24.786071095 dB**，原 11 轮、50 步、23 个视频完整，低于 25 dB 门槛。单 GPU 质量试验，未替换当前双 GPU 的 289.486308 秒生产版本，也没有覆盖本机正式材料。

保留全部原始 209 生产源码；新增 evaluator 别名，只在原 `configure_denoiser` 后安装残差量化 hook，原 `PROCESS_START_TIME` 与计时结束位置不动。benchmark 别名仅替换 evaluator 路径和自己的递归调用路径，继续使用官方 PSNR、结果工具及完整 workload。输入和权重 INT8 打包均进入每次重放，首样本校准及初始化进入完整进程时间。

`generated_assets.py` 从原基线生成两个精确别名，并验证原始和生成后的 SHA；避免复制整份 evaluator。其余 `scripts/*.py` 和 JSON 为实际执行的精确文件，来源证明见[结果目录](../../../results/aggressive-20261028/fp16-balanced-residual-209/full-quality-r1)。

该目录还保留完整单项 log、JSON、PSNR、profile 和 580 条 GPU 遥测：峰值 71°C、73,364 MiB，繁忙时 SM 为 1320–1410 MHz。没有固定频率或改变功耗。

复现需要原始 `unifolm` 基线、授权模型和输入、固定环境及完整 GPU 独占监控。将本目录精确脚本／JSON 放到基线的临时副本，再用 `generated_assets.generated_assets(base)` 返回的两个路径及文本生成别名。设置 `CURRENT_EXPERIMENT_FLAGS_FILE` 为该副本的 flags 文件，使用原 `scripts/team07_env.sh` 与所选 `CUDA_VISIBLE_DEVICES`，执行 `scripts/run_fp16_balanced_residual209_full.py`。这是已知质量失败的实验，用于审阅与后续数值研究。
