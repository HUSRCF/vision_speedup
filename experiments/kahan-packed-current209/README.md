# 八处 Kahan 静态权重预打包

此目录保留已完成的单卡诊断，未替换当前双卡生产结果。

- `model-r1`：原始 209 文件不变，八处 Kahan owner 增加随权重版本更新的 Half high/low 缓冲。六组输入与 570 Linear、782 norm、192 Conv2d 参数修改／清零／恢复检查，三个输出全部逐位一致。六对完整 world 前向均更快，中位省 1.576950 ms。
- `full-quality-r1`：原始计时器内增加三行安装 hook。完整 V2 为 **324.052568 秒 / PSNR 29.675324786 dB**，11 轮、23 个视频均完整，视频 SHA256 与原始单卡基线全部相同。比历史单卡 316.774396 秒慢 7.278172 秒；这不是同次配对进程测量，没有证明完整进程加速。

模型参数、输入、参考视频及输出视频均不包含在此目录。运行需要比赛授权资源、原始环境和已验证编译缓存；原始三指针 cubin 校验未放宽。当前生产成绩仍为 V2 289.486308 秒、五项合计 838.392232 秒。

精确模型脚本以 `unifolm` 为工作根目录，复制 `model-r1/scripts` 与其 flags 后使用原模型诊断启动参数。完整试验先把 `full-quality-r1/scripts` 复制到独立的原始 209 文件源码副本，再调用该目录 `generated_assets.generated_assets(base)` 取得两个别名文件；生成器校验原始文件与生成结果 SHA256。`kahan-packed-full-added-files.json` 列出运行时归档所需资产，runner 保留原完整 V2 启动参数和官方工具。不能将完整别名 hook 安装到当前双卡 overlay 后声称该组合已验证。

全部原始测量及源码证明见 [`results/aggressive-20261028/kahan-packed-current209`](../../results/aggressive-20261028/kahan-packed-current209)，验证范围与遥测分析见 [`docs/REVIEW_RESPONSE_20261028.md`](../../docs/REVIEW_RESPONSE_20261028.md)。
