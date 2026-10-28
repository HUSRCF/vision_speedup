# 双卡实际半行形状的分离 predicate 实验

对原始 209 源码模型的三个实际 world FFN 操作数取第一 rank 半行视图，原 `XH/XL/WH/WL` 内容、原逻辑调用选择的 BM 与原基准流水深度保留。新配置将 INT8 快路径和原 Half 回退拆成两个 kernel，设备 predicate 互斥；两次启动与每次输入打包均计时。

42 个新配置加三个原补偿对照共 45 项。[1280,1280,5120] 的最佳局部合格配置省 4.608 微秒；其他形状未找到同时合格且六对更快的候选。结果尚未经过完整模型或视频 PSNR 验证，未采用到正式实现。

复现时将本目录的 `scripts/*.py` 放入原始 209 文件生产树的独立临时副本，使用本目录 flags 和授权的模型／输入，运行 `scripts/run_fp16_scalar_residual209.py`。使用与前版相同的 evaluator 和运行环境，另由调用方确保完整运行期间 GPU 独占监控。只导出聚合 JSON，不保存张量。

全部配置和证据见[结果目录](../../../results/aggressive-20261028/fp16-grouped-residual-209/r2)，后续正向改进见[残差通道缩放](../../fp16-balanced-residual-209)。
