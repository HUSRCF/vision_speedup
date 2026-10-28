# 更大分组的 INT8 残差原型

在实际 world FFN 激活上测试 K256、K1024、完整 K 分组，以及 N64／N128 tile。原 FP16 主项仍使用 K64 短求和，INT8 残差在组内做 INT32 累加，组末才解量化。主项与残差的组合舍入不同于生产实现，需要独立质量验证。

18 个新配置和三个原补偿对照的[结果](../../results/aggressive-20261028/fp16-grouped-residual-209)与[范围说明](../../docs/FP16_SCALAR_EXPERIMENTS_20261028.md)均已保存。只有一个配置同时通过局部精度筛选且六对更快，中位数仅省 3.226 微秒；没有完整模型、视频 PSNR 或生产安装结论。

复现方法与 [前一组实验](../fp16-scalar-residual-209/README.md)相同：将这里的五个 `scripts/*.py` 覆盖到原始 209 文件生产树的临时副本，加载该目录的 flags，运行 `scripts/run_fp16_scalar_residual209.py`。必须保留原 evaluator 与授权外部资源，整段独占监控由调用方提供。本诊断重新采集内存操作数，仅写聚合 JSON。
