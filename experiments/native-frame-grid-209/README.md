# 完整 N16 描述符下的半帧原生卷积实验

- `inventory-r2`：六种几何、三种输入布局、两个 rank，共 36 组合成检查。填充后的前八帧与原完整 16 帧的对应半帧逐位一致；只读图检查找到 24 个原缓存识别条件支持的卷积节点。
- `fixture-r1`：六种几何、NCHW／NC-permuted、两个 rank，共 24 组。保持所有参数字节、函数、block、shared、split-K 和原生 memset，只缩小空间启动网格。每组六个动态状态的前八帧输出均逐位一致；恢复原网格后全部 16 帧逐位一致。六对完整卷积计时包含原变换与 epilogue，所有组合均更快，单次省约 0.128～0.843 ms。

这里只计算该 rank 的有效八帧，另八帧是填充位置。尚未完成双卡通信、完整 ResBlock／模型或视频质量检查，不能视为完整推理加速，也未替换生产代码。

脚本在原始 209 文件版本中运行，需要原依赖、flags 和已验证图句柄扩展；只使用脚本内生成的合成输入和权重，无需授权模型或输入缓存。没有导出原始操作数或 native 参数地址。完整范围与限制见 [`docs/NATIVE_FRAME_GRID_EXPERIMENTS_20261028.md`](../../docs/NATIVE_FRAME_GRID_EXPERIMENTS_20261028.md)，原始结果见 [`results/aggressive-20261028/native-frame-grid-209`](../../results/aggressive-20261028/native-frame-grid-209)。
