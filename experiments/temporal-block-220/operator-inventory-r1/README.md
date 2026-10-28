# adopted220 时间块只读清单

此检查只记录现有17个时间 Transformer 的公开 Tensor shape/stride/dtype、选定维度和 JIT 名称；不记录输入值、权重值、地址或 CUDA driver 参数。观察器不改变参数、grid、选项、精度或通信。两卡一次完整 world denoiser 的三路输出与同进程实际 adopted220 逐位相同（含 signed zero），RNG 不变。**这是算子清单，尚无新加速或完整视频质量结论。**

`temporal220_observer.py` 在 configure 前导入；`ADOPTED220_ROOT` 指向 materialize 后的实际220源树，`CANONICAL_ROOT` 指向原209源树，`INIT_FFN_CACHE_DIR` 使用220已有初始FFN缓存。既有锁、严格空闲检查、上下文所有权和全程外来上下文检查沿用已发布模型探针。未导出任何媒体、模型或凭据。

保留完整5D GroupNorm。16个常规块使用 Linear 投影，init_attn 使用 Conv1d，内部512通道。前者既有FFN末端汇合可覆盖更早的独立空间行计算；后者先汇合原始FFN输出、再加残差，不能直接套用相同边界。当前优化实验按此区别限制范围。
