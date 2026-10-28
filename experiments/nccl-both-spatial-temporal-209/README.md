# 当前正式 V2：两阶段空间分片＋world 时间分片

完整 V2 为 **289.486308 秒 / PSNR 29.675324786 dB**，使用两张 A800。decision/world 各 16 个完整空间 Transformer 分配独立行或帧；world 另拆 16 个时间 FFN 与 88 个时间卷积。vendor GEMM 的逻辑尺寸和原精度标志保持原值。

`production-overlay/` 直接保存该正式运行 source.tar.gz 中新增的六个 host helper。`production_sources.py` 从原 evaluator 只追加四处生命周期 hook；`overlay-source-hashes.json` 验证生成入口与六个 helper 都逐字节等于正式快照，移除 hook 后也必须精确恢复原入口。原有 209 文件和 110 个 JIT 函数未改写。`materialize.py` 将其合并为精确 215 文件版本。

使用仓库根目录的 `reproduce/run_v2_parallel.py --gpus 1,3 --workdir /path/to/new/directory`。默认即此版本；历史 world-only 215 和 FFN 213 版本可通过 `--variant` 选择。运行仍需授权模型、输入及原 CUDA 扩展环境，冷缓存编译指纹的已知限制见 [审阅回复](../../docs/REVIEW_RESPONSE_20261028.md)。

原始进程计时包含 helper 启动、通信、双模型传输、图捕获、全部输出和 helper／图退出。两个阶段分别通过六组模型逐位检查；完整 23 视频与原单卡和另一 rank 全部 SHA256 相同，官方 103 视频复核通过。代码、计时和遥测的对应关系见 [完整证据](../../results/aggressive-20261028/nccl-both-spatial-temporal-full-r1)。
