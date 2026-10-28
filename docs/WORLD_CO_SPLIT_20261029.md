# World Winograd 按输出通道分片

本轮以已采用的 **223 文件双卡版本**为共同基线，检验审阅建议中的 CO 分片。保留原编译内核、全部 constexpr、K 累加顺序、batch stride 和固定 CUBIN 校验，仅让各 rank 计算 tile 对齐的半数输出通道。原生输入／权重／输出变换和 bias 保留，在 Conv2d 返回处打包本地通道、汇合一次并恢复原输出 stride。没有新增 Triton JIT。

完整卷积计时包括 pack、all-gather 和布局恢复；不将 GEMM 毛节省当作整块收益。校验只使用已有 owner 管理的合法参数，不导出地址或原始操作数。master 版本变化和原有 23 个卷积 owner、45 个 head cache 的完整检查先执行，再安装增量。

## 卷积边界与选择

[`conv-r2`](../results/aggressive-20261028/world-co-split223/conv-r2) 覆盖 14 种几何、NCHW／NC_permuted 两种布局、两个 rank、六种动态状态，共 336 次完整输出逐位检查。每个案例先污染原生输出工作区，再验证完整 16 帧输出；输入／权重改变、清零、恢复、RNG 和原生／数值路径恢复均通过。

只选择当前 23 个真实 owner 中，两个 rank 和两种布局下六对计时全部更快，且中位净收益最小值超过 0.05 ms 的项。最终 9 个 owner：2 个 packed、7 个 Kahan；新增 9 次 world 汇合，每 rank 本地片段合计 187.5 MiB／forward。完整选择与未选中项见 [`selection.json`](../experiments/world-co-split223/model-r1/selection.json)。未在当前模型中出现的快几何不计入预算。

| 几何 CO, COLS, K | 真实 owner 数 | 两 rank／布局最小中位净省 ms |
|---|---:|---:|
| 1280, 192, 640 | 1 | 0.094208 |
| 1280, 192, 1920 | 1 | 0.577365 |
| 1280, 640, 1280 | 1 | 0.640000 |
| 640, 640, 1280 | 1 | 0.536747 |
| 640, 640, 960 | 1 | 0.335360 |
| 640, 2560, 640 | 1 | 0.730795 |
| 320, 2560, 960 | 1 | 0.354304 |
| 320, 2560, 640 | 2 | 0.327851 |

加权合计 3.924480 ms／world forward，只是合成卷积预算。320,2560,320 和 Kahan 640,640,320 退化；640,640,640 的净省仅约 0.019456 ms，均未启用。首轮三个形状全是 packed，其中两个快形状不属于当前 23 个 owner；不能把首轮结果称为 Kahan 或当前模型提速。

初始 CPU checker 在 channels-last 的 byte view 处失败，候选恢复布局本身保持原 stride。修正 checker 为连续字节比较后才启动 GPU；[失败源码和日志](../experiments/world-co-split223/cpu-checker-failed-r0) 保留。

## 真实模型验证

[`model-r1`](../results/aggressive-20261028/world-co-split223/model-r1) 在 GPU0／1、同一进程和同一组 master 下交替测试。六种状态覆盖 570 个 Linear、782 个 norm、192 个 Conv2d 参数；三个输出逐位相同，RNG、master 恢复与条件缓存刷新检查通过。

两个 rank 的六对完整 world 前向均更快，中位净省分别 **3.836426／3.806206 ms**，约从 229 ms 降至 225 ms。按 550 次前向粗算约 2.1 秒；这不是完整进程收益或视频质量证明。原始 harness 日志仍带历史 `INITIAL_BOUNDARY221_MODEL` 标签，实际 baseline identity 明确绑定当前 223 文件版本。

完整 223／225 V2 对照使用同 GPU0／1、相同 guard、相同原始 process timer 与私有 tmpfs 适配器，依次执行。225 仅增加两个 helper 并修改 runtime 接入，不改变原算术源码。初始化、图捕获、通信、视频写入和退出计入耗时。完整运行已结束并通过新103视频验收：**260.425380 → 257.945205秒**，本次净省 **2.480175秒**；world生成CUDA **138.413630 → 136.213375秒**，decision **110.257646 → 110.292218秒**。两个rank的23视频与原正式版本SHA相同、PSNR **29.675324786dB**。已采用225，五项 **771.314275秒**，本机GPU双文件已更新，网站未上传。相对历史正式265.216325秒少7.271120秒，其中包含跨运行波动。原始对照、温度／频率汇总见[完整AB](../results/aggressive-20261028/world-co-split223/complete-process-AB.json)，验收原文见[official103](../results/aggressive-20261028/world-co-split223/official103)，旧778.585395秒材料保留于[历史目录](../results/history/official223-before-world-co225)。

可在仓库根目录执行 `python3 reproduce/verify_world_co223_evidence.py` 检查已发布卷积和模型证据；完整进程结果与正式采用结论单独记录。

实验入口为 `python3 reproduce/run_v2_world_co225.py --gpus 0,1 --workdir /var/tmp/new-co225 --prepare-only`。已在比赛机用这份公开脚本实际完成 CPU preparation，重建的 225 文件及计时资源 SHA 与完整运行一致，没有初始化 GPU，home 余量未改变。去掉 `--prepare-only` 运行时需要与原223相同的授权权重／输入、可执行缓存和合格的 `--storage-config`；225已设为`run_v2_parallel.py`默认，旧223可显式传入`--variant decision-temporal223`。两份公开入口已分别在比赛机CPU验证，无GPU执行、home余量不变。
