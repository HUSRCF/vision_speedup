# 精确时间嵌入复用：独立实验

当前正式五项合计764.366152秒，Z1的150.838389秒来自已经验收的动态深层特征复用。本页研究另一个方向：同一阶段、同一timestep和帧率下，在后续轮次精确复用时间MLP输出。本实验完整运行 **157.446954秒／46.553398187dB**，25视频与原63正式视频及另一rank的SHA一致，但慢于当前150.838389秒，因此未替换正式材料。

源码确认WMA的`time_embed`仅依赖timestep，`fps_embedding`仅依赖帧率，各`ResBlock.emb_layers`仅接收两者组合。只选择这24个小模块，不缓存视频特征、action／state、attention或采样输出。第一轮每阶段完整计算50个slot；后续轮次读取相同slot。仍运行全部12轮、每阶段50步、每次原动态主干、head和采样更新。两阶段分别保存表和图，保留33个双卡汇合；四张图要求132个普通Torch模块记录。

每次使用检查声明的timestep／fs值、backend模式、权重对象与版本及执行顺序。变更时拒绝旧缓存，要求新运行；没有隐式继承旧表。缓存、图与新增kernel编译均位于原进程计时内。

CPU原型覆盖FP32和Half各12轮／24段50步序列，共2400个完整输出逐位一致，RNG不变；另检查7类错误／失效输入。实际WMA `ResBlock`接线的200次输出也逐位一致，200次调用都经过并行执行器，动态卷积每次执行。这些CPU结果不证明GPU完整视频质量或速度。

GPU尝试与修正边界如下：

| 版本 | 已知结果 | 下一步变化 |
|---|---|---|
| r1 | 原Torch `index_copy_`在捕获时报错，无有效成绩 | 增加纯位复制Triton读写 |
| r2 | 新kernel的launcher写入home时ENOSPC，无有效成绩 | 原warm键只读、新键放私有可执行盘 |
| r3 | 全局自定义manager触发原FFN的缓存隔离断言，无有效成绩 | 把manager限制在新增读写kernel的词法范围 |
| r4 | 全程独占guard、12轮／25视频、质量和原63视频SHA全部通过，157.446954秒 | 比当前正式慢，未采用 |

r4保留原FFN断言和三个数学helper。新增kernel调用前保存环境、Triton全局缓存类及名称，结束后逐项恢复；CPU检查随后进入原FFN缓存上下文，核对仍使用原FileCacheManager。rank0／1分别使用`/run/lock`内0700目录，home warm键只读。没有清理home、共享缓存或历史材料。

r4在计时内增加两种dtype共100个slot的CUDA Graph复制自检，包括负零、无穷和NaN原始位比较；该检查已在本次比赛机启动中通过，随后四张原denoiser图捕获逐位核对成功。完整两阶段各600次调用，其中各50次填表、550次读表；24个模块共96条时间缓存捕获记录，原双卡132个捕获节点全部核对。完整进程未显示可采用的加速。普通Tensor参数与显式slot索引实现缓存，没有读取原生CUDA图参数或导出张量数组。

- [冻结源码和guards](../experiments/nonv2-static-timestep65)
- [CPU原型](../experiments/static-timestep-cache/cpu-r1)
- [公开重建与失败证据](../results/aggressive-20261029/nonv2-static-timestep65)

四份公开入口均在比赛机禁用CUDA重建，源码与计时资源SHA匹配。当前实验入口：

```bash
python reproduce/verify_static_timestep65.py
python reproduce/run_nonv2_static_timestep65_r4.py --gpus 1,2 \
  --workdir /var/tmp/team07-review-static-time-r4 --prepare-only
```

完整GPU复现去掉`--prepare-only`，传入已验证的`--storage-config`、原FFN形状`--cache-dir`，以及新的可执行`--kernel-cache-dir /run/lock/team07-review-static-kernels-r4`。后者必须不存在，入口会创建按rank隔离的目录；初始化与GPU／NCCL工作仍计入原计时。
