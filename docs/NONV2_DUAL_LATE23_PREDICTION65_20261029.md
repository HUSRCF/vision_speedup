# 双臂堆箱：23块双卡时间步预测，已采用

GPU0／1完整7轮15视频为 **84.577327秒／PSNR26.556920797dB**，两rank视频SHA一致，严格全程守护clean。比此前13块双卡正式90.557285秒少 **5.979958秒（6.60%）**；这是历史成绩比较，没有同期同卡完整AB。

新的官方103视频汇总为accepted，五项合计 **738.805454秒**，external_wall_s为738.905615秒，其他88视频SHA保持一致。本机`GPU/runtime.log`和`GPU/result.json`已备份更新，网站未上传。五项材料分别来自各次完整运行，汇总不代表连续五项墙钟计时。

## 实现

使用已在清理铅笔通过的23块、`[20,40)`两次完整刷新预测，但独立验证双臂任务质量。保留双臂原始采样参数、全部7轮及每阶段50次DDIM更新，浅层与head继续实时执行。每阶段每轮10次预测、40次完整刷新；预测历史只来自完整刷新，系数按真实位置间距计算。两阶段各350调用／70预测，总alpha直方图为1.0共14次、0.5共126次。

原63文件数学辅助模块不变；相对于铅笔23块65源码，仅运行时的三个任务断言替换为双臂任务。计时入口与预测算术完全相同，任务flags保留双臂原配置。四图分别为decision／world完整和复用图：完整33次汇合，复用3次，各阶段36条、共72条捕获记录。复用图保留init_attn及最后output块的两个Transformer模块。

公开入口在比赛机禁用CUDA，精确重建65源码及实际计时资源；1200次玩具执行器调用、3600个预测输出、23块覆盖、复用3次汇合、区间及词法缓存环境／Triton缓存类恢复检查通过。CPU结论不代替完整15视频质量。

此次双臂运行与独立GPU2上的V2候选部分重叠执行，各自严格检查所属GPU无外来上下文。完整耗时包含本次并发时的系统环境，不将跨GPU进程耗时差解释为纯算子提升。官方汇总保留原有`not active_jobs`检查，在两个GPU实验均终止后才启动。

## 复现与证据

```bash
python reproduce/verify_nonv2_dual_late23_prediction65.py
~/.venv/bin/python reproduce/run_nonv2_dual_late23_prediction65.py \
  --gpus 0,1 --workdir /path/on/private-tmpfs/dual-late23 \
  --storage-config /path/to/qualified-storage.json \
  --cache-dir /path/to/qualified-FFN-cache \
  --prediction-cache-parent /path/on/executable-filesystem
```

- [冻结源码、计时资源和守护脚本](../experiments/nonv2-late23-prediction65/dual-full-r1)
- [完整log、JSON、两rank报告与遥测](../results/aggressive-20261029/nonv2-late23-prediction65/dual-full-r1)
- [公开CPU精确重建](../results/aggressive-20261029/nonv2-late23-prediction65/dual-public-repro-cpu-r1/summary.json)
- [新103视频官方汇总及脚本](../results/aggressive-20261029/nonv2-late23-prediction65/dual-official103)
- [采用记录与双文件SHA](../experiments/nonv2-late23-prediction65/dual-accepted-outcome.json)
- [此前744.785412秒材料](../results/history/official-before-dual-late23-prediction65)

135条遥测中，两卡峰值显存均31788MiB，最高温61／72°C，利用率至少50%的样本中SM频率分别1335–1410／1290–1410MHz。没有据此推断温度或频率对速度差异的因果关系。
