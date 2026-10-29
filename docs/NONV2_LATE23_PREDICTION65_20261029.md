# 清理铅笔：23块双卡时间步预测，已采用

当前五项均使用两张A800。清理铅笔新结果为 **95.640695秒／PSNR26.798606240dB**，完整8轮、17视频，两rank视频SHA一致。相对历史正式105.949243秒减少 **10.308548秒（9.73%）**；这是历史比较，尚无同GPU同期完整AB，不能将差值全部解释为纯算子收益。

新的官方工具103视频汇总为accepted，五项合计 **744.785412秒**，external_wall_s为744.872363秒。其他86个视频SHA保持一致。本机`GPU/runtime.log`与`GPU/result.json`已备份并更新，网站尚未上传。五项材料来自分别完成的运行，汇总校验不代表连续五项墙钟计时。

## 实现及精度边界

上一13块`[20,40)`候选质量通过但109.448996秒，未采用。本候选保留其65个源码文件及预测算术，计时入口仅把`input_start=6,output_end=6`改为`input_start=1,output_end=11`。选择input_blocks.1–11、middle、output_blocks.0–10共23个父块，复用深层分支；浅层、head与原采样更新仍实时执行。

两阶段都保留全部50次DDIM更新，在零起始位置`[20,40)`内交替完整刷新和预测，每轮每阶段10次预测、40次刷新。预测使用最近两次完整刷新及真实位置间距：`latest + (latest - previous) * alpha`，预测结果不进入刷新历史。首次alpha为1，后续为0.5；总直方图为16／144。它是近似特征预测，已通过本次完整视频PSNR，不能宣称与原路径数值相同，也不能继承到其他任务。

原63文件分片数学辅助模块、计时起点、profile写入、原采样参数保持不变。四个图分别为decision/world的完整与复用图；完整图各33个汇合，复用图各3个，两个阶段各36条、总72条捕获记录。每阶段400次调用、80次预测，两rank均核对。保留严格双GPU入场、双锁和全程外来上下文守护。

## 复现和证据

- [冻结65源码、计时资源及守护脚本](../experiments/nonv2-late23-prediction65/full-r1)
- [完整分项log、JSON、两rank报告及遥测](../results/aggressive-20261029/nonv2-late23-prediction65/full-r1)
- [公开入口CPU精确重建](../results/aggressive-20261029/nonv2-late23-prediction65/public-repro-cpu-r1/summary.json)
- [官方103视频汇总与脚本](../results/aggressive-20261029/nonv2-late23-prediction65/official103)
- [采用记录与双文件SHA](../experiments/nonv2-late23-prediction65/accepted-outcome.json)
- [旧755.093960秒材料](../results/history/official-before-late23-prediction65)

```bash
python reproduce/verify_nonv2_late23_prediction65.py
~/.venv/bin/python reproduce/run_nonv2_late23_prediction65.py \
  --gpus 1,3 --workdir /path/on/private-tmpfs/late23 \
  --storage-config /path/to/qualified-storage.json \
  --cache-dir /path/to/qualified-FFN-cache \
  --prediction-cache-parent /path/on/executable-filesystem
```

公开入口已在比赛机禁用CUDA，逐文件重建实际计时65源码及资源；1200次玩具执行器调用、3600个仿射预测输出、23块与3次汇合覆盖、预测区间以及环境和Triton缓存类恢复均通过。完整图重放自检覆盖FP16／FP32、直接／残差预测及尾块和修改输入后的重放。CPU检查与图自检均不能代替完整视频质量校验。

152次遥测中，两卡峰值显存均31788MiB，最高温72／58°C；利用率至少50%的样本中SM频率分别1320–1410／1290–1410MHz。未据此判断温度对耗时差异的因果关系。
