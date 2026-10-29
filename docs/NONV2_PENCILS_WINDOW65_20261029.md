# 清理铅笔：双卡短区间时间步预测

以下为13块候选完成时的历史记录；当前已采用23块方案，见[新记录](NONV2_LATE23_PREDICTION65_20261029.md)。当时五项正式成绩均已采用双卡，合计755.093960秒。非V2四项分别为Z1 145.327923秒、双臂90.557285秒、清理铅笔105.949243秒、G1 155.314304秒。

本轮继续优化清理铅笔。此前13块预测使用零起始位置区间`[10,40)`，完整8轮17视频为100.661865秒，但PSNR仅19.202957201dB，不能采用。本候选只把区间缩至`[20,40)`；每阶段每轮预测10次，其余40次完整刷新。仍执行全部8轮、每阶段50个DDIM更新，不以减少采样步数换取速度。

保留两次完整刷新历史的原预测算术、13块范围、原63文件数学路径和双卡分片。预测历史不包含预测输出。相对于前一65文件候选，唯一变化的源码模块是最终工作量断言（15改10）；计时入口的配置参数start从10改20。完整图仍33次汇合，预测图21次；四图共108条捕获记录。需要完整GPU运行实际确认，不能从CPU结果推断完整视频PSNR。

冻结源码、计时资源和守护脚本见[`full-r1`](../experiments/nonv2-pencils-window65/full-r1)。公开入口已在比赛机关闭CUDA，重建65个源码文件并逐文件匹配计时清单及资源；1200次玩具模型调用、3600个预测输出、区间与缓存作用域恢复检查通过。

```bash
python reproduce/verify_nonv2_pencils_window65.py
python reproduce/run_nonv2_pencils_window65.py --gpus 1,3 --workdir /path/on/private-tmpfs/trial --storage-config /path/to/qualified-storage.json --cache-dir /path/to/qualified-FFN-cache --prediction-cache-parent /path/on/executable-filesystem
```

已在GPU1／2完整完成8轮17视频：**109.448996秒／PSNR35.389549586dB**，两rank视频SHA一致。每阶段400次调用，其中80次预测；四图和108条捕获记录核对通过。质量通过，但高于历史正式105.949243秒，未采用；不同卡组、非同期运行，不能直接推导纯算子回退量。

164次遥测观测中，两卡峰值显存均31174MiB，最高温67／69°C；利用率至少50%的样本中SM频率范围分别1275–1410／1290–1410MHz。完整原始log、JSON和遥测见[`full-r1`](../results/aggressive-20261029/nonv2-pencils-window65/full-r1)。没有主张温度或频率是耗时差异的原因。

原等待GPU1／3的进程在进入计时前终止，日志保留；GPU2空闲后，以同一源码、未改动的双锁及全程外来上下文守护，在GPU1／2重新入场。终态见[`trial-status.json`](../experiments/nonv2-pencils-window65/trial-status.json)。正式五项与本机GPU双文件保持原成绩。
