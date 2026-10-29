# V2短区间两阶段预测：原209数学路径独立验证

此候选用于验证V2时间步复用的完整视频精度，先使用一张空闲A800；没有替换当前225文件双卡正式配置。保留全部11轮、每阶段50个DDIM更新、FP32 attention及原209数学辅助模块。两阶段都选择13个深层父块，在零起始位置`[20,40)`内每隔一步预测一次，共每阶段每轮10次；两次完整刷新历史及预测算术与前版一致。

此前两阶段预测有两次启动失败：第一次的专用Conv1d缓存要求完整图138份缓存／143个reader，部分图实际62／64；第二次关闭通用缓存，又触发原decision补偿数学路径的依赖检查。这次保留`UNIFOLM_NATIVE_GRAPH_WEIGHT_CACHE=1`，只在decision复用图的词法作用域内关闭`UNIFOLM_NATIVE_CONV1D_WEIGHT_CACHE`和`UNIFOLM_NATIVE_REMAINING_FILTER_CACHE`，包括异常路径均恢复原值。完整图的缓存设置、专用模块的节点数断言及全部数学辅助代码均未修改。World复用图继续使用原生head变换，保留之前的完整图head缓存。

新增模块为预测控制器与运行时；原209源码仅改变CUDA Graph共享池参数，共211文件。与前两阶段候选相比，计时入口只把预测窗口从`[10,40)`缩为`[20,40)`，计时起点和profile写入保持不变。

公开入口已在比赛机关闭CUDA精确重建实际计时源码及资源，3600个原预测CPU输出、显式阶段识别、两项专用缓存开关的正常／异常恢复均通过。另以12个input和12个output块的仿射模型执行全部1100次调用，核对13块选择、220次预测、真实间距系数、浅层与head持续执行、RNG不变。CPU和图自检不能代替完整视频质量结论。

```bash
python reproduce/verify_v2_late_prediction211.py
~/.venv/bin/python reproduce/run_v2_prediction211.py \
  --trial both-late-r3 --gpu 1 --workdir /path/to/new/workdir \
  --native-handles /path/to/qualified/team07_cuda_graph_handles.so
```

- [冻结源码及实际计时资源](../experiments/v2-prediction211/both-late-r3)
- [公开CPU重建、执行日志和源码SHA](../results/aggressive-20261029/v2-prediction211/public-late-repro-cpu-r3)
- [13块完整GPU日志、JSON及遥测](../results/aggressive-20261029/v2-prediction211/both-late-r3)

13块候选已在GPU1完整执行11轮23视频，**310.127282秒／PSNR18.889815363dB**，严格守护clean，四图、两阶段各550调用及110次预测全部通过工作量检查，但未通过25dB质量门槛，未采用。890条遥测中峰值显存78856MiB、最高温72°C，利用率至少50%的样本中SM频率1305–1410MHz。

逐视频证据中，decision／world首轮分别34.5142／33.1335dB，第二轮降至19.7227／20.5262dB；这一现象与闭环误差放大一致，但尚未隔离action／state head及视频反馈各自的贡献。不能从首轮或单步误差推断完整质量。

继续测试的7块候选`both-late7-r4`只把深层范围缩为`input_start=9,output_end=3`，同样保留窗口和全部50步。其公开源码精确重建、3600个CPU预测输出与实际7块／1100调用／220预测的仿射检查已通过；完整GPU2结果为 **325.592034秒／PSNR20.224752075dB**，11轮23视频、四图、每阶段550调用／110预测完整通过工作量检查，但仍未通过25dB，未采用。完整[日志、JSON与遥测](../results/aggressive-20261029/v2-prediction211/both-late7-r4)已保留。两次候选在不同GPU上运行，没有同期同卡完整AB，不能用进程差值解释纯算子性能。

7块候选934条遥测中，峰值显存81014MiB，最高温73°C，利用率至少50%的样本中SM频率1305–1410MHz。较小的复用范围保留更多复用图计算，不能假设其图内存一定更小；新增图或历史缓冲需要重新检查显存余量。
