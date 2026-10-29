# 已采用：清理铅笔末端5个残差块双卡分工

清理铅笔正式耗时由95.640695降至 **92.232482秒**，PSNR保持26.798606240dB。新的官方103视频汇总accepted，全部103视频与上一版的SHA一致，本机`GPU/runtime.log,result.json`已更新，网站未上传。五项分项合计从738.805454降至 **735.397241秒**，少3.408213秒；这是各次完整分项汇总，不是一次连续五项运行。

## 改了什么

保留当前65文件双卡Transformer和23块`[20,40)`时间步预测。在以下5个高分辨率ResBlock内，每张卡只算自己的8帧空间部分：

```text
output_blocks.7.0
output_blocks.8.0
output_blocks.9.0
output_blocks.10.0
output_blocks.11.0
```

局部执行原二维归一化／卷积、时间嵌入和残差，然后汇合为完整16帧，再调用原时间卷积及完整五维GroupNorm。保留原权重、精度、每轮两阶段50步和全部8轮。只新增一个普通Torch模块并扩展协调器的按图模块清单校验，共66个推理源码文件；原65中其余64个模块和计时入口字节不变，没有新增原生CUDA参数或地址检查。

完整图共38次汇合：原33次加5次；复用图共4次：原3次加末端1次。四张图分别检查实际模块清单；两阶段各400调用，其中各320刷新／80预测。

## 为什么只选5块

此前[22块全量分帧](NONV2_RESBLOCK_FRAMES72_20261029.md)虽然前向各省8.7／9.0ms，完整进程反而比同卡对照慢2.983799秒。本版保持同一分片原语，只改变启用模块清单，将新增汇合从22次减到5次。

真实模型中，decision／world两阶段、两rank各六组配对均更快，保守rank中位节省约3.75／3.45ms。816参数参与输入／权重修改、清零／恢复等六状态检查，两阶段三个输出全部逐位一致，RNG未改变。每rank新增发送片段合计50MiB／前向。cuDNN batch改变的数值性质不推广到任意输入，完整视频仍独立验证。

## 完整进程对照

同GPU0／1按以下顺序运行：

| 完整运行 | 时间 | PSNR | 与原正式视频SHA |
|---|---:|---:|---|
| 原65前置对照 | 94.368267 s | 26.798606 dB | 17个全同 |
| 新66末端5块 | **92.232482 s** | 26.798606 dB | 17个全同 |
| 原65后置对照 | 93.983744 s | 26.798606 dB | 17个全同 |

新版本分别快2.135785／1.751262秒。每次均完整8轮／17视频、两rank视频一致，保留加载、图捕获、缓存、通信和退出的原process timer。这里只有一个candidate和两个前后对照，不能视为两组独立重复配对；前置对照期间另一GPU对还做过模型诊断，后置对照时没有其他本队GPU工作。与历史正式95.640695秒的3.408213秒差值另作历史比较。

## 证据与复现

- [源码、guard、精确文件SHA及采用说明](../experiments/nonv2-resblock-tail73/)
- [模型测量、完整log／JSON、前后对照与GPU遥测](../results/aggressive-20261029/nonv2-resblock-tail73/)
- [官方103视频原始验收](../results/aggressive-20261029/nonv2-resblock-tail73/official103/validated/)
- [上一版正式双文件](../results/history/official-before-resblock-tail66/)

```bash
python reproduce/verify_snapshot.py
~/.venv/bin/python reproduce/run_nonv2_resblock_tail66_full.py \
  --gpus 0,1 --workdir /var/tmp/team07-review-tail66 \
  --storage-config /var/tmp/team07-review-storage/storage-config.json \
  --cache-dir /path/to/executable/FFN-shape-cache
```

模型诊断入口为`reproduce/run_nonv2_resblock_tail73.py`。两个公开入口均已在比赛机禁用CUDA精确重建并完成CPU检查。复现仍需原比赛资源及合格的私有tmpfs适配器；不分发权重、输入、特征或视频。冻结trial／summary中的未采用标记记录准备和首次验收时状态，当前采用状态以`accepted-outcome.json`及官方103材料为准。
