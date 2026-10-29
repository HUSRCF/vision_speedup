# G1 双卡：时间 attention 共用已有 FFN 汇合

新67文件完整 G1 为 **158.629774 秒 / PSNR 26.107156332 dB**。23个视频与正式版本、另一rank的SHA256全部相同；同GPU0随后执行的原59文件单卡为 **158.749762秒**。完整进程只少 **0.119988秒**，目前仅一次比较，尚未证明稳定收益，**没有采用**。正式G1仍为160.897537秒，五项仍为771.314275秒，本机`/GPU`双文件保持原样。

## 扩大范围，保留通信次数

此前66文件已将75次空间attention／阶段与15个空间FFN共用汇合。本轮将 **16个时间内部块、32次融合时间attention／阶段**纳入已有时间FFN汇合。每次denoiser仍共 **31次all-gather**：15个空间块末尾、16个时间内部块末尾。

时间分片沿独立的空间位置batch切分，每个位置保留完整16帧。完整5D GroupNorm在分片之前执行；原生GEMM完整形状、描述符、K次序，以及时间attention原编译CUBIN、constexpr、舍入均保留。最小不满足原生FFN网格条件的块维持完整计算。没有新增native参数扫描；时间attention接入仅使用普通Torch张量及既有Triton调用参数。

G1时间attention实际走`triton_temporal_attention._temporal_attention`。最初r4错按xFormers入口接入，decision在覆盖数量断言处终止；world在模型初始化阶段SIGSEGV，未取得模型结果，原因未确定。修正入口后的r5才进入下述测量。失败源码和记录均保留。

## 两阶段真实模型资格

保留官方frame_stride=6、FP32 feedback head。954个相关master参数经过输入变化、参数变化、清零和恢复等六状态检查，两rank三个输出均逐位一致，RNG和恢复检查通过。

每阶段在同一进程、同GPU0/1交替六对，每种条件七个样本取中位数。保守收益取每对“最快baseline rank减最慢candidate rank”；两个rank节省不能相加。

| 阶段 | rank0中位节省 | rank1中位节省 | 保守中位节省 | 六对均为正 |
|---|---:|---:|---:|---|
| Decision | 7.717888ms | 9.038334ms | 7.708672ms | 是 |
| World | 7.348221ms | 9.206272ms | 7.335423ms | 是 |

这些是包含通信的完整模型前向，**不含完整进程加载和首次捕获**，不能直接当作完整单项节省。

## 完整运行及开销

第一次完整运行在冷启动时尚未建立原时间attention的CUBIN指纹，断言终止，没有视频成绩。修复为在首次计时内eager warm-up建立原内核指纹，再分片并核对同一CUBIN。r2完整完成11轮、每阶段每轮50步、23视频。两卡整次空闲锁／外来context检查通过。

| 部分 | 原59文件单卡 | 新67文件双卡 | 单卡减双卡 |
|---|---:|---:|---:|
| Decision生成墙钟 | 75.415123s | 72.942281s | 2.472842s |
| World生成墙钟 | 74.067312s | 71.658552s | 2.408760s |
| 模型搬到GPU | 3.719234s | 5.733544s | −2.014310s |
| **完整原process timer** | **158.749762s** | **158.629774s** | **0.119988s** |

生成阶段包括首次捕获和绑定。第二进程、模型加载、NCCL初始化、指纹热身、图捕获、绑定、额外首次replay、视频和退出都计入原process timer。一次完整比较与六对模型测量分别保留。

两卡峰值显存均45,300MiB，最高温度60／71°C，最高SM时钟均1410MHz；有GPU利用率时的采样时钟中位1395／1380MHz。没有调整时钟，原始遥测保留。

当前额外研究只由rank0从主机搬运权重，再广播给rank1，以降低已测得的加载开销。需独立验证位值、dtype、stride、对齐、参数别名和完整视频，尚未成为本版本结果。

## 复现

`--prepare-only`已在比赛机仅依赖公开源码重建全部67文件与计时资源，逐一SHA256一致，不运行GPU。完整执行还需权重、输入缓存和已验证的私有tmpfs NCCL适配器。

```bash
python3 reproduce/run_g1_native_shared67.py --gpus 0,1 \
  --workdir /var/tmp/g1-native-shared67 --prepare-only
python3 reproduce/verify_g1_native_shared67.py
```

源码及失败边界见[experiments](../experiments/half-native-shared67)，完整结果、单卡对照、六对样本及遥测见[results](../results/aggressive-20261029/half-native-shared67)。公开仓库不包含权重、输入或视频。没有新的官方103视频汇总，没有网站提交。
