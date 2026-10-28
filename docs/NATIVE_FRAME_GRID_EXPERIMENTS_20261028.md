# 保留原生卷积描述符的帧分片准备

此前直接把 ResBlock 空间卷积 batch 从 16 改为 8，会改变 cuDNN 算法及原生滤波器缓存覆盖。新的实验保留完整 N16 形状、strides、卷积参数和原生 Tensor Core 算法，把其中一个 rank 应处理的八帧放在前八帧，其余位置填零。目标是之后在双卡中分别计算两半帧，汇合完整 16 帧后再执行原时间卷积。

以下算子结果来自合成单卡实验，不能把省去填零位置的时间直接作为完整视频加速。真实模型诊断、完整 V2 和官方 103 视频复核现已通过，分开记录在后文；当前采用的完整 V2 为 **287.745486 秒**。

## 36 组只读检查

六种 CO/CI/H/W 几何、NCHW／NC-permuted／channels-last 三种布局、两个 rank 的半帧共 **36 组**，填充后的前八帧与原完整 16 帧中对应的那一半全部逐位一致。捕获图的完整 16 帧输出也与填充后的 eager 输出逐位一致。NC-permuted 布局保留 `stride(C)=16*H*W`，没有改成 batch8 的物理描述符。

原 `eligible_plan` 在 NCHW 与 NC-permuted 的 **24 组**中找到已验证的滤波器变换／卷积节点；channels-last 没有通过这套原缓存识别，因此后续未修改其节点。首次检查未在捕获前启用 graph debug mode，句柄扩展拒绝读取；修正后 r2 完成全部检查，未放宽扩展或识别条件。

## 只改空间启动网格

后续在这 24 组中保留 N16 描述符及**全部 kernel 参数字节、函数、block、shared memory**，只缩小原生卷积的空间网格。XMMA 使用经过几何检查的 `grid_y`；已验证 CUTLASS 路径使用通过 tile／swizzle 一致性检查的 `grid_x`。保留 `grid_z`、split-K 分区和所有原生 memset 节点。

每组检查初始输入、输入修改、权重修改、权重清零、输入清零、全部恢复六个状态，**144 次前八帧比较全部逐位相同**。独立 eager 参考在捕获图私有内存池之外执行，状态之间不重放完整网格图，避免未计算的原生 workspace 因残留旧值而假通过。没有把猜测的 workspace 指针包装为张量，也没有导出参数字节、地址或操作数。每组结束时恢复原网格，再验证完整 16 帧逐位一致。

每组与原完整卷积做六对交替计时，包含原输入／滤波器变换与输出 epilogue，24 组均六对更快。下表是 NCHW、rank0 的结果；基准仍按 N16 描述符处理填充输入，候选只计算该 rank 的有效半帧，因此需要后续双卡汇合才能覆盖完整工作量。

| CO / CI / H / W | 原完整卷积 ms | 缩小网格 ms | 配对加速比中位数 |
|---|---:|---:|---:|
| 320 / 320 / 40 / 64 | 0.833178 | 0.531149 | 1.572× |
| 640 / 640 / 20 / 32 | 0.703334 | 0.418150 | 1.674× |
| 1280 / 1280 / 10 / 16 | 0.793037 | 0.496742 | 1.613× |
| 1280 / 1280 / 5 / 8 | 0.405658 | 0.276531 | 1.467× |
| 640 / 1920 / 20 / 32 | 1.944166 | 1.125171 | 1.728× |
| 320 / 960 / 40 / 64 | 2.242253 | 1.411123 | 1.591× |

两种布局、两个 rank 的单次节省范围约为 **0.128～0.843 ms**。这比仅缓存十处量化权重覆盖更大的单算子范围，但还不能推导完整模型节省多少秒。后续需要扩大到实际 ResBlock 的全部几何，再接入图中现有原生缓存 owner，保留其已验证的 filter 指针和恢复行为；同时重测 138+43+8 原 cache owner、全部 master 修改／恢复、通信成本和完整 V2 PSNR。

精确源码见 [`native-frame-grid-209`](../experiments/native-frame-grid-209)，全部输出误差、启动维度、配对样本和 source proof 见 [结果](../results/aggressive-20261028/native-frame-grid-209)。这些合成检查时的正式基线为 **289.486308 秒**，五项 **838.392232 秒**；后文完整结果已替换它。

## 补齐实际配置的全部空间卷积形状

使用原 209 文件对应的正式配置，在 meta device 构造视频主干；只将三个未检查的 head 工厂替换为 Identity，不加载 checkpoint、不初始化 CUDA。逐层检查 input/middle/output block 的通道及上下采样关系，得到 **22 个 ResBlock、14 种 3×3 卷积几何**。这一步证明配置的形状覆盖；实际运行的 strides 和 cuDNN 消费者仍须 GPU 检查。[拓扑及源码证明](../results/aggressive-20261028/native-frame-grid-209/topology-r1)

原六种以外的八种几何，再测两种布局和两个 rank，共 **32 组、192 次动态比较**，前八帧全部逐位一致；32 次原网格恢复后的完整 16 帧也全部逐位一致。与原实验合计 **56 组、336 次动态比较**，覆盖全部 14 种几何。新 32 组均六对交替计时更快，单算子节省中位数范围 **0.144～0.736 ms**，仍未计入分帧复制及双卡汇合。[扩展实验](../results/aggressive-20261028/native-frame-grid-209/extended-fixture-r2)

首次扩展在原先未支持的矩形 XMMA tile 检查处停止。后续只读检查发现 `tilesize256x128x32` 的实际网格为 `[ceil(CO/128), ceil(NHW/256), Z]`：第一个 tile 维度对应空间，第二个对应输出通道。只读程序完成了 32 组检查并写出进度，但汇总器字段名写错，进程失败；这些进度只用于识别内核，不作为干净完成的计时结果。修正网格规则后，上述 r2 独立完成全部动态检查和独占 GPU 守护。原来的扩展 r1 及其源码没有覆盖修改。

## 真实 ResBlock 接入方式

候选保留 N16 形状和 strides，把每个 rank 的八帧复制到前八帧。embedding Linear 仍执行完整 16 行或原单行广播，然后重排结果；skip 与 residual 相加后，只汇合两卡各自的前八帧，恢复完整 16 帧后再运行原时间卷积。生产配置不排除任何 ResBlock。

原缓存 owner 完成严格检查后，候选只修改 `cached_consumer.params` 的空间网格，保留它已经验证的缓存 filter 指针以及全部参数字节。`original_consumer` 保持原网格，供原关闭流程恢复。CPU 检查覆盖 120 个 rank／三种布局／FiLM／上下采样／embedding 广播／时间卷积边界组合，并另用模拟 owner 检查缓存指针与恢复网格；模拟测试不代表真实 GPU 缓存检查已经通过。[CPU 检查](../results/aggressive-20261028/native-frame-grid-209/resblock-cpu-r2)

真实模型 r1 在原缓存构建后，因为外层模型权重名有额外前缀，未匹配到 44 个空间卷积而主动停止，未获得有效模型计时。后续改为用权重对象身份匹配，并补上带完整前缀的 owner 测试；原 138+43+8 缓存和逐位数值门槛均保留。双卡诊断还需检查全部 Linear、norm 和 Conv2d master 修改／清零／恢复、三路输出、RNG、原基线恢复及完整前向通信耗时，通过后才有理由进入完整 V2。

r2 执行的精确源码、生成的真实模型 profile 和双卡守护脚本见 [模型候选](../experiments/native-frame-grid-209/resblock-model-r2)，CPU 预检、两张卡的完整报告及源码证明见 [候选证据](../results/aggressive-20261028/native-frame-grid-209/resblock-model-r2)。共享机器上遇到其他 GPU context 的轮次由原守护排除；最终通过的轮次全程干净。

## 真实 decision 模型 r2 通过

两张卡都捕获 **22 个 ResBlock、44 个空间卷积、14 种实际几何**。原 main cache 138 个、Conv1d cache 43 个、remaining cache 8 个的检查全部保留。修改输入、修改／恢复／清零／再次恢复全部 **570 个 Linear、782 个 norm、192 个 Conv2d FP32 master** 的六个状态中，每卡三路输出共 **18 路全部含 signed zero 逐位一致**，RNG 和原基线恢复也通过。

六对完整 decision 前向交替计时包含分帧复制、完整 embedding projection、全部 22 次汇合和原时间卷积：

| Rank | 基线中位数 ms | 候选中位数 ms | 配对差值中位数 ms |
|---|---:|---:|---:|
| 0 | 226.620415 | 217.300484 | 9.234940 |
| 1 | 223.506943 | 217.264641 | 6.184959 |

每个 rank 的六对都更快。最后一列先逐对相减再取中位数，不能用前两列之差替代；两个 rank 的差值也不能相加。该基线为两卡分别执行原单卡 decision 算法，**尚未启用当前正式版的空间 Transformer 并行**，因此这些数字还不能代表对正式 289.486308 秒 V2 的增益。

## 对当前并行算法的增量测试也通过

同进程基线和候选均启用正式版已有并行算法，只在候选额外启用上述 ResBlock 分片。两套捕获图各包含 16 个空间 Transformer，所以报告中空间捕获总数为 32；仅候选额外捕获 22 个 ResBlock、44 个 native consumer。两卡六个动态状态的全部 18 路输出、全部 master 修改、RNG、基线恢复和原缓存检查均通过。

| Rank | 已有并行算法中位数 ms | 加卷积分片中位数 ms | 配对差值中位数 ms |
|---|---:|---:|---:|
| 0 | 207.526398 | 198.218246 | 9.342461 |
| 1 | 207.588348 | 198.220291 | 9.372673 |

每卡六对均更快；两个 rank 的差值不能相加。这里测的是完整 decision 前向，包含已有空间汇合、新增 22 次汇合及所有分帧复制。它提供了对当前并行算法的直接增量证据，但还没有包含整个评测进程的启动、模型加载、视频编码和退出成本。[精确源码](../experiments/native-frame-grid-209/resblock-incremental-model-r1) · [全部配对样本和动态检查](../results/aggressive-20261028/native-frame-grid-209/resblock-incremental-model-r1)

## 218 文件计时入口

在已验证的 215 文件 overlay 上增加三个 helper：空间网格规划函数、原生 ResBlock 分片、与已有并行算法组合的控制器。原 209 个文件及 110 个 JIT 算术主体不变，评测入口仍是相同的四个可逆生命周期插入；初始化、额外缓存／网格设置、通信和 helper 退出都处于原进程计时之内。关闭路径保留原生原网格恢复。

一轮端到端预检已完成，两张卡各生成三个视频，所有哈希相同，阶段调用和捕获计数通过。预检不是正式 V2 成绩；后续完整运行结果列在下节。

冻结的 218 文件生成器及源码 overlay 见 [计时入口](../experiments/native-frame-grid-209/production-smoke-r1)，计时证明、两卡控制器和监控记录见 [预检证据](../results/aggressive-20261028/native-frame-grid-209/production-smoke-r1)。生成器从原评测入口逐字节重建四个插入并检查 SHA，不提交带继承尾随空格的整份评测脚本。

## 完整 V2 与正式材料已采用

完整运行测得 **287.745486 秒 / PSNR 29.675324786 dB**，11 轮、两个阶段各 550 步、23 个视频全部完成。23 个视频与原始单卡生产基线及另一 rank 的 SHA256 全部相同。相对上一正式 289.486308 秒，净省 **1.740822 秒**。随后用官方工具重新检查五项的 **49 轮、103 个视频**，五项合计 **836.651410 秒**，本机 `GPU/` 的原始 log／JSON 已同步，未自动上传网站。

| 测量 | 前一正式双卡版本 | 本次 218 文件版本 | 差值 |
|---|---:|---:|---:|
| decision generation CUDA 秒 | 127.778013 | 123.523878 | −4.254135 |
| world generation CUDA 秒 | 150.847363 | 151.679250 | +0.831887 |
| process wall 秒 | 289.486308 | 287.745486 | −1.740822 |

进程总时间扣除上述两项 CUDA 时间后的差额增加约 1.681426 秒。这是两次独立完整进程，不能把全部差异归因于代码改动；也不能把前向微测收益直接当成完整进程收益。

采用轮次全程通过独占 GPU context 检查，物理 GPU 2、3 各有 447 个遥测样本，峰值显存各 **74,168 MiB**，最高温度 **74°C / 59°C**。利用率至少 90% 时，频率范围分别为 **1305–1410 / 1290–1410 MHz**，中位数均为 **1410 MHz**。前面被其他任务打断的尝试已排除；它们没有提供此次有效计时。没有调整频率或功耗。

已采用的固定入口与源码见 [`nccl-native-frame-grid-209`](../experiments/nccl-native-frame-grid-209)，完整单项 log／JSON、两卡控制器、23 视频比较、遥测及官方 103 视频汇总证据见 [完整结果](../results/aggressive-20261028/nccl-native-frame-grid-full-r1)。默认复现命令已切换到此 218 文件版本，历史 215 文件版本保留。

## 下一项：world 的 FP16 Winograd 分帧原型

新增 [两个实验 GEMM](../experiments/world-native-frame-prefix-209/source-r1)，分别对应原 packed 和 Kahan 三乘积路径。保留原 `COLS` 的全部内存步长、mask、K 分块、Half 转换、乘积和 FP32 累加，仅用 `ACTIVE_COLS=COLS/2` 改变 CTA 行列映射。两个函数的 AST 逆变换可精确恢复原函数，14 种生产配置的 CTA 覆盖检查通过。未对齐的末 tile 可能写入有效前缀之后的少量列，不能声称所有 store 严格停在半矩阵边界。

从本次实际生产日志提取的 23 个 world SGEMM owner 中，**21 个属于已检查的 ResBlock，2 个属于外层 Upsample**。后者没有在同一位置进行帧重排／汇合，后续必须按原权重对象绑定筛选，不能把 23 个 owner 无条件全部分片。[源码及覆盖证明](../results/aggressive-20261028/world-native-frame-prefix-209/source-r1)

上述 source-r1 只有 CPU 源码与映射证明。后续独立矩阵检查已通过，但原生卷积的连续前缀映射失败，详见下节；未用于上述正式 218 文件版本。

## World 矩阵检查通过，原生卷积前缀假设失败

GPU 矩阵 r2 对全部 14 种 packed／Kahan 配置各测试初始、修改输入、修改权重、权重清零、输入清零和恢复，共 **84 次前半列比较，全部包含 signed zero 逐位一致**。输出 canary、未写尾部、RNG 和 14 个新 CUBIN 的实际 graph 捕获／指针 ABI 检查通过。仅修改 CTA 的列覆盖，没有减少每个有效输出的三乘积或补偿精度。[精确执行源码](../experiments/world-native-frame-prefix-209/matrix-r2) · [完整样本与遥测](../results/aggressive-20261028/world-native-frame-prefix-209/matrix-r2)

14 种配置的六对交替计时均更快，独立中位数之比约 **1.41～1.98×**。例如 packed `[320,2560,320]` 从约 0.750763 ms 降到 0.389291 ms，Kahan `[640,2560,640]` 从约 3.310251 ms 降到 1.672021 ms。这只覆盖一半矩阵工作量，未包含完整卷积、另一半帧、复制或通信；不能作为整模型加速。

首次 r1 在数值测试前因 `/tmp` 的 noexec 属性导致 Triton launcher 无法加载。r2 为 launcher 新建私有 home 目录，大型编译中间文件仍在独立 `/tmp`，原正式缓存不变。采用轮次通过全程 GPU context 检查，原 209 文件在前后均重新核验。

随后独立卷积 fixture 保留原生输入／输出变换、滤波器缓存及 bias，并在每个动态状态前毒化完整 GEMM 输出 workspace，防止旧值假通过。首个 NCHW `[320,2560,320]` 用例即失败，未获得有效卷积计时。[失败的精确源码](../experiments/world-native-frame-prefix-209/conv-prefix-failed-r1)与[原始日志](../results/aggressive-20261028/world-native-frame-prefix-209/conv-prefix-failed-r1)单独保留。**矩阵的连续前半列不能直接当作视频前八帧**；后续必须重新验证实际列与帧的映射。新 owner 的整模型接入、完整 V2 与质量均未验证，正式成绩保持 **287.745486 秒**。

## World 交错帧映射通过原生卷积检查

将列映射改为 `physical_col=(logical_col//8)*16+logical_col%8`，在每组 16 列中取前八帧。r2 独立完成 **14 种几何、两种布局、两个 rank、六个动态状态，共 336 次卷积输出逐位检查**，全部包含 signed zero 相同；另有 84 次矩阵检查和 14 个新捕获图／CUBIN 指针 ABI 证明。每个动态状态均先确认完整 numerical GEMM 的填充输入结果与完整参考的对应半帧逐位相同，再毒化候选 GEMM 输出 workspace 并运行交错映射。原 numerical 全16帧恢复、原生 FP32 全16帧恢复、RNG 和缓存 canary 检查均通过。[精确源码及复现](../experiments/world-native-frame-interleaved-209) · [全部输出与配对样本](../results/aggressive-20261028/world-native-frame-interleaved-209/conv-r2)

原始输入／输出变换、bias 和权重 cache 刷新均进入卷积计时；不包含双卡复制、汇合或完整模型。**11/14 种几何**在两种布局和两个 rank 中均六对更快。以下为 NCHW、rank0 的配对差值中位数，两个 rank 的差值不能相加：

| 路径／几何 CO,COLS,K | 每次节省 ms |
|---|---:|
| packed 320,2560,960 | +0.212480 |
| Kahan 640,640,1280 | +0.315051 |
| Kahan 640,2560,640 | +0.690347 |
| packed 1280,192,1280 | **−0.105131** |
| packed 1280,192,2560 | **−0.190635** |
| packed 1280,640,1280 | **−0.013483** |

三种 packed 几何退化，不能统一采用。此前连续前缀矩阵的 1.41～1.98× 加速不适用于这条原生交错列路径。PTX 检查显示交错访问中更多加载被编译成标量操作；随后为每组连续八列增加 alignment／contiguity 提示，80 组 CPU 索引检查通过，新的独立 vector8 试验也完成相同的 336+84 次逐位检查。但仍是同样 11/14 种几何更快，未证明提示带来有意义的额外收益。[vector8 精确源码](../experiments/world-native-frame-interleaved-209/vector8-r1) · [完整结果](../results/aggressive-20261028/world-native-frame-interleaved-209/vector8-r1)

交错映射的第一次 harness 在进入 Kahan 路径时因 owner 没有 `replay()` 方法而停止，没有提供最终有效计时。r2 按原模型语义先刷新 owner，再重放其 graph；未放宽数值门槛或改动生产 owner。两轮采用的结果均有独占 GPU context 全程守护，原 209 文件前后核验一致。

以上仅提供合成卷积证据，没有真实 world ResBlock 的 GPU 执行、双卡通信、完整模型或 V2 PSNR 结果。后续集成必须先通过原 23+45 owner 与旧 CUBIN SHA 门槛，再按权重对象筛选 21 个 ResBlock owner，保留另外两个 Upsample owner；不能直接替换所有 23 个 SGEMM。当前正式 V2 仍为 **287.745486 秒**，五项 **836.651410 秒**。

## 真实 world 集成候选：CPU 通过，GPU 结果待验证

初次真实模型候选已实现并通过 **120 组 CPU 边界检查**，另用实际 23 个 owner 的 meta 权重对象验证：额外模型名前缀不影响 21 个目标 SGEMM 的身份绑定，两个外层 Upsample 不参与分帧。实际 21 个目标属于 **12 个 ResBlock**；其余十个没有目标 SGEMM，保持原执行，避免增加无收益的汇合。该选择覆盖全部 21 个已检查目标，并非减少它们的覆盖范围。[冻结源码](../experiments/world-native-frame-interleaved-209/model-r1) · [CPU 证据](../results/aggressive-20261028/world-native-frame-interleaved-209/model-r1-preflight)

基准与候选都启用当前 world 空间／时间并行；候选额外执行这 12 个块的 N16 填充与前八帧重排，在原完整时间卷积前汇合。保留原 23+45 owner 和旧 Kahan CUBIN 门槛，新 kernel 使用独立缓存，编译与实际捕获图参数检查后恢复原缓存路径。新的捕获见证使用完整、类型正确的 Torch 缓冲，**不重放未初始化缓冲，不声称它提供数值结果**。真正的数值门槛仍是后续六个完整模型动态状态、570／782／192 master 修改、三路逐位输出、RNG 和原图恢复。

这份快照还没有完成真实 GPU 模型检查或计时。首次实际模型任务现已退出：尝试 1 因出现其他 CUDA context 被排除；尝试 2 在 rank 启动时检查 `own` NVML context 数量为一的断言失败，父进程退出。没有通过后续完整模型门槛，也没有可用配对计时。该失败发生在模型数值检查之前，不能判断候选算术是否正确或是否更快。[终态日志和状态](../results/aggressive-20261028/world-native-frame-interleaved-209/model-r1-terminal)。等待进程仍按原规则只在两卡满足空闲条件、取得两把锁并再次核验后创建 CUDA context；正式材料不变。后续新尝试需要先完善上下文初始化失败的分类，不能将该次退出当作数值失败或成功。

等待期间另在 CPU 上完成了实际使用的 **9 种新 kernel** 的 SM80 静态编译，`CUDA_VISIBLE_DEVICES=''`、Torch CUDA 始终未初始化，产物写入新 kernel 专用缓存。初次尝试已生成第一个 CUBIN，但在 Triton2.3.1 随后构造运行时包装器时失败；该包装器要求无设备环境的 `UnsupportedDriver.binary_ext`。r2 在独立 CPU 进程中将最终返回对象换为编译文件读取器，保留全部编译阶段与 cache key，九种产物均完成。GPU worker 不导入此 helper，原 CUBIN 门槛与新捕获图 ABI 检查保留。该操作不提供 GPU 执行、整模型速度或质量证明，也不是冷编译耗时基准。[精确源码和说明](../experiments/world-native-frame-interleaved-209/offline-compile-r2) · [九种产物 SHA 与日志](../results/aggressive-20261028/world-native-frame-interleaved-209/offline-compile-r2) · [首次包装器失败](../results/aggressive-20261028/world-native-frame-interleaved-209/offline-compile-r1-failed)

## 后续接入：上下文分类与新 kernel 缓存隔离

r2 为 rank 启动时上下文数量非一的情况补充原因，并让重试器同时检查父日志与 `ranks.log`。上下文不明或其他进程进入的整个尝试被排除；模型数值失败仍立即终止。四个 CPU 子进程故障场景已执行真实重试脚本，验证缺失／多个／外来 context 会重试，数值错误不会重试；严格空闲与恰好一个 rank context 的断言不变。[源码与 CPU 检查](../experiments/world-native-frame-interleaved-209/startup-guard-r2)。

实际 r2 随后在新 kernel 的 `CompiledKernel` 构造阶段退出，二进制被当文本读取，在第 18 字节发生 `UnicodeDecodeError`，没有完整模型数值或耗时结果。[r2 冻结源码](../experiments/world-native-frame-interleaved-209/model-r2) · [终态日志](../results/aggressive-20261028/world-native-frame-interleaved-209/model-r2-terminal)。源码中 `FileCacheManager.put` 会先生成 `<stem>.cubin.tmp.pid_*`，而 `CompiledKernel` 使用 `<stem>.*` 读取文件，只把后缀恰好 `.cubin` 的文件当二进制。两个 rank 共用新缓存目录时可能互相看到临时二进制。纯 CPU 测试暂停真实 writer 的原子替换，再执行实际安装版本的 reader，复现第 18 字节解码错误；分 rank 目录时不再出现。仅替换 launcher 导入与 driver 描述对象，不加载、执行 GPU 二进制或修改原缓存。[复现源码](../experiments/world-native-frame-interleaved-209/cache-temp-race-cpu-r1) · [CPU 结果](../results/aggressive-20261028/world-native-frame-interleaved-209/cache-temp-race-cpu-r1)。这与 GPU 异常一致，是该次故障原因的推断；静态预编译记录也不等于实际运行的全部 specialization 已命中缓存。

r3 因此只把新增 kernel 缓存拆为 `PREFIX_CACHE_DIR/rank-0` 与 `rank-1`，退出作用域后恢复原缓存路径；不安装全局 custom manager，不改编译器、旧 CUBIN 门槛或候选算术。120 项 CPU 边界与 21 owner／12 block 身份检查再次通过，目前有界 GPU2/3 任务仍在进行，完整模型动态门槛、配对耗时和 V2 质量均未完成。[r3 源码](../experiments/world-native-frame-interleaved-209/model-r3) · [静态 CPU 证据](../results/aggressive-20261028/world-native-frame-interleaved-209/model-r3-preflight)。正式 V2 仍为 **287.745486 秒**，五项 **836.651410 秒**。
