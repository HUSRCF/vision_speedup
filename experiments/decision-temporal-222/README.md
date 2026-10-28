# 已采用223：保留vendor GEMM形状的decision时间块分片

原222双卡版本的decision时间Transformer内部仍重复执行。候选在完整5D GroupNorm之后，沿独立空间位置划分17个时间块（含init_attn），每个位置保留全部16时间步。原Python张量形状、vendor GEMM描述符及数学函数体不变，仅对已支持的Triton启动调整本地行范围，并在内部BasicTransformerBlock残差完成后增加一次all-gather。新增17次汇合、QKV回写和缓存成本均纳入整网计时。

当前222基线与候选的两个rank均通过六种动态输入／参数状态的三路输出逐位检查，覆盖570个Linear master、782个norm参数和192个Conv2d参数的修改、清零及恢复；RNG不变。合成master修改后显式失效条件缓存key，保留原输出storage，防止过期缓存掩盖检查。两个rank的六对完整decision前向均更快，中位净省 **16.610313／16.626686 ms**，由约192ms降至175ms。两个rank的节省不能相加。

按550次前向外推约9.14秒，仅为模型预算。完整同GPU0/1、同guard／私有tmpfs的222／223对照为 **270.119302 → 265.216325秒**，净省 **4.902977秒**；decision生成CUDA119.431→114.393秒，world138.396→138.467秒。实际完整收益较小，不能用模型外推替代进程结果。这是一对完整运行，六对证据来自独立模型检查。

候选完整11轮／23视频、两rank及原正式视频SHA一致，官方PSNR **29.675324786dB**。新103视频官方汇总accepted，五项 **778.585395秒**；根 `results` 与本机GPU双文件已更新，网站未上传。相对历史正式V2少3.627904秒，原782.213299秒双文件保留在 `results/history/official-other63-before-decision223`。其余四项耗时和质量均不变。

`model-r1`为冻结的精确诊断源码，baseline文件清单已核对为222。脚本中保留了继承诊断框架的旧标题和`INITIAL_BOUNDARY221_MODEL`输出标签，它们不代表实际加载的基线版本；应以`baseline-production-identity.json`及源码哈希为准。正式集成仅新增生产report／capture计数适配，不更改这个已检查路由的计算路径。

原home镜像只剩151552字节，因此新形状缓存位于容器根文件系统上`/run/lock`下的0700私有目录；CPU编译和dlopen检查通过。旧temporal缓存逐字节复制后再编译新变体，原缓存和CUBIN SHA门槛不变；既有固定哈希的原内核仍使用原warm缓存。该路径只解决新实验编译存储，模型初始化和编译仍在原process timer内。

默认 `reproduce/run_v2_parallel.py` 现转交冻结223入口；独立入口为 `reproduce/run_v2_decision223.py`。两个入口均已在比赛机通过CPU准备检查，223文件和计时资源与实际运行完全一致。需要已验证的私有tmpfs存储配置与可执行缓存：

```bash
INIT_FFN_CACHE_DIR=/path/to/initial-cache \
TEMPORAL_BLOCK_CACHE_DIR=/path/to/temporal-cache \
~/.venv/bin/python reproduce/run_v2_parallel.py --gpus 0,1 \
  --workdir /var/tmp/team07-review223 \
  --storage-config /var/tmp/team07-review-storage/storage-config.json \
  --decision-cache-dir /path/to/decision-cache
```

`--prepare-only`只重建源码，`--smoke`仅一轮，不能作为成绩。冷缓存编译仍在原进程计时内，可能更慢。旧222可用 `--variant both-spatial-temporal-resblock-grid-native-attention-initial-ffn-temporal-block-initial-block` 重建。单项shell因另外四项缺席退出1；选中V2的推理、profile和PSNR全部通过。五项提交请使用重新验收后的根双文件。

计时证明元数据勘误：冻结的`timing-proof.json`中`no_new_collectives=true`由父222证明继承，未正确更新，**不能用于描述223**。本候选明确新增每次decision前向17次汇合，原始证明的`new_collectives_per_decision_forward=17`与实际17个controller记录一致。`full-candidate-r1/timing-proof-erratum.json`给出更正；原始运行的证明字节和耗时均保留，没有改写成另一份运行证据。
