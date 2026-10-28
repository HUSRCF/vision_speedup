# 推理复现

## 环境与资源

实际环境为 Linux x86-64、Python 3.10、Torch 2.3.1+cu121、Triton 2.3.1、A800 80 GB。当前最快 V2 使用两张卡，保留的单卡基线为 316.774396 秒。`environment/requirements-observed.txt` 是比赛环境的已安装包记录；`unifolm/pyproject.toml` 是上游项目依赖。请不要据此随意升级 Torch/Triton，原生图句柄和 kernel 参数布局均固定到该环境。

比赛共享路径：

```
/share/weights/UnifoLM-WMA-0-Dual/unifolm_wma_dual.ckpt
/share/problems/unifolm-world-model-action/examples/compute_benchmark/preprocessed_inputs
/share/huggingface/hub
/share/results/unifolm/fp32/reference/outputs
/share/tools/unifolm-psnr.py
```

本仓库不分发上述资源。赛外机器需取得授权资源、离线模型缓存与官方工具，并按 `unifolm/scripts/team07_env.sh` 调整路径。原脚本默认使用 `~/.venv` 和 `~/.tmp`。

## 检查与运行

当前双卡版本在仓库根目录使用比赛虚拟环境运行：

```bash
python3 reproduce/verify_snapshot.py
INIT_FFN_CACHE_DIR=/path/to/initial-cache \
TEMPORAL_BLOCK_CACHE_DIR=/path/to/temporal-cache \
~/.venv/bin/python reproduce/run_v2_parallel.py --gpus 2,3 \
  --workdir /var/tmp/unifolm-parallel-review-223 \
  --storage-config /var/tmp/team07-review-storage/storage-config.json \
  --decision-cache-dir /path/to/executable/decision-cache
```

`--workdir` 必须是新目录。默认重建固定223文件源码：两阶段空间块、原生attention、decision原生卷积分帧、world初始FFN、16个常规world时间块及初始时间块边界扩展，再分片17个decision时间块内的独立Triton工作。Decision保留vendor GEMM描述符，新增17次块末汇合。完整5D GroupNorm和16时间步保留，既有FFN汇合边界复用。守护程序等待两张空闲卡并检查整段GPU占用；模型导入、双模型传输、通信及helper退出仍计入原始进程耗时。已验证完整结果为265.216325秒、PSNR29.675324786dB。`--variant both-spatial-temporal-resblock-grid-native-attention-initial-ffn` 可重建旧220版本（277.436969秒）；其余历史variant继续保留。单轮 `--smoke` 仅供排查，不能作为成绩。

本轮同GPU0/1、同guard／适配器／存储的222／223完整对照为270.119302→265.216325秒，省4.902977秒；相对历史正式222少3.627904秒。23视频与原正式及另一rank的SHA一致，103视频官方汇总accepted，五项778.585395秒。六对模型检查的16.6ms净省不能代替完整进程差值。原222可通过`--variant both-spatial-temporal-resblock-grid-native-attention-initial-ffn-temporal-block-initial-block`重建。此前221／222完整对照272.043547→268.844229秒的历史证据继续保留。历史221可通过`--variant both-spatial-temporal-resblock-grid-native-attention-initial-ffn-temporal-block`重建。`--prepare-only`仅建立源码/guard，不使用GPU。

223完整守护需要上述私有tmpfs配置，先按下面命令构建。比赛机/dev/shm空间受限，使用已经通过CPU和双卡图内通信测试的[私有tmpfs适配器](../experiments/private-nccl-tmpfs)。它只调整新建NCCL控制文件的存储位置，完整对照使用相同设置。选择两个新目录：

```bash
~/.venv/bin/python experiments/private-nccl-tmpfs/build_and_check.py \
  --workdir /var/tmp/team07-review-storage \
  --exec-dir "$HOME/.tmp/team07-review-storage-exec"
~/.venv/bin/python reproduce/run_v2_parallel.py --gpus 0,1 \
  --workdir /var/tmp/team07-review223 \
  --storage-config /var/tmp/team07-review-storage/storage-config.json \
  --decision-cache-dir /path/to/executable/decision-cache
```

编译缓存仍需可执行挂载及足够空间；可设置`INIT_FFN_CACHE_DIR`、`TEMPORAL_BLOCK_CACHE_DIR`和`--decision-cache-dir`复用已验证的私有warm缓存；未指定缓存时需保证可执行空间足够，或用`TEAM07_EXECUTABLE_CACHE_PARENT`选择私有可执行缓存父目录。适配器配置不放宽原CUBIN检查，也不清理旧缓存或历史文件。两rank及官方validator使用普通视频目录，源码、metadata、输出和新NCCL控制文件放在私有tmpfs，host `.so`保留可执行挂载。适配器的loader符号/路径初始化早于Python计时；GPU/NCCL/模型工作没有移出原计时。

原始 209 文件清单中的 `src/unifolm_wma/utils/projector.py` 含既有语法错误（条件使用赋值号），全树 `compileall` 会在该文件失败。当前正式推理配置未使用它，本次保留原始字节；新增 helper、修改入口和实际运行路径另有解析、逐位及完整视频验证。

保留的单卡基线运行命令：

```bash
python3 reproduce/verify_snapshot.py
python3 reproduce/run_v2.py --gpu 0
```

后一条将加载 `reproduce/v2-effective-flags.json`，清除继承的 `UNIFOLM_*` 配置后启动原始 `team07_run.sh case-v2`。运行需要空闲 GPU、约 72 GiB 峰值显存、已预热的依赖/编译缓存；不同机器或冷缓存可能更慢。模型、权重缓存与图捕获的初始化仍在进程计时内。

当前223双卡入口的输出位于所选 `--workdir/diagnostic/runs/`，历史单卡入口仍位于 `~/benchmark-runs/unifolm/<timestamp>-case-v2-<pid>/`。本轮正式 219/220 同卡对照都使用私有 RAM 文件系统输出，native `.so` 保留在可执行 home 挂载；该存储变化须与历史计时区别看待。这是单项完整 11 轮运行，其他四项缺席会在完整五项校验中显示缺项；不能直接拿该单项结果冒充完整五项提交。

此前221正式推理因空间紧张，将rank0视频分放私有/tmp、rank1留在RAM。原官方校验器拒绝符号链接视频根；原失败证据保留。完成推理后，把同次rank1视频与rank0及220逐文件比对SHA，并硬链接为普通目录，官方校验通过后才汇总103视频。完整计时270.825199秒未修改。精确守护脚本及包装证明见 `experiments/temporal-block-220/full-split-r3`。常规复现应使用空间充足的普通目录，无需采用这个存储方案。

`results` 中保留了当前已验证的完整五项 log/JSON。重新提交必须使用新运行的真实输出，通过官方工作量和质量检查，保留同一份验证产生的完整 log/JSON，禁止改写耗时或 PSNR。

## 已采用的非 V2 分项

Z1、双臂堆箱和清理铅笔已分别采用63文件的双卡完整内部Transformer块分片；G1保留59文件单卡快照。例如：

```bash
~/.venv/bin/python reproduce/run_z1_parallel.py --gpus 0,1 \
  --workdir /var/tmp/team07-review-z1-63 \
  --storage-config /var/tmp/team07-review-storage/storage-config.json \
  --cache-dir /path/to/executable/FFN-shape-cache
~/.venv/bin/python reproduce/run_nonv2_parallel.py --case dual --gpus 0,1 \
  --workdir /var/tmp/team07-review-dual-63 \
  --storage-config /var/tmp/team07-review-storage/storage-config.json \
  --cache-dir /path/to/executable/FFN-shape-cache
~/.venv/bin/python reproduce/run_nonv2_parallel.py --case pencils --gpus 2,3 \
  --workdir /var/tmp/team07-review-pencils-63 \
  --storage-config /var/tmp/team07-review-storage/storage-config.json \
  --cache-dir /path/to/executable/FFN-shape-cache
python3 experiments/nonv2-vae-all-display/run_case.py --case g1 --gpu 0
```

每条命令均运行完整对应分项。所需外部模型、输入、验证工具与上面的 V2 相同。反馈帧使用原始 FP32 VAE；展示帧使用独立的 Half 权重缓存，原始 master weights 保留。

Z1入口先核对并建立63文件源码，使用冻结的整段双卡守护脚本运行12轮／25视频。`--prepare-only`只准备源码，不使用Torch/GPU，已在本机核对完整63个源码哈希。GPU运行前需要上述私有tmpfs配置；使用前文`build_and_check.py`构建。FFN形状缓存必须位于可执行挂载，比赛机home空间很少，本次复用既有私有缓存。新kernel编译、模型导入、双模型传输、通信、图捕获、写出和helper退出都在原始进程计时内；冷缓存可能更慢。输出与计时／单项质量材料位于所选workdir的`diagnostic/runs`。其单项shell会因另外四项缺席退出1；选中的Z1执行、profile和PSNR单独检查，不能将单项原始log/JSON直接当成五项提交。

63文件版本把16个空间和17个时间内部块纳入原来33次FFN汇合，保留全局帧action mask、完整5D GroupNorm和16时间步；条件投影缓存按完整／本地模式分库。原59个数学源码字节不变，原生Half GEMM形状变化会影响舍入。两阶段756参数的六状态检查与六对模型计时通过；完整Z1为152.203279秒／PSNR46.553398187dB、两rank25视频一致。当时103视频官方汇总accepted、五项802.134091秒。随后双臂堆箱94.319011秒／31.972206776dB和清理铅笔105.949243秒／34.881926888dB分别完成7／8轮、15／17视频，两rank一致，重新汇总103视频accepted，当时五项782.213299秒；随后V2采用223，当前五项778.585395秒，其他四项不变，网站尚未提交。

`run_nonv2_parallel.py`的`--prepare-only`在比赛机已检查三个分项的精确63文件清单与计时资源，不使用GPU。它只按分项替换三个数据集断言，数学helper与Z1相同，原分项flags、步数和frame stride保留。`--case g1`仅用于复现失败候选：完整11轮／23视频为142.834192秒但PSNR19.081776215dB，未通过质量，不能作为当前G1成绩。当前G1仍用上面的单卡命令。

`--variant temporal62`和`ffn61`可重建前序候选。历史单卡60入口仍保留在`experiments/nonv2-fused-half-geglu/run_case.py`；旧`nonv2-vae-all-display/run_case.py --case stackbox`仍转交这个历史60版本，加`--historical59`可重跑旧59基线。它们不代表当前63双卡默认复现入口。历史60的GPU2完整对照为171.304477→170.633517秒，GPU0正式记录166.780992秒；当前63相对该历史记录少14.577713秒，跨历史和卡数比较不能全部归因于单个kernel。

## CUDA Graph 扩展

已附带本次运行使用的 `unifolm/src/unifolm_wma/utils/native-graph-handles/team07_cuda_graph_handles.so`，SHA256 为 `ab2f19b88cb9af02b42fb5fdcb39da49e062346fd30794be45fb81e12cb1b927`。

扩展源码见 `unifolm/scripts/cuda_graph_handles.cpp`、构建脚本见 `build_cuda_graph_handles.py`。重新构建的字节哈希可能因构建路径或工具链不同而改变；不要直接删除生产代码的固定哈希检查。应先独立复核新扩展与 ABI，再一致地重新固定哈希并重跑正确性验证。当前脚本提供的是原始二进制的精确复现路径。
## 编译缓存的已知限制

当前生产代码对七类 Kahan GEMM 检查完整 cubin SHA256。核对发现，不同源码绝对路径会改变 cubin 的调试信息，从而使完整哈希不同，即使可执行指令、常量与参数 ABI 区段逐位相同。现有比赛机原启动环境与缓存上的检查通过；全新缓存环境尚不能保证直接运行成功。原始验证没有放宽，详见 [双卡实验记录](AGGRESSIVE_PARALLEL_EXPERIMENTS.md) 与 `results/aggressive-20261028/cubin-source-path-debug-proof-all7.json`。这与源码快照校验通过是不同的检查。

## 初始 FFN 的新形状缓存

220 版本只将原两个 JIT 的 M 改为20480，保持函数体和 BK32 不变。新变体使用私有、每 rank 的标准 Triton 缓存，原完整形状和固定 CUBIN 检查仍使用原 warm cache。默认启动器在可执行的 `~/.tmp` 下创建独立缓存；可通过 `INIT_FFN_CACHE_DIR` 指定已验证的私有 warm cache，避免与其他同时运行的任务共用。冷缓存编译成本仍在原进程计时中，可能比本次已预热运行慢。不得删除旧缓存或放宽原 CUBIN 门槛。

## 当前223的可执行缓存

比赛机home镜像余量151552字节，容器根文件系统仍有约70GB。此次新decision形状使用`/run/lock`下0700私有目录，CPU编译和dlopen检查通过；旧warm缓存未清理或改写。`/var/tmp`、`/tmp`和`/dev/shm`均为noexec，不能用于新launcher `.so`。公开223入口及默认转交入口均在比赛机完成CPU准备检查，重建的223源码及计时资源与实际运行相同；检查本身不初始化GPU。该检查不保证任意冷缓存能通过原Kahan cubin完整SHA门槛，原路径限制仍适用。
