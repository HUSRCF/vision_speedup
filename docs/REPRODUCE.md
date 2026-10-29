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
  --workdir /var/tmp/unifolm-parallel-review-225 \
  --storage-config /var/tmp/team07-review-storage/storage-config.json \
  --decision-cache-dir /path/to/executable/decision-cache
```

`--workdir` 必须是新目录。默认重建固定225文件，在223上新增9个world Winograd CO分片，保留原CUBIN、constexpr、batch stride和K顺序。完整同GPU0/1对照260.425380→257.945205秒，净省2.480175秒；23视频SHA及PSNR29.675324786dB相同，新103视频官方汇总accepted，五项771.314275秒。pack、9次新增world汇合、布局恢复和初始化均计入原process timer。公开入口与默认入口CPU重建均匹配完整运行的225文件及资源SHA。旧223可加`--variant decision-temporal223`重建。

历史223重建固定源码：两阶段空间块、原生attention、decision原生卷积分帧、world初始FFN、16个常规world时间块及初始时间块边界扩展，再分片17个decision时间块内的独立Triton工作。Decision保留vendor GEMM描述符，新增17次块末汇合。完整5D GroupNorm和16时间步保留，既有FFN汇合边界复用。守护程序等待两张空闲卡并检查整段GPU占用；模型导入、双模型传输、通信及helper退出仍计入原始进程耗时。已验证完整结果为265.216325秒、PSNR29.675324786dB。`--variant both-spatial-temporal-resblock-grid-native-attention-initial-ffn` 可重建旧220版本（277.436969秒）；其余历史variant继续保留。单轮 `--smoke` 仅供排查，不能作为成绩。

历史同GPU0/1、同guard／适配器／存储的222／223完整对照为270.119302→265.216325秒，省4.902977秒；相对历史正式222少3.627904秒。23视频与原正式及另一rank的SHA一致，103视频官方汇总accepted，五项778.585395秒。六对模型检查的16.6ms净省不能代替完整进程差值。原222可通过`--variant both-spatial-temporal-resblock-grid-native-attention-initial-ffn-temporal-block-initial-block`重建。此前221／222完整对照272.043547→268.844229秒的历史证据继续保留。历史221可通过`--variant both-spatial-temporal-resblock-grid-native-attention-initial-ffn-temporal-block`重建。`--prepare-only`仅建立源码/guard，不使用GPU。

223／225完整守护需要上述私有tmpfs配置，先按下面命令构建。比赛机/dev/shm空间受限，使用已经通过CPU和双卡图内通信测试的[私有tmpfs适配器](../experiments/private-nccl-tmpfs)。它只调整新建NCCL控制文件的存储位置，完整对照使用相同设置。选择两个新目录：

```bash
~/.venv/bin/python experiments/private-nccl-tmpfs/build_and_check.py \
  --workdir /var/tmp/team07-review-storage \
  --exec-dir "$HOME/.tmp/team07-review-storage-exec"
~/.venv/bin/python reproduce/run_v2_parallel.py --gpus 0,1 \
  --workdir /var/tmp/team07-review225 \
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

当前225双卡入口的输出位于所选 `--workdir/diagnostic/runs/`，历史单卡入口仍位于 `~/benchmark-runs/unifolm/<timestamp>-case-v2-<pid>/`。本轮正式 219/220 同卡对照都使用私有 RAM 文件系统输出，native `.so` 保留在可执行 home 挂载；该存储变化须与历史计时区别看待。这是单项完整 11 轮运行，其他四项缺席会在完整五项校验中显示缺项；不能直接拿该单项结果冒充完整五项提交。

此前221正式推理因空间紧张，将rank0视频分放私有/tmp、rank1留在RAM。原官方校验器拒绝符号链接视频根；原失败证据保留。完成推理后，把同次rank1视频与rank0及220逐文件比对SHA，并硬链接为普通目录，官方校验通过后才汇总103视频。完整计时270.825199秒未修改。精确守护脚本及包装证明见 `experiments/temporal-block-220/full-split-r3`。常规复现应使用空间充足的普通目录，无需采用这个存储方案。

`results` 中保留了当前已验证的完整五项 log/JSON。重新提交必须使用新运行的真实输出，通过官方工作量和质量检查，保留同一份验证产生的完整 log/JSON，禁止改写耗时或 PSNR。

## 已采用的非 V2 分项

当前Z1采用65文件双卡13块两次刷新预测，双臂堆箱采用65文件双卡23块短区间预测，清理铅笔采用66文件23块预测＋末端5块空间分帧，G1采用68文件双卡时间attention与精确权重广播。对应入口：

```bash
~/.venv/bin/python reproduce/run_nonv2_prediction65.py --revision 2 --gpus 0,1 \
  --workdir /var/tmp/team07-review-z1-prediction65 \
  --storage-config /var/tmp/team07-review-storage/storage-config.json \
  --cache-dir /path/to/executable/FFN-shape-cache
~/.venv/bin/python reproduce/run_nonv2_dual_late23_prediction65.py --gpus 0,1 \
  --workdir /var/tmp/team07-review-dual-late23-prediction65 \
  --storage-config /var/tmp/team07-review-storage/storage-config.json \
  --cache-dir /path/to/executable/FFN-shape-cache
~/.venv/bin/python reproduce/run_nonv2_resblock_tail66_full.py --gpus 2,3 \
  --workdir /var/tmp/team07-review-pencils-tail66 \
  --storage-config /var/tmp/team07-review-storage/storage-config.json \
  --cache-dir /path/to/executable/FFN-shape-cache
~/.venv/bin/python reproduce/run_g1_native_shared68.py --gpus 0,1 \
  --workdir /var/tmp/team07-review-g1-68 \
  --storage-config /var/tmp/team07-review-storage/storage-config.json \
  --cache-dir /path/to/executable/FFN-shape-cache
```

每条命令均运行完整分项，先核对冻结源码与计时资源；`--prepare-only`只重建源码，不使用GPU。所需外部模型、输入和验证工具与V2相同。source与NCCL backing需位于同一私有tmpfs；FFN新形状缓存必须位于可执行挂载。初始化、模型搬运、通信、图捕获、视频写出及helper退出均计入原进程时间。单项shell会因其他四项缺席退出1，选中分项的执行、profile和PSNR另行检查，不能把单项原始log／JSON直接当成五项提交。

Z1为145.327923秒／28.992932876dB，完整12轮、25视频，两阶段各600次调用。13块深层分支由最近两次完整刷新预测当前特征，每阶段每轮15个位置预测，保留全部50步、浅层、head和采样更新；四张图分别核对33／21个汇合。新预测kernel缓存作用域恢复环境及Triton缓存类，原FFN断言保留；可选`--prediction-cache-parent`指定可执行缓存父目录。该双卡组合独立验证质量，新103视频汇总accepted、其他78视频SHA不变，该轮正式五项合计758.855686秒。比历史正式Z1少5.510466秒，非同期配对，详见[完整证据](NONV2_PREDICTION65_20261029.md)。历史7块65入口`run_nonv2_crossstep65.py`继续保留。

双臂随后采用同一13块预测模块，完整7轮／15视频为90.557285秒／26.513549583dB，保留全部50步和33／21汇合；新103视频验收通过，其他88视频SHA不变，五项更新为755.093960秒。清理铅笔同方案100.661865秒／19.202957201dB，质量失败，未采用。详见[完整记录](NONV2_OTHER_PREDICTION65_20261029.md)。

清理铅笔随后扩大为23块、缩短至`[20,40)`预测窗口，完整8轮17视频 **95.640695秒／26.798606240dB**，原50步和浅层／head保留，两rank视频一致。新103视频验收通过，其他86视频SHA不变，当时五项 **744.785412秒**；详见[23块方案与完整证据](NONV2_LATE23_PREDICTION65_20261029.md)。

双臂继续采用23块`[20,40)`双卡预测，完整7轮15视频为 **84.577327秒／26.556920797dB**，比历史正式少5.979958秒，非同期配对。新103视频验收通过，其他88视频SHA不变，当时五项 **738.805454秒**；详见[双臂23块方案](NONV2_DUAL_LATE23_PREDICTION65_20261029.md)。

G1为155.314304秒／26.107156332dB，68文件入口保留其原始Half GEMM／GEGLU和31次汇合，只有rank0进行权重H2D，然后精确广播。两次同GPU完整对照及103视频验收见[G1记录](G1_ROOT_WEIGHT_TRANSPORT68_20261029.md)。

历史Z1的63入口`run_z1_parallel.py`以及62／61候选继续保留；历史G1单卡59与失败63候选也保留。`run_nonv2_parallel.py --case g1`只复现PSNR19.081776215dB的失败候选，不能作为当前G1成绩。精确静态时间嵌入缓存仍是独立实验，没有替换上述正式配置。

双臂／清理铅笔的双卡深层时间步缓存65候选另见[完整记录](NONV2_OTHER_CROSSSTEP65_20261029.md)。两项完整运行均通过质量及双rank一致性，但未改善历史正式时间，未采用。入口 `run_nonv2_other_crossstep65.py --case dual|pencils` 使用相同的 `--gpus`、`--workdir`、`--storage-config`、`--cache-dir` 参数；可用 `--prepare-only` 精确重建而不执行GPU。检查命令为 `python reproduce/verify_nonv2_other_crossstep65.py`。

## CUDA Graph 扩展

已附带本次运行使用的 `unifolm/src/unifolm_wma/utils/native-graph-handles/team07_cuda_graph_handles.so`，SHA256 为 `ab2f19b88cb9af02b42fb5fdcb39da49e062346fd30794be45fb81e12cb1b927`。

扩展源码见 `unifolm/scripts/cuda_graph_handles.cpp`、构建脚本见 `build_cuda_graph_handles.py`。重新构建的字节哈希可能因构建路径或工具链不同而改变；不要直接删除生产代码的固定哈希检查。应先独立复核新扩展与 ABI，再一致地重新固定哈希并重跑正确性验证。当前脚本提供的是原始二进制的精确复现路径。
## 编译缓存的已知限制

当前生产代码对七类 Kahan GEMM 检查完整 cubin SHA256。核对发现，不同源码绝对路径会改变 cubin 的调试信息，从而使完整哈希不同，即使可执行指令、常量与参数 ABI 区段逐位相同。现有比赛机原启动环境与缓存上的检查通过；全新缓存环境尚不能保证直接运行成功。原始验证没有放宽，详见 [双卡实验记录](AGGRESSIVE_PARALLEL_EXPERIMENTS.md) 与 `results/aggressive-20261028/cubin-source-path-debug-proof-all7.json`。这与源码快照校验通过是不同的检查。

## 初始 FFN 的新形状缓存

220 版本只将原两个 JIT 的 M 改为20480，保持函数体和 BK32 不变。新变体使用私有、每 rank 的标准 Triton 缓存，原完整形状和固定 CUBIN 检查仍使用原 warm cache。默认启动器在可执行的 `~/.tmp` 下创建独立缓存；可通过 `INIT_FFN_CACHE_DIR` 指定已验证的私有 warm cache，避免与其他同时运行的任务共用。冷缓存编译成本仍在原进程计时中，可能比本次已预热运行慢。不得删除旧缓存或放宽原 CUBIN 门槛。

## 当前223的可执行缓存

比赛机home镜像余量151552字节，容器根文件系统仍有约70GB。此次新decision形状使用`/run/lock`下0700私有目录，CPU编译和dlopen检查通过；旧warm缓存未清理或改写。`/var/tmp`、`/tmp`和`/dev/shm`均为noexec，不能用于新launcher `.so`。公开223入口及默认转交入口均在比赛机完成CPU准备检查，重建的223源码及计时资源与实际运行相同；检查本身不初始化GPU。该检查不保证任意冷缓存能通过原Kahan cubin完整SHA门槛，原路径限制仍适用。

G1时间步预测70文件与全刷新诊断对照见[完整记录](G1_PREDICTION70_20261029.md)。`run_g1_prediction70.py --depth 13|23`复现两个质量失败候选，`run_g1_refresh_control70.py`复现全部步骤完整计算、23视频与原68逐一SHA相同的接入对照；三者均未替换上面的正式G1入口。

G1只读时间步轨迹诊断：运行`reproduce/run_g1_feature_trajectory71.py`，资源配置与原G1入口一致。71源码与计时资源已在比赛机禁用CUDA精确重建，完整11轮／23视频与原正式SHA一致。使用`python reproduce/verify_g1_feature_trajectory71.py`核对证据，详见[G1轨迹诊断](G1_FEATURE_TRAJECTORY71_20261029.md)。该诊断不是新的正式速度结果。

非V2残差块双卡分帧：模型入口`reproduce/run_nonv2_resblock_frames72.py`、完整入口`reproduce/run_nonv2_resblock66_full.py`，沿用已有私有tmpfs配置并显式提供可执行文件系统上的`--cache-dir`。公开重建与GPU证据见[分帧实验](NONV2_RESBLOCK_FRAMES72_20261029.md)；完整进程回退，未采用。

清理铅笔现采用[末端5块分帧66文件](NONV2_RESBLOCK_TAIL73_20261029.md)，完整92.232482秒／26.798606240dB；前后原65同卡对照94.368267／93.983744秒。官方103视频汇总accepted，所有103视频与上一版SHA相同，本机双文件已更新，当时五项735.397241秒，最新见下。

Z1现采用[末端5块分帧＋原13块预测66文件](NONV2_OTHER_TAIL66_20261029.md)，完整138.945151秒／28.992932876dB；同卡原65对照146.315426秒。新103视频官方汇总accepted、全部103视频SHA不变，五项729.014469秒。入口`reproduce/run_nonv2_other_tail66.py --case z1`，其余`--gpus`、`--workdir`、`--storage-config`和可执行`--cache-dir`参数同清理铅笔完整入口；`--case dual`仅复现尚未采用的双臂候选。公开入口已在比赛机禁用CUDA精确重建两项66源码并通过CPU边界／预测器／缓存检查。使用`python reproduce/verify_nonv2_other_tail66.py`核对完整证据。
