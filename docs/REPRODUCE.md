# V2 复现

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
~/.venv/bin/python reproduce/run_v2_parallel.py --gpus 2,3 --workdir /tmp/unifolm-parallel-review-1
```

`--workdir` 必须是新目录。默认重建固定221文件源码：两阶段空间块、原生attention、decision原生卷积分帧、world初始FFN及16个常规world时间块扩展。完整5D GroupNorm和16时间步保留，既有FFN汇合边界复用。守护程序等待两张空闲卡并检查整段GPU占用；模型导入、双模型传输、通信及helper退出仍计入原始进程耗时。已验证完整结果为270.825199秒、PSNR29.675324786dB。`--variant both-spatial-temporal-resblock-grid-native-attention-initial-ffn` 可重建旧220版本（277.436969秒）；其余历史variant继续保留。单轮 `--smoke` 仅供排查，不能作为成绩。

原始 209 文件清单中的 `src/unifolm_wma/utils/projector.py` 含既有语法错误（条件使用赋值号），全树 `compileall` 会在该文件失败。当前正式推理配置未使用它，本次保留原始字节；新增 helper、修改入口和实际运行路径另有解析、逐位及完整视频验证。

保留的单卡基线运行命令：

```bash
python3 reproduce/verify_snapshot.py
python3 reproduce/run_v2.py --gpu 0
```

后一条将加载 `reproduce/v2-effective-flags.json`，清除继承的 `UNIFOLM_*` 配置后启动原始 `team07_run.sh case-v2`。运行需要空闲 GPU、约 72 GiB 峰值显存、已预热的依赖/编译缓存；不同机器或冷缓存可能更慢。模型、权重缓存与图捕获的初始化仍在进程计时内。

当前221双卡入口的输出位于所选 `--workdir/runs/`，历史单卡入口仍位于 `~/benchmark-runs/unifolm/<timestamp>-case-v2-<pid>/`。本轮正式 219/220 同卡对照都使用私有 RAM 文件系统输出，native `.so` 保留在可执行 home 挂载；该存储变化须与历史计时区别看待。这是单项完整 11 轮运行，其他四项缺席会在完整五项校验中显示缺项；不能直接拿该单项结果冒充完整五项提交。

本次221正式推理因空间紧张，将rank0视频分放私有/tmp、rank1留在RAM。原官方校验器拒绝符号链接视频根；原失败证据保留。完成推理后，把同次rank1视频与rank0及220逐文件比对SHA，并硬链接为普通目录，官方校验通过后才汇总103视频。完整计时270.825199秒未修改。精确守护脚本及包装证明见 `experiments/temporal-block-220/full-split-r3`。常规复现应使用空间充足的普通目录，无需采用这个存储方案。

`results` 中保留了当前已验证的完整五项 log/JSON。重新提交必须使用新运行的真实输出，通过官方工作量和质量检查，保留同一份验证产生的完整 log/JSON，禁止改写耗时或 PSNR。

## 已采用的非 V2 分项

Z1已采用60文件的Half projection／GEGLU融合版本，其他三个非V2项保留59文件快照。两者都使用非V2精度策略。例如：

```bash
python3 experiments/nonv2-fused-half-geglu/run_case.py --gpu 0 --workdir /fresh/executable/path/z1-review
python3 experiments/nonv2-vae-all-display/run_case.py --case dual --gpu 0
python3 experiments/nonv2-vae-all-display/run_case.py --case pencils --gpu 0
python3 experiments/nonv2-vae-all-display/run_case.py --case g1 --gpu 0
```

每条命令均运行完整对应分项。所需外部模型、输入、验证工具与上面的 V2 相同。反馈帧使用原始 FP32 VAE；展示帧使用独立的 Half 权重缓存，原始 master weights 保留。

Z1入口先核对并建立60文件源码和独立git快照，再使用已验证的整段GPU守护脚本运行12轮／25视频。`--prepare-only`只准备源码，不使用GPU。输出为所选workdir下的普通`outputs`目录，计时／官方验证材料位于`runs`。新kernel编译、模型导入、图捕获和退出仍在原始进程计时内；冷缓存可能更慢，缓存中的host `.so`需位于可执行挂载。旧的`nonv2-vae-all-display/run_case.py --case stackbox`默认转交当前60版本；加`--historical59`可重跑旧基线。完整GPU2对照为171.304477→170.633517秒，25视频一致；GPU0正式采用166.780992秒并通过新的103视频官方汇总，跨卡／跨历史运行差值不能全部归因于kernel。

## CUDA Graph 扩展

已附带本次运行使用的 `unifolm/src/unifolm_wma/utils/native-graph-handles/team07_cuda_graph_handles.so`，SHA256 为 `ab2f19b88cb9af02b42fb5fdcb39da49e062346fd30794be45fb81e12cb1b927`。

扩展源码见 `unifolm/scripts/cuda_graph_handles.cpp`、构建脚本见 `build_cuda_graph_handles.py`。重新构建的字节哈希可能因构建路径或工具链不同而改变；不要直接删除生产代码的固定哈希检查。应先独立复核新扩展与 ABI，再一致地重新固定哈希并重跑正确性验证。当前脚本提供的是原始二进制的精确复现路径。
## 编译缓存的已知限制

当前生产代码对七类 Kahan GEMM 检查完整 cubin SHA256。核对发现，不同源码绝对路径会改变 cubin 的调试信息，从而使完整哈希不同，即使可执行指令、常量与参数 ABI 区段逐位相同。现有比赛机原启动环境与缓存上的检查通过；全新缓存环境尚不能保证直接运行成功。原始验证没有放宽，详见 [双卡实验记录](AGGRESSIVE_PARALLEL_EXPERIMENTS.md) 与 `results/aggressive-20261028/cubin-source-path-debug-proof-all7.json`。这与源码快照校验通过是不同的检查。

## 初始 FFN 的新形状缓存

220 版本只将原两个 JIT 的 M 改为20480，保持函数体和 BK32 不变。新变体使用私有、每 rank 的标准 Triton 缓存，原完整形状和固定 CUBIN 检查仍使用原 warm cache。默认启动器在可执行的 `~/.tmp` 下创建独立缓存；可通过 `INIT_FFN_CACHE_DIR` 指定已验证的私有 warm cache，避免与其他同时运行的任务共用。冷缓存编译成本仍在原进程计时中，可能比本次已预热运行慢。不得删除旧缓存或放宽原 CUBIN 门槛。
