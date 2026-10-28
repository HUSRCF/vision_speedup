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

`--workdir` 必须是新目录。默认重建固定的 215 文件源码、等待两张空闲卡并检查整段 GPU 占用；模型导入、双模型传输、通信和 helper 退出仍计入原始进程耗时。完整结果为 301.930663 秒、PSNR 29.675324786 dB。添加 `--variant world-ffn-temporal` 可运行保留的旧 213 文件版本，历史结果为 312.462900 秒。单轮 `--smoke` 仅供排查，不能作为成绩。

保留的单卡基线运行命令：

```bash
python3 reproduce/verify_snapshot.py
python3 reproduce/run_v2.py --gpu 0
```

后一条将加载 `reproduce/v2-effective-flags.json`，清除继承的 `UNIFOLM_*` 配置后启动原始 `team07_run.sh case-v2`。运行需要空闲 GPU、约 72 GiB 峰值显存、已预热的依赖/编译缓存；不同机器或冷缓存可能更慢。模型、权重缓存与图捕获的初始化仍在进程计时内。

输出位于 `~/benchmark-runs/unifolm/<timestamp>-case-v2-<pid>/`。这是单项完整 11 轮运行，其他四项缺席会在完整五项校验中显示缺项；不能直接拿该单项结果冒充完整五项提交。

`results` 中保留了已验证的历史完整五项 log/JSON。重新提交必须使用新运行的真实输出，通过官方工作量和质量检查，保留同一份验证产生的完整 log/JSON，禁止改写耗时或 PSNR。

## 已采用的非 V2 分项

已采用的四个非 V2 分项使用单独的 59 文件源码快照，避免误用 V2 的精度策略。例如：

```bash
python3 experiments/nonv2-vae-all-display/run_case.py --case stackbox --gpu 0
python3 experiments/nonv2-vae-all-display/run_case.py --case dual --gpu 0
python3 experiments/nonv2-vae-all-display/run_case.py --case pencils --gpu 0
python3 experiments/nonv2-vae-all-display/run_case.py --case g1 --gpu 0
```

每条命令均运行完整对应分项。所需外部模型、输入、验证工具与上面的 V2 相同。反馈帧使用原始 FP32 VAE；展示帧使用独立的 Half 权重缓存，原始 master weights 保留。

## CUDA Graph 扩展

已附带本次运行使用的 `unifolm/src/unifolm_wma/utils/native-graph-handles/team07_cuda_graph_handles.so`，SHA256 为 `ab2f19b88cb9af02b42fb5fdcb39da49e062346fd30794be45fb81e12cb1b927`。

扩展源码见 `unifolm/scripts/cuda_graph_handles.cpp`、构建脚本见 `build_cuda_graph_handles.py`。重新构建的字节哈希可能因构建路径或工具链不同而改变；不要直接删除生产代码的固定哈希检查。应先独立复核新扩展与 ABI，再一致地重新固定哈希并重跑正确性验证。当前脚本提供的是原始二进制的精确复现路径。
## 编译缓存的已知限制

当前生产代码对七类 Kahan GEMM 检查完整 cubin SHA256。核对发现，不同源码绝对路径会改变 cubin 的调试信息，从而使完整哈希不同，即使可执行指令、常量与参数 ABI 区段逐位相同。现有比赛机原启动环境与缓存上的检查通过；全新缓存环境尚不能保证直接运行成功。原始验证没有放宽，详见 [双卡实验记录](AGGRESSIVE_PARALLEL_EXPERIMENTS.md) 与 `results/aggressive-20261028/cubin-source-path-debug-proof-all7.json`。这与源码快照校验通过是不同的检查。
