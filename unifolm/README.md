# NEWBIE 2.0 · UnifoLM

优化 UnifoLM-WMA 的完整视频生成推理，在输出正确的前提下缩短五个小项的运行时间。每项 40 分，本题共 200 分；与 MPI Attention 合计 320 分。

## 当前比赛环境

软件信息于 2026-10-11 核实；Python GPU 栈为安装脚本配置，并在 testteam 私人环境中验证：

| 项目 | 当前配置 |
| --- | --- |
| 玩家容器 | Ubuntu 24.04 LTS，x86_64 |
| CPU | 双路 Intel Xeon Platinum 8358P @ 2.60 GHz，64 个可用 CPU（0–63） |
| 比赛 GPU 配置 | 宿主 GPU 4、5、6、7：4 × NVIDIA A800-SXM4-80GB |
| 私人 Python 环境 | `~/miniconda3/envs/unifolm`，Python 3.10.18 |
| PyTorch | 2.3.1+cu121，torchvision 0.18.1+cu121 |
| PyTorch CUDA runtime | 12.1，cuDNN 8.9.2 |
| 玩家存储 | 独立 home，限额 50 GiB；代码、环境、缓存和输出均存入自己的 home |
| 共享资料 | `/share` 只读，含赛题、权重、离线缓存和 FP32 参考视频 |

**设备映射：**玩家容器配置指定宿主 GPU 4、5、6、7，所有玩家共享使用；CPU 和内存不设容器配额。配置需要管理员执行 `deploy/player/apply-resources.sh` 重建容器后生效。使用前执行 `nvidia-smi -L` 检查实际可见设备；容器内 CUDA 逻辑编号以实际枚举为准。现有 baseline 入口固定使用可见设备 0，并未实现自动四卡并行。

`nvidia-smi` 显示的 CUDA Version 是驱动兼容能力，不是私人环境内 PyTorch 的 CUDA runtime 版本；环境以 `torch.version.cuda` 为准。CPU/GPU 为共享资源。

## 1. 安装自己的环境

系统工具由容器提供，每位玩家独立安装 Miniconda 和依赖。第一次执行：

```bash
mkdir -p ~/.tmp
export TMPDIR="$HOME/.tmp"
bash /share/tools/install-miniconda.sh
~/miniconda3/bin/python -m pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
bash /share/tools/install-unifolm-env.sh
```

已有 `~/miniconda3` 时跳过 Miniconda 安装步骤。激活环境后，也可以使用以下命令换 pip 默认源：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ~/miniconda3/envs/unifolm
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

pip 默认源用于未显式指定源的依赖安装；安装脚本中的 PyTorch CUDA 包仍使用指定的 PyTorch 官方源。此配置不修改 Conda 软件源。环境安装需要联网，模型权重使用共享文件。

安装脚本安装的是依赖，不会把某个赛题副本固定成全局代码入口；运行时用当前仓库和 `PYTHONPATH` 明确指定源码。

## 2. 一条命令跑完整 FP32 baseline

环境准备好后执行：

```bash
python3 /share/tools/run-unifolm-baseline.py
```

该脚本使用自己的 `~/competition/unifolm-fp32-baseline`，目录不存在时自动复制并补写权限；已有副本保持不变。它直接读取共享权重、离线 CLIP 缓存、固定预处理输入和参考视频，并要求两个 TF32 开关关闭。

本次产物：

```text
~/benchmark-runs/unifolm/<运行时间>/
├── result.json       可提交的完整结果
├── runtime.log       同次运行的完整日志
├── profiles/         分项计时及 PSNR
├── outputs/          生成视频
├── environment.txt
└── gpu-info.txt
```

最后输出准确的 `RESULT_JSON` 和 `RUNTIME_LOG` 路径。脚本不会自动上传成绩。

## 3. 准备自己的优化代码

在目标尚不存在时复制一次：

```bash
mkdir -p ~/competition
cp -r /share/problems/unifolm-world-model-action ~/competition/unifolm-world-model-action
chmod -R u+rwX ~/competition/unifolm-world-model-action
cd ~/competition/unifolm-world-model-action
```

之后直接修改这个目录中的源码。普通 Python 修改不需要重新安装环境，下次启动会加载新代码；正在运行的进程应结束后重新启动。新增依赖或编译扩展则需要相应安装、构建。

## 4. 链接共享权重，避免重复下载

主 ckpt 约 16 GB，已经位于：

```text
/share/weights/UnifoLM-WMA-0-Dual/unifolm_wma_dual.ckpt
```

在自己的仓库根目录中，可以将它链接到默认下载路径：

```bash
mkdir -p downloads/UnifoLM-WMA-0-Dual
ln -s /share/weights/UnifoLM-WMA-0-Dual/unifolm_wma_dual.ckpt \
  downloads/UnifoLM-WMA-0-Dual/unifolm_wma_dual.ckpt
readlink -f downloads/UnifoLM-WMA-0-Dual/unifolm_wma_dual.ckpt
```

这仅创建软链接，不复制模型，不会再占一份 16 GB 配额。已有文件或链接时先检查，不要盲目覆盖。另一种方式是直接设置 `CKPT_PATH` 指向共享文件，如下一节所示。

CLIP 是另一套约 3.7 GB 的权重：

```text
/share/weights/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin
```

程序通过 OpenCLIP/Hugging Face 缓存加载它。需同时设置 `/share/huggingface/hub` 和离线开关；仅链接主 ckpt 不足以阻止 CLIP 联网下载。

## 5. 运行自己的修改版

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ~/miniconda3/envs/unifolm
cd ~/competition/unifolm-world-model-action
mkdir -p ~/.tmp
export TMPDIR="$HOME/.tmp"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export CKPT_PATH=/share/weights/UnifoLM-WMA-0-Dual/unifolm_wma_dual.ckpt
export HF_HUB_CACHE=/share/huggingface/hub
export HUGGINGFACE_HUB_CACHE=/share/huggingface/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
export BENCHMARK_N_ITER=original
export INPUT_CACHE_DIR=/share/problems/unifolm-world-model-action/examples/compute_benchmark/preprocessed_inputs
export RESULT_DIR="$HOME/benchmark-runs/unifolm-custom/$(date -u +%Y%m%dT%H%M%SZ)-$$"
export PROFILE_DIR="$RESULT_DIR/profiles"
export RESULT_JSON_PATH="$RESULT_DIR/result.json"
export LOG_PATH="$RESULT_DIR/runtime.log"
bash scripts/run_compute_benchmark.sh
```

此命令使用 `~/competition/unifolm-world-model-action` 的修改版；一键 baseline 使用的是另一个目录 `~/competition/unifolm-fp32-baseline`，不要修改一份却运行另一份。可用以下命令确认模型包位置：

```bash
python -c 'import unifolm_wma; print(list(unifolm_wma.__path__))'
```

当前入口默认 `tee` 保存标准输出和错误输出；结果与日志已有时拒绝覆盖，每次选择新目录。计算资源共享，建议使用持久终端会话运行长任务，避免 SSH 中断。

## 6. 完整工作量与正确性

| 小项 | 完整交互轮数 | 分值 |
| --- | ---: | ---: |
| `unitree_z1_stackbox` | 12 | 40 |
| `unitree_z1_dual_arm_stackbox` | 7 | 40 |
| `unitree_z1_dual_arm_stackbox_v2` | 11 | 40 |
| `unitree_z1_dual_arm_cleanup_pencils` | 8 | 40 |
| `unitree_g1_pack_camera` | 11 | 40 |

完整运行共 49 轮、103 个视频。固定输入缓存冻结初始张量和首次 DDIM 采样前的随机状态，不缓存后续交互轮次、模型权重或推理输出。`configs/train/meta.json` 是推理模型构建依赖的历史元数据文件，不要求进行训练。

生成视频与以下只读 FP32 reference 比较：

```text
/share/results/unifolm/fp32/reference/outputs
```

每项 PSNR 必须 **严格大于 25 dB**，并匹配所需视频集合、帧数和尺寸。JSON 中记录 `psnr_db`、精确匹配标记及通过标记，日志输出对应 PSNR 和 TIMING 行。精确一致产生的无穷大 PSNR 由工具用 `null` 加精确匹配标记表示，不手工写入非法 JSON 数值。

不能减少轮数、跳过必需计算或输出、回放参考视频、修改正确性阈值或伪造计时。`BENCHMARK_N_ITER` 的减量模式只用于本地排查，不是可提交工作量。

### TF32 实测注意事项

FP32 baseline 关闭 cuDNN 和矩阵乘法两个 TF32 开关。自己的优化实现可尝试 TF32，但每次仍须满足 PSNR。

此前完整运行中，`unitree_z1_dual_arm_stackbox_v2` 仅开 cuDNN TF32 为 23.45 dB、仅开矩阵乘法 TF32 为 22.48 dB、两者同开为 20.91 dB，均不通过。因此不能直接将这些全局开关方案视为该项的合格优化。

## 7. 计时、评分和上传

评分使用每项 `process_wall_s`，包含该项进程导入、模型加载、推理及必需输出写入；`pure_model_cuda_s` 仅用于诊断。PSNR 校验耗时独立记录，不计入推理性能时间。各项时间之和与包含启动间隔的整体 `external_wall_s` 可以不同。

```text
该项得分 = 40 × 全场该项最短有效 process_wall_s / 本队该项有效 process_wall_s
```

五项相加得到本题成绩。每队各项分别取历史最好有效成绩；某项失败只使本次该项为 0 分，其余有效项可以计分，平台可能显示「部分通过」。后续失败不会覆盖已有历史有效成绩。整份材料格式错误或最终状态冲突仍会导致整份校验失败。

在 <http://10.120.20.219:8000> 用队伍名与队伍 token 登录，选择 UnifoLM，同时上传本次 `result.json` 和 `runtime.log`，每个文件最多 5 MiB。不上传视频、权重或单项 A/B 诊断 JSON，无需 Git URL 或 Git SHA。

当前网站核对上传证据，不自动执行隐藏测试或重跑代码。当前 SUBMISSION.md 已统一为双文件上传及获奖材料交付契约。管理员复核及原始材料保留仍适用。

完整入门、命令行上传及故障排查见 `/share/QUICKSTART.md`，PSNR 细节见同目录 `PSNR.txt`。


## 2.0.1 发布与规则入口

本版发布于 2026-10-14。提交方式和正确性约束统一以当前 [SUBMISSION.md](SUBMISSION.md)、`/share/PARTICIPANT_HANDBOOK.md` 和 `/share/QUICKSTART.md` 为准。获奖候选队伍需提交可复现代码与技术报告，按组织者安排复跑后确认奖项。

新版入口逐项记录执行状态，推理失败继续其他独立项，并在结束时校验已完成项。缺项和视频损坏只影响该项；健康项的 PSNR/计时保留。旧私人副本需同步新版 run_compute_benchmark.sh。
