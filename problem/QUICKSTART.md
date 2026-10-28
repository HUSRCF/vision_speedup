# NEWBIE 2.0 快速开始 / Quick Start

中英双语版 · 2026-10-20 / Bilingual edition · 2026-10-20

[中文](#zh-cn) · [English](#english)

<a id="zh-cn"></a>

## 中文

适用环境：当前比赛服务器的 Ubuntu 24.04 玩家容器。更新日期：2026-10-20。

本文命令均在自己的玩家账号下执行，不使用 `sudo`。使用管理员分配给你的参赛用户名，例如 `team01`；SSH 私钥和网站 token 由管理员分别派发，两者用途不同。

### 1. 登录与目录约定

在自己的电脑上登录，例如：

```bash
ssh -i /本地路径/team01_ed25519 -p 22000 team01@10.120.20.219
```

私钥权限过宽时，在本地先执行 `chmod 600 /本地路径/team01_ed25519`。账号、密钥路径按实际分配替换。

| 容器内路径 | 用途 |
| --- | --- |
| `~`，即 `/home/<队伍名>` | 自己的代码、环境、缓存和结果；每位玩家限额 50 GiB |
| `/share/problems` | 只读赛题原件，复制到自己的 home 后修改 |
| `/share/weights` | 已下载的主模型和 CLIP 权重，只读共享 |
| `/share/huggingface/hub` | 已准备好的 Hugging Face 离线模型缓存 |
| `/share/results/unifolm/fp32/reference/outputs` | UnifoLM 正确性校验使用的 FP32 参考视频 |
| `/share/tools` | 环境安装、一键 baseline、结果整理和 PSNR 校验工具 |
| `~/benchmark-runs` | 一键 baseline 的逐次运行记录 |

宿主机上的共享目录为 `/data/newbie-data/share`，玩家容器内使用 `/share`。home 对应宿主机 `/data/newbie-data/home/<队伍名>`。

`/share` 只读。复制过来的文件可能仍然只读，必须给自己的副本补充权限。示例（目标目录尚不存在时执行）：

```bash
mkdir -p ~/competition
cp -r /share/problems/mpi-attention ~/competition/mpi-attention-baseline
chmod -R u+rwX ~/competition/mpi-attention-baseline
```

不要反复复制到已存在的目标目录，以免多嵌套一层目录。一键 baseline 会在源码目录不存在时自动完成复制和权限调整。

### 2. 检查基础环境

容器已提供编译工具、OpenMPI 等系统工具。Python/Conda 环境由各队独立安装到自己的 home。

```bash
whoami
printf 'HOME=%s\n' "$HOME"
df -h "$HOME"
command -v gcc make mpicc mpirun python3
mpirun --version
nvidia-smi
mkdir -p ~/.tmp
export TMPDIR="$HOME/.tmp"
```

比赛容器配置为共享宿主机 GPU 4、5、6、7，共四张 A800 80GB，CPU 和内存不设容器配额。资源配置需要管理员重建容器才能生效；请用 `nvidia-smi -L` 核对实际可见设备。容器内 CUDA 逻辑编号以实际枚举为准，不直接使用宿主编号。现有 baseline 默认只用一张可见 GPU。计算资源共享，跑分会受到其他任务影响，尽量避免同时运行多个完整 benchmark。

### 3. 安装自己的 Miniconda 和 UnifoLM 环境

MPI baseline 不需要 Conda，可以直接跳到第 5 节。

#### 第一次安装

```bash
mkdir -p ~/.tmp
export TMPDIR="$HOME/.tmp"
bash /share/tools/install-miniconda.sh
~/miniconda3/bin/python -m pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
bash /share/tools/install-unifolm-env.sh
```

如果 `~/miniconda3` 已安装，不要重复运行 Miniconda 安装脚本。第二个脚本会创建自己的 `~/miniconda3/envs/unifolm`，安装 Python 3.10.18、PyTorch 2.3.1+cu121、torchvision 0.18.1+cu121 和赛题依赖。

安装依赖前，先将自己账号的 pip 默认源切换到阿里云。若已经激活 Conda 环境，也可以直接执行：

```bash
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
```

该配置用于后续未显式指定源的 pip 安装；安装脚本中的 PyTorch CUDA 12.1 包仍使用显式指定的 PyTorch 官方源。此命令不修改 Conda 的软件源。

安装环境需要访问软件源；共享权重能避免重新下载模型，不能代替 Conda/pip 依赖安装。

#### 激活与检查

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ~/miniconda3/envs/unifolm
python -c 'import torch; print(torch.__version__); print(torch.cuda.is_available())'
```

每次新登录后可以重复执行上述激活命令。环境、软件包缓存也占自己的 50 GiB 配额；不要复制其他队伍的 Conda 环境。

### 4. 将共享 ckpt 链接到自己的代码目录，避免重复下载

主权重约 16 GB，CLIP 权重约 3.7 GB，已经在共享目录中。请直接读取或创建软链接，不要再次下载或复制。

先准备自己用于修改的 UnifoLM 源码副本（目标目录尚不存在时执行）：

```bash
mkdir -p ~/competition
cp -r /share/problems/unifolm-world-model-action ~/competition/unifolm-world-model-action
chmod -R u+rwX ~/competition/unifolm-world-model-action
cd ~/competition/unifolm-world-model-action
```

#### 方式 A：软链接到程序默认的 downloads 路径

在自己的 UnifoLM 仓库根目录执行一次：

```bash
mkdir -p downloads/UnifoLM-WMA-0-Dual
ln -s /share/weights/UnifoLM-WMA-0-Dual/unifolm_wma_dual.ckpt \
  downloads/UnifoLM-WMA-0-Dual/unifolm_wma_dual.ckpt

readlink -f downloads/UnifoLM-WMA-0-Dual/unifolm_wma_dual.ckpt
test -r downloads/UnifoLM-WMA-0-Dual/unifolm_wma_dual.ckpt && echo 'CKPT 可读'
```

若提示 `File exists`，先用 `ls -l` 检查已有文件或链接；已有正确链接无需重建。这是软链接，模型数据仍在 `/share` 中，不会在自己的 home 再占一份 16 GB。

#### 方式 B：直接指定共享权重路径

不创建链接也可以，在运行前设置：

```bash
export CKPT_PATH=/share/weights/UnifoLM-WMA-0-Dual/unifolm_wma_dual.ckpt
```

`CKPT_PATH` 优先于默认 `downloads/...` 路径。一键 baseline 已经直接使用该共享路径，无需手工链接。

#### CLIP 权重与离线缓存

主 ckpt 之外，程序还会通过 OpenCLIP 加载 CLIP 权重。仅链接主 ckpt 不足以阻止 CLIP 再次联网；运行前还应设置：

```bash
export HF_HUB_CACHE=/share/huggingface/hub
export HUGGINGFACE_HUB_CACHE=/share/huggingface/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

现有缓存已经指向共享 CLIP 模型。通常不需要把 CLIP 再链接到自己的仓库；若自己的代码改成显式读取本地 CLIP 权重，可使用：

```text
/share/weights/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin
```

这些环境变量只在当前 shell 生效。一键 UnifoLM baseline 会自动设置它们。

### 5. 最快获得一份完整 baseline

#### MPI Attention：一条命令

```bash
python3 /share/tools/run-mpi-baseline.py
```

它使用自己的 `~/competition/mpi-attention-baseline`，不存在时复制共享源码；已有副本会继续使用，不会重置自己的修改。脚本构建程序，在自己的 `~/benchmark-cache/mpi/reference` 生成或复用 FP64 参考输出，并核对 manifest 中的 SHA-256。

默认使用 16 个 MPI rank、每 rank 1 个 OpenMP 线程，完成三个小项，每项预热 1 次、测量 5 次取中位数。

输出目录：

```text
~/benchmark-runs/mpi/<本次运行时间>/
├── result.json       ← 提交这个 JSON
├── runtime.log       ← 同一次运行的完整日志
└── outputs/          ← 各小项产物
```

脚本最后打印本次 `RESULT_JSON` 和 `RUNTIME_LOG` 的准确路径。公共 baseline 的成绩可能进入管理员复核，等待复核不代表程序运行失败。

#### UnifoLM：一条命令

先完成第 3 节的私人环境安装，再执行：

```bash
python3 /share/tools/run-unifolm-baseline.py
```

它使用 `~/competition/unifolm-fp32-baseline`；不存在时自动复制源码。该入口要求两个 TF32 开关均为关闭状态，适合获取 FP32 baseline。它自动使用共享权重、离线 CLIP 缓存、预处理输入和参考视频，不需要额外下载 ckpt。

完整评测包含五个小项，共 49 轮，产出 103 个视频。不要通过减少轮数来提交成绩。

| 小项 | 完整轮数 |
| --- | ---: |
| `unitree_z1_stackbox` | 12 |
| `unitree_z1_dual_arm_stackbox` | 7 |
| `unitree_z1_dual_arm_stackbox_v2` | 11 |
| `unitree_z1_dual_arm_cleanup_pencils` | 8 |
| `unitree_g1_pack_camera` | 11 |

输出目录：

```text
~/benchmark-runs/unifolm/<本次运行时间>/
├── result.json       ← 提交这个 JSON
├── runtime.log       ← 同一次运行的完整日志
├── profiles/         ← 分项计时和 PSNR 报告
├── outputs/          ← 生成的视频
├── environment.txt
└── gpu-info.txt
```

FP32 完整运行在此前环境中约 90 分钟，仅作耗时参考；共享负载和代码变化会影响实际时间。建议在 `tmux` 等持久会话内运行，避免 SSH 断开中止任务（使用前先检查工具是否已安装）。

### 6. 修改代码后如何运行

#### MPI：使用自己的源码与参考输出

首次先运行第 5 节的 MPI baseline，确保已生成并核验参考输出。随后修改自己的实现，再运行：

```bash
cd ~/competition/mpi-attention-baseline
bash scripts/run_benchmark.sh \
  --manifest manifest.json \
  --ref-dir "$HOME/benchmark-cache/mpi/reference" \
  --out-dir "$HOME/benchmark-runs/mpi-custom/$(date -u +%Y%m%dT%H%M%SZ)-$$" \
  --runs 5 --proc 16
```

如果自己的现有仓库是 `~/problems/mpi-attention`，将 `cd` 路径替换为它即可。`--proc` 可以按比赛允许的资源调整；不能绕过冻结文件、工作量和正确性约束。

该入口默认保存 `runtime.log`，并生成含状态字段的 `result.json`；底层中间报告 `results.json` 在转换完成后自动删除。共享入口和当前 team01 的已有入口已更新；较早复制的其他私人脚本可能仍是旧版本，运行前应确认结果格式。

#### UnifoLM：使用自己的优化代码

在第 4 节准备的私人源码副本内执行：

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
export RESULT_DIR="$HOME/benchmark-runs/unifolm-custom/$(date -u +%Y%m%dT%H%M%SZ)-$$"
export PROFILE_DIR="$RESULT_DIR/profiles"
export RESULT_JSON_PATH="$RESULT_DIR/result.json"
export LOG_PATH="$RESULT_DIR/runtime.log"
export INPUT_CACHE_DIR=/share/problems/unifolm-world-model-action/examples/compute_benchmark/preprocessed_inputs
bash scripts/run_compute_benchmark.sh
```

入口默认通过 `tee` 同时显示和保存标准输出、错误输出；最终 JSON 和日志包含运行时长、分项时长与 PSNR。每次运行必须使用新目录，已有结果路径会被拒绝，以免覆盖证据。

#### TF32 与正确性：本次实测经验

默认 FP32 baseline 的 `torch.backends.cudnn.allow_tf32` 和 `torch.backends.cuda.matmul.allow_tf32` 均关闭。自己的优化代码可以尝试 TF32，但必须重新校验 PSNR。

针对 `unitree_z1_dual_arm_stackbox_v2`，此前完整 11 轮实测如下：

| TF32 设置 | PSNR | 是否满足严格大于 25 dB |
| --- | ---: | --- |
| 仅 cuDNN 开启 | 23.45 dB | 否 |
| 仅矩阵乘法开启 | 22.48 dB | 否 |
| 两者同时开启 | 20.91 dB | 否 |

这说明该用例不能直接套用上述全局开关方案。其他四项在此前两种 TF32 同开的一次完整运行中通过了 PSNR，但不能据此免除之后运行的校验。

### 7. 正确性与计分规则

#### MPI Attention：120 分

`short`、`mixed`、`long` 各 40 分。评测用 FP64 参考输出进行正确性检查，manifest 的相对 Frobenius 误差容差为 `0.01`（1%）。公共用例的形状和随机种子固定，冻结文件必须保持不变。

每项得分：

```text
40 × 本队该项有效吞吐量 / 全场该项最高有效吞吐量
```

吞吐量单位为 GF/s，越大越好。总几何平均吞吐仅作整体展示，评分按三个分项分别计算。

#### UnifoLM：200 分

五项各 40 分。每项生成视频与共享 FP32 reference 比较，要求 **PSNR 严格大于 25 dB**，25 dB 本身不通过。精确一致时 PSNR 为无穷大，由官方脚本用相应 JSON 字段表示。

每项得分：

```text
40 × 全场该项最短有效 process_wall_s / 本队该项有效 process_wall_s
```

`process_wall_s` 越小越好，包含该项进程的导入、模型加载、推理和必要的输出写入；`pure_model_cuda_s` 用于诊断，不作为该项评分时间。PSNR 校验耗时单独记录，不计入推理性能时间。

#### 历史最好与部分通过

总分为 320 分。榜单对每队、每个小项分别取历史最好有效成绩，不要求所有最优小项来自同一次提交。后来某项失败，不会抹去该项此前的有效成绩。

某小项失败、缺失或证据不一致，该次该项得 0 分，其余通过的小项可以计分；平台可能显示「部分通过」。JSON 无法解析、整份 JSON 与 log 最终状态不一致等整份材料错误仍会使本次提交校验失败。待管理员复核的成绩在获准前不计分。

每个小项的全场最快成绩更新后，所有队伍该项的百分比会重新计算。

### 8. 登录网站并提交

访问：<http://10.120.20.219:8000>。

1. 输入自己的队伍名及管理员派发的队伍 token 登录。
2. 打开提交页面，选择 `MPI Attention` 或 `UnifoLM`。
3. 同时上传 **同一次运行** 的 `result.json` 和 `runtime.log`。
4. 在「我的提交」中查看解析状态、失败原因、分项成绩和运行日志。

每个文件最多 5 MiB，使用原始 UTF-8 文本文件。不要上传视频、压缩包、单项诊断 JSON，也不要混用两次运行的材料。不要手工添加 `accepted` 或修改指标来绕过校验。

后台核对 JSON 与 log，不在服务器自动重新运行选手代码。无需配置 Git、提供 Git SHA 或启动评测容器。

网站 token 与 SSH 私钥不同。队伍 token 只用于自己的登录和提交，不要写进代码仓库或分享给其他队伍。

#### 可选：命令行上传

将以下两个路径替换为脚本最后打印的实际文件路径；`problem=mpi` 用于 MPI，UnifoLM 改为 `problem=unifolm`。

```bash
read -rsp '队伍 token: ' TEAM_TOKEN; echo
curl --fail-with-body -X POST http://10.120.20.219:8000/api/submissions \
  -H "X-Team-Token: ${TEAM_TOKEN}" \
  -F 'problem=mpi' \
  -F 'result_file=@/home/team01/benchmark-runs/mpi/本次运行目录/result.json' \
  -F 'log_file=@/home/team01/benchmark-runs/mpi/本次运行目录/runtime.log'
unset TEAM_TOKEN
```

HTTP 请求成功不等于校验通过，仍需检查返回的 `status`、`verification` 和 `reason`。

### 9. 常见问题

#### 复制赛题后无法 mkdir 或修改文件

共享原件是只读的，复制会继承权限。只对自己的副本补权限，例如：

```bash
chmod -R u+rwX ~/competition/mpi-attention-baseline
chmod -R u+rwX ~/competition/unifolm-world-model-action
```

若仍失败，检查 `ls -ld` 的所有者和 `df -h "$HOME"` 的剩余空间，再联系管理员。不要对 `/share` 执行递归改权限。

#### 「结果 JSON 必须有 accepted/rejected/failed/timeout 状态」

检查是否误选旧版底层 `results.json`。应提交单数的 `result.json`，与同目录、同次运行的 `runtime.log` 配对。当前更新后的入口会自动整理并删除中间报告。

#### 没有日志，或日志与 JSON 不匹配

使用本文的运行入口，它们自动保存日志。不要把多个运行的输出追加到同一个 log 中，不要把终端截取的片段当完整运行日志上传。缺少原始日志时应重新运行，不能补造日志。

#### UnifoLM 仍尝试下载权重

依次检查：主 ckpt 的链接是否可读、`CKPT_PATH` 是否正确、两个 Hugging Face 缓存变量是否指向 `/share/huggingface/hub`、离线开关是否设置。主 ckpt 和 CLIP 是两套权重，检查其中一个不能代替另一个。

#### UnifoLM 退出非零，但部分视频已生成

检查 `runtime.log` 和 `profiles/psnr.json`。PSNR 不通过也会返回非零，并保留结果与日志。可以提交真实的成对材料，由平台独立判断有效小项；不要只改 JSON 状态为成功。

#### 磁盘空间不够

```bash
df -h "$HOME"
du -sh ~/miniconda3 ~/.conda ~/.cache ~/benchmark-runs 2>/dev/null
```

权重使用共享软链接，参考视频直接读取 `/share`，避免把它们复制进 home。自己的环境、依赖缓存、MPI 参考输出和生成视频仍占配额。清理旧产物前确认不再需要复现或管理员抽查，优先保留每次提交的原始 `result.json`、`runtime.log` 和必要的配置记录。

### 10. 参赛规则、获奖交付与 2.0.1 更新

完整规则、每题优化方向、禁止事项及获奖队伍必须提交的代码和报告，见同目录 [PARTICIPANT_HANDBOOK.md](PARTICIPANT_HANDBOOK.md)。

2.0.1 修复 MPI 冻结汇总器及自测，已有私人副本请执行 `python3 /share/tools/update-mpi-release.py --repo <自己的MPI仓库>` 后 `make test`，不要仅覆盖 manifest 或覆盖自己的实现。旧成绩保留，升级后新跑成绩使用新版。

UnifoLM 新入口按项记录退出码、继续健康项，并在结束时补校验。已有副本需同步新版 `scripts/run_compute_benchmark.sh`；共享工具已统一更新。缺项/损坏不清空其他项，执行失败或工作量不完整仍不能计分。

---

<a id="english"></a>

## English

Environment: the Ubuntu 24.04 player container on the competition server. Documentation updated: 2026-10-20.

Run all commands as your own player account, without `sudo`. Use the participant username assigned by the organizer, for example `team01`. The organizer distributes SSH private keys and website tokens separately; they serve different purposes.

### 1. Login and directory layout

Connect from your own computer, for example:

```bash
ssh -i /path/on/your/computer/team01_ed25519 -p 22000 team01@10.120.20.219
```

If the private key permissions are too open, run `chmod 600 /path/on/your/computer/team01_ed25519` locally first. Replace the account and key path with those assigned to you.

| Path inside the container | Purpose |
| --- | --- |
| `~`, or `/home/<team-account>` | Your code, environments, caches, and results; each player has a 50 GiB capacity limit |
| `/share/problems` | Read-only problem packages; copy them into your home before editing |
| `/share/weights` | Downloaded main-model and CLIP weights, shared read-only |
| `/share/huggingface/hub` | Prepared Hugging Face offline model cache |
| `/share/results/unifolm/fp32/reference/outputs` | FP32 reference videos for UnifoLM correctness checks |
| `/share/tools` | Environment installers, baseline runners, result finalizers, and PSNR tools |
| `~/benchmark-runs` | Separate records for each baseline run |

The host shared directory is `/data/newbie-data/share`, mounted as `/share` inside the player container. Each home corresponds to `/data/newbie-data/home/<team-account>` on the host.

`/share` is read-only. Copied files may retain read-only permissions, so make your own copy writable. Run this example only when the destination does not already exist:

```bash
mkdir -p ~/competition
cp -r /share/problems/mpi-attention ~/competition/mpi-attention-baseline
chmod -R u+rwX ~/competition/mpi-attention-baseline
```

Do not repeatedly copy into an existing destination, which may create an extra nested directory. The baseline helpers copy the source and adjust permissions automatically when their source directory does not exist.

### 2. Check the basic environment

The container already provides compilation tools, OpenMPI, and other system tools. Each team installs its own Python/Conda environments inside its home.

```bash
whoami
printf 'HOME=%s\n' "$HOME"
df -h "$HOME"
command -v gcc make mpicc mpirun python3
mpirun --version
nvidia-smi
mkdir -p ~/.tmp
export TMPDIR="$HOME/.tmp"
```

The player configuration shares host GPUs 4, 5, 6, and 7: four A800 80GB GPUs. CPU and memory have no container quotas. Resource configuration changes take effect after the organizer recreates the container; check the devices actually visible with `nvidia-smi -L`. Use the CUDA logical indices enumerated inside the container, rather than assuming host GPU numbers work inside it. The existing baseline uses one visible GPU by default. Resources are shared, so other jobs affect performance; avoid launching several complete benchmarks simultaneously.

### 3. Install your own Miniconda and UnifoLM environment

The MPI baseline does not need Conda; you may skip to Section 5.

#### First installation

```bash
mkdir -p ~/.tmp
export TMPDIR="$HOME/.tmp"
bash /share/tools/install-miniconda.sh
~/miniconda3/bin/python -m pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
bash /share/tools/install-unifolm-env.sh
```

If `~/miniconda3` is already installed, skip its installer. The second script creates your private `~/miniconda3/envs/unifolm` environment with Python 3.10.18, PyTorch 2.3.1+cu121, torchvision 0.18.1+cu121, and the benchmark dependencies.

Set your account's default pip index to Aliyun before installing dependencies. If a Conda environment is already activated, you may also run:

```bash
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
```

This setting applies to pip installations without an explicitly specified index. The install script still fetches the CUDA 12.1 PyTorch packages from the explicitly specified official PyTorch index. This command does not change Conda channels.

Environment installation requires access to package repositories. Shared weights avoid another model download; they do not replace Conda/pip dependency installation.

#### Activate and check

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ~/miniconda3/envs/unifolm
python -c 'import torch; print(torch.__version__); print(torch.cuda.is_available())'
```

Repeat these activation commands after logging in again. Environments and package caches count toward your 50 GiB limit. Do not copy another team's Conda environment.

### 4. Link the shared checkpoint instead of downloading it again

The main checkpoint is approximately 16 GB and the CLIP weights approximately 3.7 GB. Both are already shared. Read them directly or create symbolic links instead of downloading or copying them.

Prepare your own editable UnifoLM source copy first, only if the destination does not already exist:

```bash
mkdir -p ~/competition
cp -r /share/problems/unifolm-world-model-action ~/competition/unifolm-world-model-action
chmod -R u+rwX ~/competition/unifolm-world-model-action
cd ~/competition/unifolm-world-model-action
```

#### Option A: symlink into the default downloads location

Run once from the root of your own UnifoLM repository:

```bash
mkdir -p downloads/UnifoLM-WMA-0-Dual
ln -s /share/weights/UnifoLM-WMA-0-Dual/unifolm_wma_dual.ckpt \
  downloads/UnifoLM-WMA-0-Dual/unifolm_wma_dual.ckpt

readlink -f downloads/UnifoLM-WMA-0-Dual/unifolm_wma_dual.ckpt
test -r downloads/UnifoLM-WMA-0-Dual/unifolm_wma_dual.ckpt && echo 'CKPT readable'
```

If you see `File exists`, inspect the existing file or link with `ls -l`; an already-correct link needs no replacement. A symlink leaves the actual model data in `/share`, rather than consuming another 16 GB in your home.

#### Option B: set the shared checkpoint path directly

You can skip the link and set this before running:

```bash
export CKPT_PATH=/share/weights/UnifoLM-WMA-0-Dual/unifolm_wma_dual.ckpt
```

`CKPT_PATH` overrides the default `downloads/...` location. The baseline helper already uses this shared path directly, so no manual link is needed for it.

#### CLIP weights and offline cache

The program also loads CLIP through OpenCLIP. Linking only the main checkpoint does not stop CLIP from accessing the network. Set these variables before running:

```bash
export HF_HUB_CACHE=/share/huggingface/hub
export HUGGINGFACE_HUB_CACHE=/share/huggingface/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

The prepared cache already resolves to the shared CLIP model. Normally, you do not need another CLIP link inside your repository. If your modified code explicitly reads local CLIP weights, use:

```text
/share/weights/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin
```

These environment variables affect the current shell. The UnifoLM baseline helper sets them automatically.

### 5. Run a complete baseline quickly

#### MPI Attention: one command

```bash
python3 /share/tools/run-mpi-baseline.py
```

The helper uses your `~/competition/mpi-attention-baseline` directory, copying shared source only if it does not exist. It keeps an existing copy and your changes. It builds the program, generates or reuses FP64 reference outputs in `~/benchmark-cache/mpi/reference`, and checks their SHA-256 values against the manifest.

The default is 16 MPI ranks with one OpenMP thread per rank. All three cases run with one warm-up and five timed repetitions per case, using the median.

Output layout:

```text
~/benchmark-runs/mpi/<run-id>/
├── result.json       # Upload this JSON
├── runtime.log       # Complete log from the same run
└── outputs/          # Per-case artifacts
```

The helper prints the exact `RESULT_JSON` and `RUNTIME_LOG` paths at the end. A public baseline may require organizer review; waiting for review does not mean execution failed.

#### UnifoLM: one command

Install your private environment in Section 3, then run:

```bash
python3 /share/tools/run-unifolm-baseline.py
```

The helper uses `~/competition/unifolm-fp32-baseline` and copies the source if needed. It requires both TF32 switches to remain disabled for an FP32 baseline. Shared weights, offline CLIP cache, preprocessed inputs, and reference videos are used automatically; no additional checkpoint download is needed.

The complete benchmark has five cases, 49 rounds, and 103 output videos. Reduced-round runs cannot be submitted as complete results.

| Case | Required rounds |
| --- | ---: |
| `unitree_z1_stackbox` | 12 |
| `unitree_z1_dual_arm_stackbox` | 7 |
| `unitree_z1_dual_arm_stackbox_v2` | 11 |
| `unitree_z1_dual_arm_cleanup_pencils` | 8 |
| `unitree_g1_pack_camera` | 11 |

Output layout:

```text
~/benchmark-runs/unifolm/<run-id>/
├── result.json       # Upload this JSON
├── runtime.log       # Complete log from the same run
├── profiles/         # Per-case timing and PSNR reports
├── outputs/          # Generated videos
├── environment.txt
└── gpu-info.txt
```

A previous complete FP32 run took approximately 90 minutes; this is a historical estimate, not a guaranteed duration. Shared load and code changes affect runtime. Use a persistent session such as `tmux` to prevent an SSH disconnection from stopping the job; check that the tool is installed first.

### 6. Run after modifying the code

#### MPI: your own source and reference outputs

Run the MPI baseline in Section 5 first so that reference outputs have been generated and checked. After editing your implementation, run:

```bash
cd ~/competition/mpi-attention-baseline
bash scripts/run_benchmark.sh \
  --manifest manifest.json \
  --ref-dir "$HOME/benchmark-cache/mpi/reference" \
  --out-dir "$HOME/benchmark-runs/mpi-custom/$(date -u +%Y%m%dT%H%M%SZ)-$$" \
  --runs 5 --proc 16
```

If your repository is at `~/problems/mpi-attention`, change the `cd` path accordingly. You may adjust `--proc` within the permitted resources, while preserving frozen files, workload, and correctness requirements.

The launcher saves `runtime.log` and produces `result.json` with a final status. It removes the intermediate `results.json` after conversion. The shared launcher and the team01 copy updated earlier contain this behavior; other private copies made earlier may still be old. Check your result format before submitting.

#### UnifoLM: your optimized source

Run from the private copy prepared in Section 4:

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
export RESULT_DIR="$HOME/benchmark-runs/unifolm-custom/$(date -u +%Y%m%dT%H%M%SZ)-$$"
export PROFILE_DIR="$RESULT_DIR/profiles"
export RESULT_JSON_PATH="$RESULT_DIR/result.json"
export LOG_PATH="$RESULT_DIR/runtime.log"
export INPUT_CACHE_DIR=/share/problems/unifolm-world-model-action/examples/compute_benchmark/preprocessed_inputs
bash scripts/run_compute_benchmark.sh
```

The launcher uses `tee` to display and save stdout and stderr. The JSON and log include overall runtime, per-case timings, and PSNR. Use a new output directory for every run; existing result/log paths are rejected to preserve evidence.

#### TF32 and correctness: previous measurements

The FP32 baseline disables both `torch.backends.cudnn.allow_tf32` and `torch.backends.cuda.matmul.allow_tf32`. You may experiment with TF32 in your own code, but must recheck PSNR.

Previous complete 11-round measurements for `unitree_z1_dual_arm_stackbox_v2` were:

| TF32 setting | PSNR | Strictly above 25 dB? |
| --- | ---: | --- |
| cuDNN only | 23.45 dB | No |
| Matrix multiplication only | 22.48 dB | No |
| Both enabled | 20.91 dB | No |

These global-switch configurations did not qualify for this case. The other four cases passed PSNR in one previous run with both switches enabled; that does not exempt subsequent runs from validation.

### 7. Correctness and scoring

#### MPI Attention: 120 points

`short`, `mixed`, and `long` are worth 40 points each. Correctness is checked against FP64 reference outputs with the manifest's relative Frobenius error tolerance of `0.01` (1%). Public case shapes and the random seed are fixed; frozen files must remain unchanged.

Each case scores:

```text
40 × your team's valid throughput / highest valid throughput for that case
```

Throughput is measured in GF/s; higher is better. The overall geometric mean is diagnostic; points are calculated separately for the three cases.

#### UnifoLM: 200 points

Each of five cases is worth 40 points. Generated videos are compared with the shared FP32 reference. **PSNR must be strictly greater than 25 dB**; exactly 25 dB fails. An exact match has infinite PSNR, represented by the official tool's JSON fields.

Each case scores:

```text
40 × shortest valid process_wall_s for that case / your team's valid process_wall_s
```

Lower `process_wall_s` is better. It includes process imports, model loading, inference, and required output writes. `pure_model_cuda_s` is diagnostic and does not determine case points. PSNR validation time is recorded separately and excluded from inference performance time.

#### Historical best and partial results

The total maximum is 320 points. For each team and case, the leaderboard takes the best valid historical result. Your best cases need not come from the same submission. A later failure does not erase a previous valid result.

A failed, missing, or inconsistent case earns zero for that case in that submission; other validated cases can still contribute, and the platform may show “部分通过” (partially accepted). Whole-material errors, such as invalid JSON or a final-status mismatch between JSON and log, still fail the entire submission. Pending-review results do not count until approved.

When a new fastest result appears for a case, every team's percentage score for that case is recalculated.

### 8. Log in and submit

Visit <http://10.120.20.219:8000>.

1. Log in with your team account and the team token distributed by the organizer.
2. Open the submission page and choose `MPI Attention` or `UnifoLM`.
3. Upload `result.json` and `runtime.log` **from the same run**.
4. Check parsing status, reasons for failure, per-case scores, and logs under “我的提交” (My Submissions).

Each file must be at most 5 MiB and contain original UTF-8 text. Do not upload videos, archives, single-case diagnostic JSON, or mixed-run materials. Do not manually add `accepted` or change metrics to bypass validation.

The backend compares JSON and log; it does not automatically rerun your code. No Git setup, Git SHA, or evaluation container is required for submission.

Website tokens differ from SSH private keys. Use your token only for your own login and uploads; do not commit it or share it with other teams.

#### Optional: upload from the command line

Replace both file paths below with the actual paths printed by your runner. Use `problem=mpi` for MPI and `problem=unifolm` for UnifoLM.

```bash
read -rsp 'Team token: ' TEAM_TOKEN; echo
curl --fail-with-body -X POST http://10.120.20.219:8000/api/submissions \
  -H "X-Team-Token: ${TEAM_TOKEN}" \
  -F 'problem=mpi' \
  -F 'result_file=@/home/team01/benchmark-runs/mpi/RUN_ID/result.json' \
  -F 'log_file=@/home/team01/benchmark-runs/mpi/RUN_ID/runtime.log'
unset TEAM_TOKEN
```

A successful HTTP request does not guarantee acceptance. Check the returned `status`, `verification`, and `reason`.

### 9. Troubleshooting

#### Cannot create directories or edit copied problem files

Copies inherit the read-only permissions of the shared original. Make only your own copy writable:

```bash
chmod -R u+rwX ~/competition/mpi-attention-baseline
chmod -R u+rwX ~/competition/unifolm-world-model-action
```

If it still fails, check ownership with `ls -ld` and space with `df -h "$HOME"`, then contact the organizer. Do not recursively change permissions on `/share`.

#### “结果 JSON 必须有 accepted/rejected/failed/timeout 状态”

This means the result JSON needs a supported final status. Check whether you selected an old raw `results.json` by mistake. Upload the singular `result.json` with `runtime.log` from the same directory and run. Updated launchers finalize the report and remove the intermediate file automatically.

#### Missing log, or JSON and log do not match

Use the launchers in this guide, which save logs automatically. Do not append multiple runs to one log or submit a terminal excerpt as a complete log. If the original log is missing, rerun the benchmark rather than fabricating it.

#### UnifoLM still attempts to download weights

Check the checkpoint link, `CKPT_PATH`, both Hugging Face cache variables pointing to `/share/huggingface/hub`, and the offline switches. The main checkpoint and CLIP are separate sets of weights; checking one does not verify the other.

#### UnifoLM exits nonzero after producing some videos

Inspect `runtime.log` and `profiles/psnr.json`. A PSNR failure also produces a nonzero exit while preserving JSON/log. You may upload the genuine paired materials for independent per-case decisions. Do not change only the JSON status to success.

#### Running out of space

```bash
df -h "$HOME"
du -sh ~/miniconda3 ~/.conda ~/.cache ~/benchmark-runs 2>/dev/null
```

Link shared weights and read reference videos directly from `/share`, rather than copying them into your home. Your environment, dependency caches, MPI reference outputs, and generated videos still consume your allocation. Before deleting old artifacts, make sure they are no longer needed for reproduction or review. Prioritize keeping original `result.json`, `runtime.log`, and the necessary configuration records for each submission.

### 10. Rules, winner deliverables, and the 2.0.1 update

See [PARTICIPANT_HANDBOOK.md](PARTICIPANT_HANDBOOK.md#english) for complete rules, optimization ideas, prohibited actions, and the source code and report required from award-winning teams.

Version 2.0.1 fixes the frozen MPI aggregator and its self-test. For an existing private copy, run `python3 /share/tools/update-mpi-release.py --repo <your-MPI-repository>` and then `make test`. Do not replace only the manifest or overwrite your own implementation. Historical results are retained; rerun after upgrading to produce a new-version result.

The new UnifoLM launcher records per-case exit codes, continues other independent cases, and validates completed cases at exit. Existing copies need the updated `scripts/run_compute_benchmark.sh`; the shared tools have already been updated. Missing or corrupt cases do not erase healthy evidence, while failed execution or incomplete work still cannot score.
