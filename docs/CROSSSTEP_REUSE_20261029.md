# V2 跨 DDIM 时间步特征复用

本轮重点转向相邻时间步之间的动态 U-Net 特征复用。用户明确要求保留全部轮数；实现保留 V2 的 11 轮、decision/world 各轮 50 步、23 个输出视频。该候选尚未替换当前 225 双卡正式结果。

## 第一版策略

只在 world 阶段复用深层特征。每轮按采样执行顺序编号 0–49，位置 11、13、…、39 使用最近一次完整刷新产生的特征，共 15 步；其余 35 步完整计算。decision 的全部 50 步保持原路径。

缓存范围为 `input_blocks[6:]`、`middle_block`、`output_blocks[:6]`，合计 13 个块。浅层 input blocks、后六个 output blocks、action/state head 和每步 DDIM 更新继续计算。缓存只来自本次运行中前面的真实网络计算，不使用参考视频或最终输出缓存。它会改变模型近似误差，需要官方完整视频 PSNR 独立验收。

每轮开始刷新；参数版本或条件变化强制刷新；不允许丢失或重复采样位置。缓存使用普通 Torch Tensor，在计时内的 eager warm-up 分配。完整 world 图更新缓存，复用图读取缓存并返回副本；两图顺序执行、共享图内存池，避免第三个独立激活池占满显存。原图框架仅增加 `pool` 参数，原始卷积、GEMM、归一化等算术不变。

思路参考 [DeepCache 官方实现](https://github.com/horseee/DeepCache) 的高层特征跨步复用。当前实现采用固定刷新窗口，尚未使用 TeaCache 的动态阈值或声称其论文中的加速比适用于本模型。

## 已完成检查及其边界

- CPU：完整 11 轮、两个阶段、每阶段 50 步，共 1100 次 sampler 调用。浅层和 head 1100 次，所选深层每块实际计算 935 次，复用 165 次。参数／条件／轮次失效、RNG、跳步拒绝检查通过。
- A800 小模型：真实 `DenoiserGraph` 的刷新／复用图切换，共 200 次调用、三个输出的 600 次逐位检查通过。包含参数和条件中途改变后的强制刷新、共享内存池与 RNG 检查。
- 这些逐位检查比较的是“相同复用策略的 eager 和 graph”，不表示复用策略与原始每步全算的结果逐位相同。
- 真实 V2 的 r3／r4／r5 完整 11 轮已完成，见下表；r6 继续验证。不能用小模型检查代替完整视频质量检查。

真实首跑 r1 在第一个 world 刷新图构建时，由原有 Kahan CUBIN 指纹检查中止，尚未执行复用，完成轮数为 0，没有有效成绩。新缓存重新编译的二进制与既有 witness 不同；相同缓存键的原有二进制 SHA 则匹配 witness。r2 改为只读已有 warm cache、新键独立写入，保留全部原校验和相同 211 生产源码。失败范围保存在 [failed-r1](../results/aggressive-20261029/crossstep211/failed-r1/summary.json)，没有把它视为精度失败或有效计时。

r2 已通过完整 world 图（分配约31.144GiB），随后在首次复用图构建时触发旧 head 权重缓存的完整23卷积断言，仍未完成一轮、没有有效成绩。[失败范围](../results/aggressive-20261029/crossstep211/failed-r2/summary.json)保留。r3 仅让复用图使用原生 head 权重变换，刷新图保留原缓存和全部断言；真实复用图分配约32.452GiB，完整结果如下。

另一张空闲卡运行 r4 保守版本：只复用 `input_blocks[9:]`、middle、`output_blocks[:3]`，合计7块，仍是同样15个world位置。r3／r4 的精确源码、参数和计时入口分别见 [full-r3](../experiments/crossstep211/full-r3)／[full-r4](../experiments/crossstep211/full-r4)。

## 完整结果：直接输出复用未通过

| 候选 | 完整进程 | PSNR | 结论 |
|---|---:|---:|---|
| r3：13块 | 304.517598秒 | 20.962911dB | 质量失败 |
| r4：7块 | 320.608958秒 | 20.626567dB | 质量失败 |
| r5：7块残差 | 317.579932秒 | 22.346030dB | 质量失败 |

三者都完成11轮、23视频、decision/world各550次调用、165次world复用，三个CUDA Graph，独占guard通过。这里的 `clean` 仅表示实验未受外部GPU上下文干扰，不表示视频质量通过。单项结果JSON中其他四项缺失，也不构成五项正式提交。

r3 的decision/world生成墙钟为135.007356／160.267098秒；r4为137.233012／173.995529秒。r3对历史原209单卡316.774396秒少12.256798秒（3.87%），但不是同期配对，且质量失败；r4没有显示完整进程改善。不能与当前225双卡257.945205秒直接比较优化因果。

逐视频结果显示：r3初次decision为47.102dB、初次world为26.006dB，下一轮decision降至20.892dB；r4初次world为32.456dB，但下一轮decision也降至21.094dB。它们支持“闭环误差放大”的判断，尚不能单独区分视频反馈与state head的贡献。官方 `ddim_eta=1.0` 保持原值；没有为复用改成确定性采样。

NVML峰值显存分别76236／77522MiB、最高温度66／73°C，活跃频率存在波动，完整遥测随结果保留。上述32GiB左右是Torch allocated，不等于进程峰值占用。

## r5：保持当前分支输入，复用残差

7块分支的入口和出口具有相同形状。完整刷新时保存 `delta = branch_output - branch_input`，复用时使用 `current_branch_input + cached_delta`，避免整段输出直接停留在上一时间步。其余深层内部特征仍缓存，head仍执行，最终质量必须重新测量。

新增CPU检查验证新输入确实穿过残差边界；新的真实DenoiserGraph小模型完成200调用／600输出逐位检查，包括参数／条件失效与RNG。该检查依然只比较相同策略的eager与graph，不表示与全算等价。[冻结源码](../experiments/crossstep211/full-r5)与[残差检查](../results/aggressive-20261029/crossstep211/residual-sanity)已经归档。

r5完整11轮为 **317.579932秒／22.346030dB**。相比r4质量提高1.719463dB，但两次来自不同GPU且不是重复配对；质量仍失败，也没有相对原209单卡316.774396秒的完整进程改善。它不支持直接集成正式225。

## r6：更早、更窄的窗口

正在验证另一个完整11轮候选：`input_blocks[5:]`、middle、`output_blocks[:8]`，合计16块，入口／出口同为640通道。只在world位置2、4、6、8、10复用残差，之后11–49连续39步完整计算，每轮50步不变，11轮预期55次复用。目的是检查更长的后续完整计算能否改善闭环质量，尚无加速或质量结论。[冻结源码和参数](../experiments/crossstep211/full-r6)与上一版分开保留。

当前原型基于原始 **209 单卡路径＋2 个模块＝211 文件**，不是当前正式的 225 双卡路径。需要先验证复用质量，再组合双卡分片并作完整进程对照。不能直接将单卡原型与正式双卡耗时作因果比较。

## 审阅文件

- [实现与冻结清单](../experiments/crossstep211)：3 个生产 overlay 文件，其余 208 文件保持原始 209 清单字节；CPU 和 GPU 小模型检查脚本也已保留。
- [检查结果](../results/aggressive-20261029/crossstep211)：标量报告和遥测，不包含权重、输入、特征张量或视频。
- [完整计时入口](../experiments/crossstep211/timing-assets/scripts/evaluation/world_model_interaction_crossstep.py)：原始进程计时开始位置保留，新增安装、图捕获、缓存分配、计算和退出均在计时内。
- `full-run_crossstep_full211.py` 和 `full-guard_crossstep211.py` 是完整实验的归档名称；独立实验目录运行时分别恢复为 `run_crossstep_full211.py` 和 `guard_crossstep211.py`。原始 guard 与完整 guard 分别保留哈希。
- `python3 reproduce/verify_crossstep211.py` 校验冻结源码和现有证据的范围。正式复现入口仍保持 225。

独立重建入口为 `reproduce/run_crossstep211.py`。以下命令仅重建并核对211文件，不导入Torch或申请GPU，已在本机执行通过：

```bash
python3 reproduce/run_crossstep211.py --revision 5 \
  --output /tmp/crossstep211-review --prepare-only
```

比赛机上的完整实验去掉 `--prepare-only`，使用新的 `--output`、指定 `--gpu` 和可执行文件系统上的新 `--cache-dir`（例如私有 `/run/lock/team07-crossstep-review-cache`）。需要原授权权重、输入、参考视频、既有依赖和warm Triton缓存；原CUBIN指纹检查保留，不能声称可在任意空环境冷编译复现。入口与guard都检查空卡，guard沿用team07锁和全过程外部上下文排除。该命令是实验复现入口，未成为默认正式方案。
