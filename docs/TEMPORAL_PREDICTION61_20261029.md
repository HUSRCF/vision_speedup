# 两次刷新预测：13块质量通过，23块残差失败

本轮不再直接读取上一次动态特征，而是保存最近两次**完整刷新**的结果，用它们的变化趋势预测当前特征。Z1两个独立单卡试验均完整执行12轮、两阶段各600次调用、25视频，并通过全程干扰检查。

| 范围 | 本轮 process_wall_s | 本轮 PSNR dB | 历史同范围直接缓存 PSNR dB | 结果 |
|---|---:|---:|---:|---|
| 13块完整特征 | 161.216743 | 29.507847903 | 24.469409469 | 质量通过，可继续测双卡组合 |
| 23块，当前输入＋预测残差 | 138.840686 | 19.499227323 | 19.504188169 | 质量未通过 |

当前正式Z1为150.838389秒、五项764.366152秒，均为双卡。这两次单卡试验未替换正式材料，没有新五项103视频验收，不能据此宣称双卡速度或质量。历史对比不是同期配对实验；13块历史直接缓存耗时161.281284秒，现有数据支持继续研究质量补偿，不足以证明单卡速度改进。

对当前采样位置 `p`，最近两次完整刷新的位置为 `q0 < q1 < p`：

```text
alpha = (p - q1) / (q1 - q0)
prediction = latest + (latest - previous) * alpha
```

复用窗口仍为位置10到39，每隔一步复用，实际复用位置11、13、…、39。第一次位置11使用9、10的刷新，系数为1；随后位置13使用10、12，系数为0.5。每阶段每轮15次复用，35次完整刷新，**所有50次DDIM更新均执行**。12轮两阶段合计系数1出现24次、0.5出现336次。

13块范围为 `input_blocks[6:]`、middle、`output_blocks[:6]`；23块为 `input_blocks[1:]`、middle、`output_blocks[:11]`。后者在出口处对跨边界残差作预测，再加当前边界输入。差值和残差预测使用FP32，普通完整特征历史保持原dtype，输出转回原dtype；浅层、剩余输出块、head与sampler更新继续执行。

历史只能由真正完成的采样步骤推进，CUDA Graph捕获和额外warm-up不算新的时间步。每次新生成段清空历史；权重版本或条件变化后，必须重新完成两次完整刷新才能再预测。不跨轮读取上一轮动态特征，不读取参考视频或已有最终视频。

新增 Triton kernel 仅处理声明的普通Torch Tensor，固定尾块掩码和禁用FMA，计算顺序与Torch参考一致。完整计时内的GPU自检覆盖Half／FP32、直接／残差四组，257元素尾块，捕获后修改历史与系数再重放，输出与参考相同。CPU检查覆盖3600个仿射模型完整输出、实际刷新间隔、失效后两次刷新、重复warm-up、RNG与漏步拒绝。**这些测试不代替完整闭环视频PSNR。**

两个61文件公开入口分别在比赛机禁用CUDA精确重建，全部计时源码与资源SHA一致，并重新执行CPU预测与显式阶段识别检查。原59文件数学路径保留，只有CUDA Graph增加可选共享池，再增加两个动态复用模块。两阶段按条件里的显式 `sim_mode` 识别，不根据TF32开关猜测。

- [冻结源码及原guard](../experiments/temporal-prediction61)
- [完整log／JSON、CPU重建与GPU遥测](../results/aggressive-20261029/temporal-prediction61)
- [复现入口](../reproduce/run_temporal_prediction61.py)

```bash
python reproduce/verify_temporal_prediction61.py
python reproduce/run_temporal_prediction61.py --trial deep13-r1 --gpu 1 \
  --workdir /run/lock/team07-review-predict13 --prepare-only
python reproduce/run_temporal_prediction61.py --trial residual23-r1 --gpu 3 \
  --workdir /run/lock/team07-review-predict-residual23 --prepare-only
```

完整运行去掉 `--prepare-only`，需授权模型、输入、官方验证工具和空闲GPU。工作目录及新launcher缓存需要可执行挂载，原warm cache保持只读；新kernel使用独立常量标签，原数值与缓存门槛不放宽。初始化、图捕获、自检、视频写出均在原process timer内。单项原始log／JSON包含另外四项缺项，不能直接作为五项提交。
