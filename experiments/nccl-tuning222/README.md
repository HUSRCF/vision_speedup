# NCCL tuning on the adopted222 payload geometry

The initial screen keeps all model/source/arithmetic changes out of the test. Its input is the adopted222 controller's FP32 local all-gather payload histogram:38 decision calls/450MiB and121 world calls/1399.375MiB per rank per forward. Only five sizes occur:1.5625,6.25,12.5,25,40MiB. Histograms are grouped by size; they are not the actual model operation order or overlap pattern.

Each configuration starts fresh rank processes under the existing two-GPU locks and whole-attempt context monitor. Tests use a patterned synthetic FP32 input, verify initial/changed/zero/restored bit patterns in both ranks, then time32 consecutive all-gathers in one CUDA Graph to reduce CPU launch-skew effects. Each timing has six groups of five samples. They are not six alternating baseline/candidate pairs. A separate size-weighted surrogate graph uses the phase call counts and shared per-size buffers; it does not reproduce model compute, cache state or scheduling.

On GPU0/1, the first eight-configuration screen favors32 channels. The five-configuration combination screen favors:

```text
NCCL_MIN_NCHANNELS=32
NCCL_MAX_NCHANNELS=32
NCCL_P2P_READ_ENABLE=0
```

Its surrogate decision/world medians are5.220864/18.861055ms versus default6.405632/22.863873ms; a final default repeat gives6.409728/22.834688ms. Raw debug lines are retained privately; published transport summaries confirm32 coll channels and P2P/CUMEM write. All13 screen configurations completed cleanly. Four/eight channels regress; forcing Simple or LL128 alone offers no comparable improvement. These numbers are communication-only evidence, not a complete-model/video speedup or a new score.

The full candidate reuses the exact adopted222 source directory, cache paths, evaluator, guard, two-GPU policy and unchanged timer. Only the three NCCL environment values change. A complete11-round/23-video result, same-GPU default control and fresh official103 validation are required before changing official artifacts. Full outcomes, when completed, are under `results/aggressive-20261028/nccl-tuning222/full-*`; raw single-case JSON files intentionally list the four absent cases and are not five-case submissions.

Run the source-frozen communication screen after building the qualified private-tmpfs adapter:

```bash
python experiments/nccl-tuning222/run_sweep.py --scope combos --gpus 0,1 \
  --workdir /var/tmp/team07-review-tuning \
  --storage-config /var/tmp/team07-review-storage/storage-config.json
```

The checkout, host `.so` and caches requiring dynamic loading must be on executable mounts. This screen uses no checkpoint, model inputs or media and performs no website submission. It does not edit global NCCL configuration files. Per-process settings follow [NVIDIA's environment documentation](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2205/user-guide/docs/env.html); P2P read/write selection is implemented in [the2.20.5 transport source](https://github.com/NVIDIA/nccl/blob/v2.20.5-1/src/transport/p2p.cc).
