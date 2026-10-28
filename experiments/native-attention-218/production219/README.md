# Accepted native attention219 production entry

This is the current adopted default. This frozen implementation
adds `nccl_native_attention219_mode.py` and changes only routing/count checks in
`nccl_parallel_runtime209.py`. The original209 files and four additive evaluator
hooks are unchanged. Removing those hooks recovers the original timed evaluator.
The two computational wrapper functions have identical ASTs to the qualified
world/decision model experiment. No new JIT kernel or collective is introduced.

Run `python materialize.py --output /fresh/private/path` from this directory.
Use the adopted218 flags with
`UNIFOLM_NCCL_PARALLEL=both-spatial-temporal-resblock-grid-native-attention`.
The model and two-GPU environment requirements are the same as adopted218;
the default `reproduce/run_v2_parallel.py` now selects this implementation.

CPU lifecycle checks and the24 mask/frame routing cases pass without CUDA.
The full V2 baseline and candidate were tested on GPU1/2 with
matching private RAM filesystem outputs, because the home filesystem has only
about15MiB free. The native graph extension remains linked to the original
executable home file because RAM filesystem mounts are `noexec`. Warm compiler
caches are retained. Python bytecode writes are disabled in both conditions.
Neither source relocation nor output storage changes the original process timer:
all model initialization, graph construction, communication setup, computation,
video writing, helper completion and teardown stay inside it. The official PSNR
and result writer remain unmodified. Storage differences from historical runs
must be considered when comparing against the old287.745486s score.

The guard takes reservation and execution locks, rechecks strict idleness, owns
both rank contexts and excludes entire attempts if another context appears.
GPU temperatures, clocks, memory and utilization are sampled throughout.

The completed218 baseline was286.724677s; candidate219 was280.850431s,
a paired whole-process saving of5.874246s. Both were clean and complete, and
all23 videos equal the historical adopted218 baseline and the other rank.
PSNR remains29.675324786dB. Fresh official49round/103video validation passed;
five-case sum829.756355s. This is an assembly of independent complete cases.
It is not a consecutive five-case wall-clock run or a website submission.
See [whole-process evidence](../../../results/aggressive-20261028/native-attention-218/full-candidate-r2)
and [review response](../../../docs/REVIEW_RESPONSE_20261029.md).
