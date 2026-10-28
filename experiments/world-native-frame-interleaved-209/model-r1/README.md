# Whole-world integration candidate: GPU evidence pending

These are the frozen sources staged for the first real-model incremental test.
CPU preflight passed 120 cases covering both ranks, three layouts, FiLM,
up/downsampling, full embedding rows and temporal boundaries. A separate meta
fixture binds all 21 actual ResBlock SGEMM weights by object identity despite an
extra owner-name prefix, and excludes the two outer Upsample weights.

The 21 target SGEMMs belong to 12 of the 22 ResBlocks. Only those 12 blocks are
remapped and gathered; the remaining ten contain no qualified SGEMM, so sharding
them would add communication without accelerating a tested native GEMM. The
CPU boundary fixtures use one selected block each; their inherited
`no_block_exclusion_rules` flag describes those fixtures, not the real-model
selection policy.

Both paired conditions keep the adopted world spatial/temporal parallel
algorithm. Only the candidate adds the 12 ResBlock frame partitions. Original
N16 shapes/strides, full16 embedding projections and full16 temporal convolution
remain. Original23 SGEMM and45 head-cache checks, including original Kahan CUBIN
hashes, run before new replacements. New kernels compile in a separate private
home cache; the original cache directory is restored before any old kernel is
used. No custom Triton cache-manager class is installed.

Each new kernel receives a fresh captured-pointer ABI witness with full-size,
properly typed Torch buffers. This witness graph is **not replayed** and claims
no numerical output from those uninitialized buffers. GPU numerical evidence
must come from the subsequent complete-model checks: six changing-input/master
states across 570 Linear,782 norm and192 Conv2d masters; all three outputs must
match signed-zero bits, and RNG/cache/baseline restoration must pass. Six paired
whole-world timings include all frame copies,12 gathers and version checks.

The first GPU task has now terminated before model gates: attempt1 was excluded
for a foreign CUDA context; attempt2 failed the rank-startup assertion requiring
exactly one non-parent NVML context. See [terminal evidence](../../../results/aggressive-20261028/world-native-frame-interleaved-209/model-r1-terminal).
No subsequent model checks or paired timings completed, so this is not evidence
for or against the candidate's model numerics or performance.
At this snapshot **none of those GPU model checks or timings has completed**.
There is no complete V2 or new quality result. Current official V2 remains
287.745486 seconds, five-case total836.651410 seconds.

To stage on a contest host, copy this entire directory into a new private scratch
directory and link `scripts/evaluation` to the canonical original209 evaluation
directory. Configure the same canonical source/runtime as the production
reproduction guide, then set:

```bash
export CANONICAL_ROOT=/path/to/original209
export PYTHONPATH="$PWD/scripts:$CANONICAL_ROOT/src:$CANONICAL_ROOT/scripts:$PWD"
export PREFIX_CACHE_DIR="$(mktemp -d "$HOME/.tmp/world-interleaved-cache-XXXXXX")"
export CUDA_VISIBLE_DEVICES=2,3 MONITORED_GPUS=2,3
export PYTHONHASHSEED=123 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export HF_HUB_CACHE=/share/huggingface/hub HUGGINGFACE_HUB_CACHE=/share/huggingface/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  "$HOME/.venv/bin/python" scripts/check_world_interleaved209_cpu.py
"$HOME/.venv/bin/python" retry_nccl_model_pair.py
```

Use a fresh directory because attempt logs are exclusive-created. The parent
waits for both GPUs to meet the strict idle criteria, obtains both reserve and
execution locks per GPU, rechecks availability, and only then creates CUDA
contexts. Foreign-context attempts are excluded and only this experiment's own
children are terminated. The original production CUBIN gates are preserved;
this candidate does not relax them for a cold cache or a different source path.

See [CPU evidence](../../../results/aggressive-20261028/world-native-frame-interleaved-209/model-r1-preflight).
