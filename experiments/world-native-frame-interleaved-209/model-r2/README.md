# Whole-world integration r2: GPU evidence pending

Same candidate kernels,21 SGEMM owners in12 ResBlocks, original209 sources and
all numerical/cache/old-CUBIN gates as model-r1. Both baseline and candidate
retain adopted world spatial/temporal parallelism. New orchestration records
unresolved rank-context ownership and excludes/retries that entire attempt,
including errors written only to ranks.log. Strict idle thresholds and exactly
one non-parent rank context remain. Model numerical failures still stop.
The [four CPU startup fixtures](../startup-guard-r2) passed.

The original120 CPU block-boundary checks and actual owner-identity binding
passed again. Nine new kernels were precompiled forSM80 in a private cache
with CUDA_VISIBLE_DEVICES empty, no CUDA initialization/load/launch, unchanged
compiler stages and an artifact-reader return object in the CPU-only helper.
The helper is never imported by GPU workers; no old production cache is edited.
Runtime captured-pointer witnesses and all complete-model gates remain required.

The GPU2/3 task terminated while loading a new kernel: CompiledKernel read
binary bytes as text and raised UnicodeDecodeError at byte18. No complete
GPU-model timing/numerical or full V2/PSNR result was produced. The installed
Triton writer's temporary CUBIN filenames match the reader's broad glob; the
[CPU reproduction](../cache-temp-race-cpu-r1) demonstrates this race and its
absence with per-rank directories. The observed GPU failure is consistent with
that cause; r3 isolates the new cache by rank to verify on actual hardware.
The CPU precompile report is static compile evidence, not proof that every
runtime specialization was cache-hit. Official V2 remains287.745486 seconds,
five-case total836.651410. Follow model-r1's staging instructions from a fresh
private directory. The optional CPU helper is described in offline-compile-r2.
[Static CPU evidence](../../../results/aggressive-20261028/world-native-frame-interleaved-209/model-r2-preflight).
