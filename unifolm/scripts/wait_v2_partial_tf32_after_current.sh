#!/usr/bin/env bash
set -euo pipefail
for pid in 1279596 1281908; do
    while kill -0 "${pid}" 2>/dev/null; do sleep 5; done
done
cd /home/team07/competition/unifolm-partial-tf32-benchmark
export WAIT_RUN_MODE=case-v2
export WAIT_TIMEOUT_SECONDS=21600
export RESERVATION_HOLD_SECONDS=120
export LATE_CONTAMINATION_SECONDS=1
export UNIFOLM_MATH_MODE=fp32
export UNIFOLM_STAGE_TF32=decision-cudnn
export UNIFOLM_STAGE_TF32_ALL_CASES=0
export UNIFOLM_STAGE_TF32_DDIM_PREFIX_STEPS=25
export UNIFOLM_STAGE_CUDNN_BENCHMARK=simulation
export UNIFOLM_CUDNN_BENCHMARK=0
export UNIFOLM_FULL_DENOISER_FP16=0
export UNIFOLM_ATTENTION_DTYPE=fp32
export UNIFOLM_ATTENTION_PROJECTION_DTYPE=fp32
export UNIFOLM_TEMPORAL_CONV_DTYPE=fp32
export UNIFOLM_BATCH_DECISION_DECODE=selective
export UNIFOLM_FUSED_SELF_QKV=0
exec bash scripts/wait_clean_run.sh
