#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
v2_waiter_pid="${V2_WAITER_PID:-1203917}"

while kill -0 "${v2_waiter_pid}" 2>/dev/null; do
    sleep 5
done

cd "${repo_root}"
exec env \
    V2_CANDIDATE_LABEL=g1-temporal-conv-fp16-strict \
    V2_CANDIDATE_ATTENTION_DTYPE=fp32 \
    V2_CANDIDATE_PROJECTION_DTYPE=fp32 \
    V2_CANDIDATE_TEMPORAL_DTYPE=fp16 \
    V2_EXPERIMENT_MIN_COMPLETED_CASES=5 \
    CANDIDATE_RUN_MODE=case-g1 \
    CANDIDATE_DATASET=unitree_g1_pack_camera \
    CANDIDATE_FULL_DENOISER_FP16=0 \
    CANDIDATE_LATE_CONTAMINATION_SECONDS=0 \
    CANDIDATE_FUSED_SELF_QKV=0 \
    bash scripts/launch_temporal_conv_after_cases.sh
