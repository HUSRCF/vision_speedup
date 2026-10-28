#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
temporal_waiter_pid="${TEMPORAL_WAITER_PID:-1194885}"

while kill -0 "${temporal_waiter_pid}" 2>/dev/null; do
    sleep 5
done

cd "${repo_root}"
exec env \
    V2_CANDIDATE_LABEL=v2-fused-self-qkv-smoke \
    V2_CANDIDATE_N_ITER=2 \
    V2_CANDIDATE_ATTENTION_DTYPE=fp32 \
    V2_CANDIDATE_PROJECTION_DTYPE=fp32 \
    V2_CANDIDATE_TEMPORAL_DTYPE=fp32 \
    V2_EXPERIMENT_MIN_COMPLETED_CASES=5 \
    CANDIDATE_RUN_MODE=smoke \
    CANDIDATE_DATASET=unitree_z1_dual_arm_stackbox_v2 \
    CANDIDATE_FULL_DENOISER_FP16=0 \
    CANDIDATE_LATE_CONTAMINATION_SECONDS=0 \
    CANDIDATE_FUSED_SELF_QKV=1 \
    bash scripts/launch_temporal_conv_after_cases.sh
