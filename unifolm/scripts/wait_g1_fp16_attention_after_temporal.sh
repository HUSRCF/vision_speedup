#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
g1_temporal_waiter_pid="${G1_TEMPORAL_WAITER_PID:-1220888}"

while kill -0 "${g1_temporal_waiter_pid}" 2>/dev/null; do
    sleep 5
done

cd "${repo_root}"
exec env \
    V2_CANDIDATE_LABEL=g1-fp16-attention-projection-full \
    V2_CANDIDATE_ATTENTION_DTYPE=fp16 \
    V2_CANDIDATE_PROJECTION_DTYPE=fp16 \
    V2_CANDIDATE_TEMPORAL_DTYPE=fp32 \
    V2_EXPERIMENT_MIN_COMPLETED_CASES=5 \
    CANDIDATE_RUN_MODE=case-g1 \
    CANDIDATE_DATASET=unitree_g1_pack_camera \
    CANDIDATE_FULL_DENOISER_FP16=0 \
    CANDIDATE_LATE_CONTAMINATION_SECONDS=1 \
    CANDIDATE_FUSED_SELF_QKV=0 \
    bash scripts/launch_temporal_conv_after_cases.sh
