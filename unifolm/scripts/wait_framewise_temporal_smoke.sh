#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${repo_root}/scripts/team07_env.sh"
deadline=$((SECONDS + ${WAIT_TIMEOUT_SECONDS:-900}))

while (( SECONDS < deadline )); do
    for gpu in 0 1 2 3; do
        used="$(nvidia-smi -i "${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits)"
        used="${used//[[:space:]]/}"
        if [[ "${used}" =~ ^[0-9]+$ ]] && (( used <= 16 )); then
            echo "TRY_GPU=${gpu} memory_used_mib=${used} time=$(date -Is)"
            started=${SECONDS}
            set +e
            env \
                UNIFOLM_MATH_MODE=selective-tf32 \
                UNIFOLM_DENOISER_MODE=native \
                UNIFOLM_FULL_DENOISER_FP16="${WAIT_FULL_DENOISER_FP16:-0}" \
                UNIFOLM_CUDA_GRAPH=1 \
                UNIFOLM_GRAPH_SKIP_UNCHANGED_INPUTS="${WAIT_GRAPH_SKIP_UNCHANGED_INPUTS:-0}" \
                UNIFOLM_CONDITION_CACHE=1 \
                UNIFOLM_CUDNN_BENCHMARK=0 \
                UNIFOLM_CUDNN_BENCHMARK_LIMIT=10 \
                UNIFOLM_STAGE_CUDNN_BENCHMARK=simulation \
                UNIFOLM_STAGE_CUDNN_BENCHMARK_ALL_CASES=0 \
                UNIFOLM_STAGE_TF32=decision-cudnn \
                UNIFOLM_STAGE_TF32_ALL_CASES=0 \
                UNIFOLM_SIM_HEAD_TF32=1 \
                UNIFOLM_SIM_HEAD_CONV1D=native \
                UNIFOLM_SIM_BASE_CUDNN_TF32=none \
                UNIFOLM_FUSED_SELF_QKV=0 \
                UNIFOLM_CROSS_ATTENTION_IMPL="${WAIT_CROSS_ATTENTION_IMPL:-xformers}" \
                UNIFOLM_ATTENTION_DTYPE="${WAIT_ATTENTION_DTYPE:-fp32}" \
                UNIFOLM_ATTENTION_PROJECTION_DTYPE="${WAIT_ATTENTION_PROJECTION_DTYPE:-fp32}" \
                UNIFOLM_FFN_DTYPE="${WAIT_FFN_DTYPE:-fp32}" \
                UNIFOLM_FUSED_RESBLOCK_NORM="${WAIT_FUSED_RESBLOCK_NORM:-0}" \
                UNIFOLM_FRAMEWISE_TEMPORAL_NORM="${WAIT_FRAMEWISE_TEMPORAL_NORM:-1}" \
                UNIFOLM_FUSED_TEMPORAL_NORM="${WAIT_FUSED_TEMPORAL_NORM:-0}" \
                UNIFOLM_CHANNELS_LAST="${WAIT_CHANNELS_LAST:-0}" \
                UNIFOLM_OPERATOR_PROFILE=0 \
                UNIFOLM_SMOKE_N_ITER="${WAIT_SMOKE_N_ITER:-2}" \
                UNIFOLM_SMOKE_DATASET="${WAIT_SMOKE_DATASET:-unitree_z1_stackbox}" \
                UNIFOLM_TENSORBOARD_VIDEO=0 \
                UNIFOLM_SKIP_INIT=1 \
                UNIFOLM_CHECKPOINT_MMAP=1 \
                UNIFOLM_CHECKPOINT_ASSIGN=1 \
                UNIFOLM_META_INIT=1 \
                UNIFOLM_ASYNC_VIDEO=1 \
                UNIFOLM_STREAM_FULL_VIDEO=1 \
                UNIFOLM_DDIM_SCHEDULE_CACHE=1 \
                UNIFOLM_SKIP_UNUSED_ACTION_PROJECTION=1 \
                UNIFOLM_BATCH_DECISION_DECODE=selective \
                UNIFOLM_SKIP_UNUSED_HEADS=1 \
                bash "${repo_root}/scripts/team07_run.sh" \
                    "${WAIT_RUN_MODE:-smoke}" "${gpu}"
            status=$?
            set -e
            elapsed=$((SECONDS - started))
            if (( status == 0 )); then
                exit 0
            fi
            if (( elapsed >= 15 )); then
                echo "Smoke failed after launch: status=${status} elapsed=${elapsed}s" >&2
                exit "${status}"
            fi
        fi
    done
    sleep 3
done

echo "Timed out waiting for an idle GPU" >&2
exit 1
