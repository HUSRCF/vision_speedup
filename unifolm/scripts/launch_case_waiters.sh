#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
launch_dir="${HOME}/benchmark-runs/unifolm-launch"
mkdir -p "${launch_dir}"
pid_file="${launch_dir}/case-waiters.pids"
: > "${pid_file}"

common_env=(
    WAIT_TIMEOUT_SECONDS=14400 IDLE_GRACE_SECONDS=0 CLEAN_POLL_SECONDS=2
    UNIFOLM_MATH_MODE=selective-tf32
    UNIFOLM_DENOISER_MODE=native
    UNIFOLM_FULL_DENOISER_FP16=non-v2
    UNIFOLM_CUDA_GRAPH=1
    UNIFOLM_GRAPH_SKIP_UNCHANGED_INPUTS=0
    UNIFOLM_CONDITION_CACHE=1
    UNIFOLM_CUDNN_BENCHMARK=0
    UNIFOLM_CUDNN_BENCHMARK_LIMIT=10
    UNIFOLM_STAGE_CUDNN_BENCHMARK=simulation
    UNIFOLM_STAGE_CUDNN_BENCHMARK_ALL_CASES=0
    UNIFOLM_STAGE_TF32=decision-cudnn
    UNIFOLM_STAGE_TF32_ALL_CASES=0
    UNIFOLM_SIM_HEAD_TF32=1
    UNIFOLM_SIM_HEAD_CONV1D=native
    UNIFOLM_SIM_BASE_CUDNN_TF32=none
    UNIFOLM_FUSED_SELF_QKV=0
    UNIFOLM_CROSS_ATTENTION_IMPL=xformers
    UNIFOLM_ATTENTION_DTYPE=fp16
    UNIFOLM_ATTENTION_PROJECTION_DTYPE=fp16
    UNIFOLM_FFN_DTYPE=fp32
    UNIFOLM_FUSED_RESBLOCK_NORM=0
    UNIFOLM_FRAMEWISE_TEMPORAL_NORM=0
    UNIFOLM_FUSED_TEMPORAL_NORM=0
    UNIFOLM_CHANNELS_LAST=0
    UNIFOLM_OPERATOR_PROFILE=0
    UNIFOLM_TENSORBOARD_VIDEO=0
    UNIFOLM_SKIP_INIT=1
    UNIFOLM_CHECKPOINT_MMAP=1
    UNIFOLM_CHECKPOINT_ASSIGN=1
    UNIFOLM_META_INIT=1
    UNIFOLM_ASYNC_VIDEO=1
    UNIFOLM_STREAM_FULL_VIDEO=1
    UNIFOLM_DDIM_SCHEDULE_CACHE=1
    UNIFOLM_SKIP_UNUSED_ACTION_PROJECTION=1
    UNIFOLM_BATCH_DECISION_DECODE=selective
    UNIFOLM_SKIP_UNUSED_HEADS=1
)

modes=(case-stackbox case-dual case-v2 case-pencils case-g1)
for mode in "${modes[@]}"; do
    log_path="${launch_dir}/wait-clean-final-${mode}-nonv2-fp16.log"
    : > "${log_path}"
    setsid nohup env "${common_env[@]}" WAIT_RUN_MODE="${mode}" \
        bash "${repo_root}/scripts/wait_clean_run.sh" \
        </dev/null >"${log_path}" 2>&1 &
    pid=$!
    printf '%s %s %s\n' "${pid}" "${mode}" "${log_path}" | tee -a "${pid_file}"
done
