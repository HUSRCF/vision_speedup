#!/usr/bin/env bash
set -euo pipefail

main_repo=/home/team07/competition/unifolm-world-model-action
experiment_repo=/home/team07/competition/unifolm-stage-benchmark
launch_dir=/home/team07/benchmark-runs/unifolm-launch
deadline=$((SECONDS + 14400))

gpu_idle() {
    local gpu=$1 util memory
    read -r util memory < <(
        nvidia-smi -i "${gpu}" \
            --query-gpu=utilization.gpu,memory.used \
            --format=csv,noheader,nounits | tr -d ','
    )
    (( util <= 5 && memory <= 16 ))
}

run_smoke() {
    local repo=$1 gpu=$2 skip=$3 log=$4
    cd "${repo}"
    UNIFOLM_MATH_MODE=fp32 \
    UNIFOLM_DENOISER_MODE=native \
    UNIFOLM_CUDA_GRAPH=1 \
    UNIFOLM_CONDITION_CACHE=1 \
    UNIFOLM_CUDNN_BENCHMARK=0 \
    UNIFOLM_CUDNN_BENCHMARK_LIMIT=10 \
    UNIFOLM_STAGE_CUDNN_BENCHMARK=simulation \
    UNIFOLM_STAGE_TF32=decision-cudnn \
    UNIFOLM_TENSORBOARD_VIDEO=0 \
    UNIFOLM_SKIP_INIT=1 \
    UNIFOLM_CHECKPOINT_MMAP=1 \
    UNIFOLM_CHECKPOINT_ASSIGN=1 \
    UNIFOLM_META_INIT=1 \
    UNIFOLM_ASYNC_VIDEO=1 \
    UNIFOLM_STREAM_FULL_VIDEO=1 \
    UNIFOLM_DDIM_SCHEDULE_CACHE=1 \
    UNIFOLM_SKIP_UNUSED_ACTION_PROJECTION=1 \
    UNIFOLM_BATCH_DECISION_DECODE=0 \
    UNIFOLM_PARALLEL_HEADS=0 \
    UNIFOLM_SKIP_UNUSED_HEADS="${skip}" \
        bash scripts/team07_run.sh smoke "${gpu}" >"${log}" 2>&1
}

while (( SECONDS < deadline )); do
    for gpu in 0 1 2 3; do
        gpu_idle "${gpu}" || continue
        sleep 15
        gpu_idle "${gpu}" || continue
        sleep 15
        gpu_idle "${gpu}" || continue
        stamp=$(date -u +%Y%m%dT%H%M%SZ)
        echo "PAIR_SELECTED_GPU=${gpu} STAMP=${stamp}"
        baseline_log="${launch_dir}/paired-${stamp}-baseline-gpu${gpu}.log"
        experiment_log="${launch_dir}/paired-${stamp}-skip-heads-gpu${gpu}.log"
        if ! run_smoke "${main_repo}" "${gpu}" 0 "${baseline_log}"; then
            echo "PAIR_BASELINE_REJECTED gpu=${gpu}"
            continue 2
        fi
        sleep 10
        if ! gpu_idle "${gpu}"; then
            echo "PAIR_ABORT_AFTER_BASELINE gpu=${gpu}"
            continue 2
        fi
        if ! run_smoke "${experiment_repo}" "${gpu}" 1 "${experiment_log}"; then
            echo "PAIR_EXPERIMENT_REJECTED gpu=${gpu}"
            continue 2
        fi
        echo "PAIR_COMPLETE baseline=${baseline_log} experiment=${experiment_log}"
        exit 0
    done
    sleep 10
done

echo PAIR_TIMEOUT
exit 124
