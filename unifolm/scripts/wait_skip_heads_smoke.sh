#!/usr/bin/env bash
set -euo pipefail

repo=/home/team07/competition/unifolm-stage-benchmark
deadline=$((SECONDS + 14400))

while (( SECONDS < deadline )); do
    for gpu in 0 1 2 3; do
        read -r util memory < <(
            nvidia-smi -i "${gpu}" \
                --query-gpu=utilization.gpu,memory.used \
                --format=csv,noheader,nounits | tr -d ','
        )
        if (( util <= 5 && memory <= 16 )); then
            sleep 10
            read -r util memory < <(
                nvidia-smi -i "${gpu}" \
                    --query-gpu=utilization.gpu,memory.used \
                    --format=csv,noheader,nounits | tr -d ','
            )
            if (( util <= 5 && memory <= 16 )); then
                cd "${repo}"
                export UNIFOLM_MATH_MODE=fp32
                export UNIFOLM_DENOISER_MODE=native
                export UNIFOLM_CUDA_GRAPH=1
                export UNIFOLM_CONDITION_CACHE=1
                export UNIFOLM_CUDNN_BENCHMARK=0
                export UNIFOLM_CUDNN_BENCHMARK_LIMIT=10
                export UNIFOLM_STAGE_CUDNN_BENCHMARK=simulation
                export UNIFOLM_STAGE_TF32=decision-cudnn
                export UNIFOLM_TENSORBOARD_VIDEO=0
                export UNIFOLM_SKIP_INIT=1
                export UNIFOLM_CHECKPOINT_MMAP=1
                export UNIFOLM_CHECKPOINT_ASSIGN=1
                export UNIFOLM_META_INIT=1
                export UNIFOLM_ASYNC_VIDEO=1
                export UNIFOLM_STREAM_FULL_VIDEO=1
                export UNIFOLM_DDIM_SCHEDULE_CACHE=1
                export UNIFOLM_SKIP_UNUSED_ACTION_PROJECTION=1
                export UNIFOLM_BATCH_DECISION_DECODE=0
                export UNIFOLM_PARALLEL_HEADS=0
                export UNIFOLM_SKIP_UNUSED_HEADS=1
                echo "WATCHER_SELECTED_GPU=${gpu}"
                exec bash scripts/team07_run.sh smoke "${gpu}"
            fi
        fi
    done
    sleep 10
done

echo WATCHER_TIMEOUT
exit 124
