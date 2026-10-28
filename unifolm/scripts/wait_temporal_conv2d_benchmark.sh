#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${repo_root}/scripts/team07_env.sh"
deadline=$((SECONDS + ${WAIT_TIMEOUT_SECONDS:-21600}))
read -r -a diagnostic_gpus <<< "${WAIT_BENCHMARK_GPU_CANDIDATES:-0 1 2 3}"
for diagnostic_gpu in "${diagnostic_gpus[@]}"; do
    case "${diagnostic_gpu}" in
        0|1|2|3) ;;
        *) echo "Invalid diagnostic GPU: ${diagnostic_gpu}" >&2; exit 2 ;;
    esac
done

run_monitored_benchmark() {
    local diagnostic_pid status process_count final_count contaminated=0
    "$@" &
    diagnostic_pid=$!
    while kill -0 "${diagnostic_pid}" 2>/dev/null; do
        process_count="$(nvidia-smi -i "${gpu}" \
            --query-compute-apps=pid --format=csv,noheader,nounits | wc -l)"
        if (( process_count > 1 && contaminated == 0 )); then
            contaminated=1
            echo "MICROBENCHMARK_CONTAMINATED gpu=${gpu} process_count=${process_count} time=$(date -Is)"
        fi
        sleep 1
    done
    set +e
    wait "${diagnostic_pid}"
    status=$?
    set -e
    final_count="$(nvidia-smi -i "${gpu}" \
        --query-compute-apps=pid --format=csv,noheader,nounits | wc -l)"
    if (( final_count > 0 && contaminated == 0 )); then
        contaminated=1
        echo "MICROBENCHMARK_CONTAMINATED gpu=${gpu} final_process_count=${final_count} time=$(date -Is)"
    fi
    if (( contaminated == 0 && status == 0 )); then
        echo "MICROBENCHMARK_CLEAN gpu=${gpu} time=$(date -Is)"
    fi
    return "${status}"
}

# Optional predecessor PIDs are checked live; never restart an observed job.
for predecessor in "$@"; do
    while kill -0 "${predecessor}" 2>/dev/null; do
        (( SECONDS < deadline )) || exit 1
        sleep 5
    done
done

while (( SECONDS < deadline )); do
    for gpu in "${diagnostic_gpus[@]}"; do
        exec {diagnostic_lock}>"${TMPDIR}/team07-unifolm-gpu${gpu}.lock"
        if ! flock -n "${diagnostic_lock}"; then
            exec {diagnostic_lock}>&-
            continue
        fi
        used="$(nvidia-smi -i "${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits)"
        used="${used//[[:space:]]/}"
        if [[ "${used}" =~ ^[0-9]+$ ]] && (( used <= 16 )); then
            echo "TEMPORAL_CONV2D_DIAGNOSTIC_GPU=${gpu} time=$(date -Is)"
            export CUDA_VISIBLE_DEVICES="${gpu}"
            if [[ -n "${WAIT_BENCHMARK_SCRIPT:-}" ]]; then
                run_monitored_benchmark python "${repo_root}/scripts/${WAIT_BENCHMARK_SCRIPT}"
            else
                run_monitored_benchmark python "${repo_root}/scripts/benchmark_temporal_conv2d.py"
                run_monitored_benchmark python "${repo_root}/scripts/benchmark_temporal_conv2d.py" --cudnn-tf32
                run_monitored_benchmark python "${repo_root}/scripts/benchmark_temporal_conv2d.py" --cudnn-benchmark
            fi
            # This microbenchmark never contributes an official timing score.
            nvidia-smi -i "${gpu}" --query-compute-apps=pid,used_memory --format=csv,noheader
            echo "TEMPORAL_CONV2D_DIAGNOSTIC_FINISHED time=$(date -Is)"
            exit 0
        fi
        flock -u "${diagnostic_lock}"
        exec {diagnostic_lock}>&-
    done
    sleep 3
done
echo "Timed out waiting for temporal Conv2d diagnostic GPU" >&2
exit 1
