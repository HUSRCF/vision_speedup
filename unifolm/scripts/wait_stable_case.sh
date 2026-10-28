#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
deadline=$((SECONDS + ${WAIT_TIMEOUT_SECONDS:-7200}))
grace_seconds="${IDLE_GRACE_SECONDS:-8}"

while (( SECONDS < deadline )); do
    for gpu in 0 1 2 3; do
        read -r used utilization < <(
            nvidia-smi -i "${gpu}" \
                --query-gpu=memory.used,utilization.gpu \
                --format=csv,noheader,nounits | tr -d ','
        )
        if (( used > 16 || utilization > 1 )); then
            continue
        fi
        echo "IDLE_CANDIDATE_GPU=${gpu} grace_seconds=${grace_seconds} time=$(date -Is)"
        sleep "${grace_seconds}"
        read -r used utilization < <(
            nvidia-smi -i "${gpu}" \
                --query-gpu=memory.used,utilization.gpu \
                --format=csv,noheader,nounits | tr -d ','
        )
        if (( used > 16 || utilization > 1 )); then
            echo "IDLE_CANDIDATE_LOST gpu=${gpu} memory_used_mib=${used} utilization=${utilization}"
            continue
        fi
        echo "STABLE_GPU=${gpu} memory_used_mib=${used} utilization=${utilization} time=$(date -Is)"
        exec bash "${repo_root}/scripts/team07_run.sh" \
            "${WAIT_RUN_MODE:-case-stackbox}" "${gpu}"
    done
    sleep 3
done

echo "Timed out waiting for a stable idle GPU" >&2
exit 1
