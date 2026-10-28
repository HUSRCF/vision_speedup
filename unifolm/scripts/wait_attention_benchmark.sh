#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${repo_root}/scripts/team07_env.sh"
deadline=$((SECONDS + ${WAIT_TIMEOUT_SECONDS:-3600}))

while (( SECONDS < deadline )); do
    for gpu in 0 1 2 3; do
        used="$(nvidia-smi -i "${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits)"
        used="${used//[[:space:]]/}"
        if [[ "${used}" =~ ^[0-9]+$ ]] && (( used <= 16 )); then
            exec 9>"${TMPDIR}/team07-unifolm-gpu${gpu}.lock"
            if ! flock -n 9; then
                continue
            fi
            echo "CLAIMED_GPU=${gpu} memory_used_mib=${used} time=$(date -Is)"
            export CUDA_VISIBLE_DEVICES="${gpu}"
            export PYTHONPATH="${repo_root}/src"
            python "${repo_root}/scripts/benchmark_attention_backends.py" --iterations 30
            exit 0
        fi
    done
    sleep 3
done

echo "Timed out waiting for an idle GPU" >&2
exit 1
