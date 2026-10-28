#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${repo_root}/scripts/team07_env.sh"
deadline=$((SECONDS + ${WAIT_TIMEOUT_SECONDS:-900}))

run_on_gpu() {
    local gpu="$1"
    local used
    exec 9>"${TMPDIR:-/tmp}/team07-unifolm-gpu${gpu}.lock"
    flock -n 9 || return 1
    used="$(nvidia-smi -i "${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits)"
    used="${used//[[:space:]]/}"
    [[ "${used}" =~ ^[0-9]+$ ]] || return 1
    (( used <= 16 )) || return 1
    echo "CLAIMED_GPU=${gpu} memory_used_mib=${used} time=$(date -Is)"
    cd "${repo_root}"
    CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${repo_root}/src" \
        python scripts/benchmark_fused_group_norm.py --iterations 50
}

while (( SECONDS < deadline )); do
    for gpu in 0 1 2 3; do
        if (run_on_gpu "${gpu}"); then
            exit 0
        fi
    done
    sleep 3
done

echo "Timed out waiting for an idle GPU" >&2
exit 1
