#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${repo_root}/scripts/team07_env.sh"
gpu="${2:?GPU required}"
case "${gpu}" in 0|1|2|3) ;; *) exit 2 ;; esac
exec 9>"${TMPDIR}/team07-unifolm-gpu${gpu}.lock"
flock -n 9 || exit 75
export CUDA_VISIBLE_DEVICES="${gpu}"
exec > "${PAIR_ATTEMPT_LOG:?Attempt log required}" 2>&1
python "${repo_root}/scripts/run_geglu_subtile32_pair.py"
grep -Fq 'GEGLU_SUBTILE32_BOTH_STAGES_CHECK passed=true' "${PAIR_ATTEMPT_LOG}"
