#!/usr/bin/env bash
set -euo pipefail
export UNIFOLM_DECISION_INLINE_ZERO_LOW_STAGES3=1
export UNIFOLM_DECISION_ZERO_LOW_STAGES3=1
export UNIFOLM_WORLD_ZERO_LOW_STAGES3=1
export UNIFOLM_CUDA_GRAPH=1
export UNIFOLM_NATIVE_GRAPH_WEIGHT_CACHE=1
export UNIFOLM_COMPENSATED_LINEAR_INPUT_PACK=wide
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${repo_root}/scripts/team07_env.sh"
gpu="${2:?GPU required}"
case "${gpu}" in 0|1|2|3) ;; *) exit 2 ;; esac
exec 9>"${TMPDIR}/team07-unifolm-gpu${gpu}.lock"
flock -n 9 || exit 75
export CUDA_VISIBLE_DEVICES="${gpu}"
exec > "${PAIR_ATTEMPT_LOG:?Attempt log required}" 2>&1
python "${repo_root}/scripts/check_decision_inline_zero_low_stages_production.py"
grep -Fq 'DECISION_INLINE_ZERO_LOW_STAGES_PRODUCTION_CHECK passed=true' "${PAIR_ATTEMPT_LOG}"
