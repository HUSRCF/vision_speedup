#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
guard_log="${DECISION_PRIMARY_GEGLU_GUARD_LOG:?Set an immutable guard log path}"
case "${UNIFOLM_DECISION_PRIMARY_GEGLU:-}" in
    value|both) ;;
    *) echo 'Set decision primary GEGLU mode=value or both' >&2; exit 2 ;;
esac
export WAIT_BENCHMARK_SCRIPT=check_decision_primary_geglu.py
export WAIT_BENCHMARK_GPU_CANDIDATES="${WAIT_BENCHMARK_GPU_CANDIDATES:-1 2 3 0}"
bash "${repo_root}/scripts/wait_temporal_conv2d_benchmark.sh" "$@" >"${guard_log}" 2>&1
cat "${guard_log}"
grep -q '^DECISION_PRIMARY_GEGLU_CHECK passed=true ' "${guard_log}"
grep -q '^MICROBENCHMARK_CLEAN ' "${guard_log}"
if grep -q '^MICROBENCHMARK_CONTAMINATED ' "${guard_log}"; then
    echo 'Decision primary GEGLU full run blocked: shared guard GPU' >&2
    exit 1
fi
export UNIFOLM_DECISION_HALF_GEGLU=1
export UNIFOLM_DECISION_VENDOR_FFN=chunked
export UNIFOLM_INITIAL_DECISION_LINEARS=1
export UNIFOLM_PREPACKED_GEGLU=1
export UNIFOLM_COMPENSATED_TEMPORAL_CONV=1
export UNIFOLM_IMPLICIT_TEMPORAL_CONV=1
exec bash "${repo_root}/scripts/run_v2_grouped_geglu_full.sh"
