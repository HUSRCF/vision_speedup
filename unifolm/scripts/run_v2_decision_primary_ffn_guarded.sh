#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
guard_log="${DECISION_PRIMARY_FFN_GUARD_LOG:?Set an immutable guard log path}"
export WAIT_BENCHMARK_SCRIPT=check_decision_primary_ffn.py
export WAIT_BENCHMARK_GPU_CANDIDATES="${WAIT_BENCHMARK_GPU_CANDIDATES:-1 3 0 2}"
bash "${repo_root}/scripts/wait_temporal_conv2d_benchmark.sh" >"${guard_log}" 2>&1
cat "${guard_log}"
grep -q '^DECISION_PRIMARY_FFN_CHECK passed=true ' "${guard_log}"
grep -q '^MICROBENCHMARK_CLEAN ' "${guard_log}"
if grep -q '^MICROBENCHMARK_CONTAMINATED ' "${guard_log}"; then
    echo 'Primary decision FFN full run blocked: shared guard GPU' >&2
    exit 1
fi
export UNIFOLM_DECISION_PRIMARY_FFN="${UNIFOLM_DECISION_PRIMARY_FFN:-all}"
export UNIFOLM_DECISION_HALF_GEGLU=1
export UNIFOLM_DECISION_VENDOR_FFN=chunked
export UNIFOLM_INITIAL_DECISION_LINEARS=1
export UNIFOLM_PREPACKED_GEGLU=1
export UNIFOLM_COMPENSATED_TEMPORAL_CONV=1
export UNIFOLM_IMPLICIT_TEMPORAL_CONV=1
exec bash "${repo_root}/scripts/run_v2_grouped_geglu_full.sh"
