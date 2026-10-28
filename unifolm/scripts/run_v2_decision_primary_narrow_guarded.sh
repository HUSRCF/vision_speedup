#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
guard_log="${DECISION_PRIMARY_NARROW_GUARD_LOG:?Set an immutable guard log path}"
export WAIT_BENCHMARK_SCRIPT=check_decision_primary_narrow.py
export WAIT_BENCHMARK_GPU_CANDIDATES="${WAIT_BENCHMARK_GPU_CANDIDATES:-1 2 0 3}"
bash "${repo_root}/scripts/wait_temporal_conv2d_benchmark.sh" "$@" >"${guard_log}" 2>&1
cat "${guard_log}"
grep -q '^DECISION_PRIMARY_NARROW_CHECK passed=true ' "${guard_log}"
grep -q '^MICROBENCHMARK_CLEAN ' "${guard_log}"
if grep -q '^MICROBENCHMARK_CONTAMINATED ' "${guard_log}"; then
    echo 'Decision C320 primary full run blocked: shared guard GPU' >&2
    exit 1
fi
export UNIFOLM_DECISION_PRIMARY_NARROW="${UNIFOLM_DECISION_PRIMARY_NARROW:-output}"
exec bash "${repo_root}/scripts/run_v2_grouped_geglu_full.sh"
