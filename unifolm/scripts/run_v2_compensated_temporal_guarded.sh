#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
guard_log="${COMPENSATED_TEMPORAL_GUARD_LOG:?Set an immutable guard log path}"
export WAIT_BENCHMARK_SCRIPT=check_compensated_temporal_runtime.py
export WAIT_BENCHMARK_GPU_CANDIDATES="${WAIT_BENCHMARK_GPU_CANDIDATES:-3 1 0 2}"
bash "${repo_root}/scripts/wait_temporal_conv2d_benchmark.sh" >"${guard_log}" 2>&1
cat "${guard_log}"
grep -q '^COMPENSATED_TEMPORAL_RUNTIME_CHECK passed=true ' "${guard_log}"
grep -q '^MICROBENCHMARK_CLEAN ' "${guard_log}"
if grep -q '^MICROBENCHMARK_CONTAMINATED ' "${guard_log}"; then
    echo 'Compensated temporal full run blocked: CUDA guard shared a GPU' >&2
    exit 1
fi
export UNIFOLM_COMPENSATED_TEMPORAL_CONV=1
exec bash "${repo_root}/scripts/run_v2_grouped_geglu_full.sh"
