#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
guard_log="${WINOGRAD_F2_MEDIUM_GUARD_LOG:?Set an immutable guard log path}"
export WAIT_BENCHMARK_SCRIPT=check_winograd_f2_medium.py
export WAIT_BENCHMARK_GPU_CANDIDATES="${WAIT_BENCHMARK_GPU_CANDIDATES:-1 3 0 2}"
bash "${repo_root}/scripts/wait_temporal_conv2d_benchmark.sh" >"${guard_log}" 2>&1
cat "${guard_log}"
grep -q '^WINOGRAD_F2_MEDIUM_CHECK passed=true ' "${guard_log}"
grep -q '^MICROBENCHMARK_CLEAN ' "${guard_log}"
if grep -q '^MICROBENCHMARK_CONTAMINATED ' "${guard_log}"; then
    echo 'F2 medium full run blocked: CUDA guard shared a GPU' >&2
    exit 1
fi
exec bash "${repo_root}/scripts/run_v2_winograd_f2_medium_full.sh"
