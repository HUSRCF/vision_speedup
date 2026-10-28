#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
guard_prefix="${PRIMARY_GEGLU_COMBO_GUARD_PREFIX:?Set an immutable guard prefix}"
case "${PRIMARY_GEGLU_COMBO:?Select ffn or pointwise}" in
    ffn)
        second_script=check_decision_primary_ffn.py
        second_marker=DECISION_PRIMARY_FFN_CHECK
        export UNIFOLM_DECISION_PRIMARY_FFN=all
        ;;
    pointwise)
        second_script=check_pointwise_nchw_runtime.py
        second_marker=POINTWISE_NCHW_RUNTIME_CHECK
        export UNIFOLM_POINTWISE_NCHW=1
        ;;
    *) echo 'Unknown primary GEGLU combination' >&2; exit 2 ;;
esac
export UNIFOLM_DECISION_PRIMARY_GEGLU=both
export UNIFOLM_DECISION_PRIMARY_GEGLU_SCOPE=all
export WAIT_BENCHMARK_GPU_CANDIDATES="${WAIT_BENCHMARK_GPU_CANDIDATES:-2 1}"
run_guard() {
    local script="$1" marker="$2" log="$3"
    export WAIT_BENCHMARK_SCRIPT="${script}"
    bash "${repo_root}/scripts/wait_temporal_conv2d_benchmark.sh" >"${log}" 2>&1
    cat "${log}"
    grep -q "^${marker} passed=true " "${log}"
    grep -q '^MICROBENCHMARK_CLEAN ' "${log}"
    if grep -q '^MICROBENCHMARK_CONTAMINATED ' "${log}"; then
        echo 'Combination full run blocked: shared guard GPU' >&2
        exit 1
    fi
}
run_guard check_decision_primary_geglu.py DECISION_PRIMARY_GEGLU_CHECK "${guard_prefix}-geglu.log"
run_guard "${second_script}" "${second_marker}" "${guard_prefix}-second.log"
export UNIFOLM_DECISION_HALF_GEGLU=1
export UNIFOLM_DECISION_VENDOR_FFN=chunked
export UNIFOLM_INITIAL_DECISION_LINEARS=1
export UNIFOLM_PREPACKED_GEGLU=1
export UNIFOLM_COMPENSATED_TEMPORAL_CONV=1
export UNIFOLM_IMPLICIT_TEMPORAL_CONV=1
exec bash "${repo_root}/scripts/run_v2_grouped_geglu_full.sh"
