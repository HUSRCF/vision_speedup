#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export UNIFOLM_COMPENSATED_GEGLU=narrow
export UNIFOLM_NVRTC_GEGLU=fmad
export UNIFOLM_NVRTC_GEGLU_VECTOR=1
export UNIFOLM_TF32X3_NARROW_FFN=0
export UNIFOLM_NATIVE_FP32_TEMPORAL_ATTENTION=0
exec bash "${repo_root}/scripts/run_v2_narrow_output_full.sh"
