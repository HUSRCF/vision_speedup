#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export UNIFOLM_TF32X3_NARROW_OUTPUT=1
export UNIFOLM_COMPENSATED_SPATIAL_CONV=0
export UNIFOLM_FUSED_TEMPORAL_ATTENTION_FP32=1
export UNIFOLM_FUSED_TEMPORAL_ATTENTION=0
export UNIFOLM_FUSED_RESIDUAL_NORM="${UNIFOLM_FUSED_RESIDUAL_NORM:-1}"
export UNIFOLM_TEMPORAL_LAYOUT_DIRECT=1
exec bash "${repo_root}/scripts/run_v2_all_split_norms_full.sh"
