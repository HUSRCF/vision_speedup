#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export UNIFOLM_COMPENSATED_SPATIAL_ATTENTION=large
export UNIFOLM_COMPENSATED_GEGLU_BACKEND=tf32
export UNIFOLM_COMPENSATED_LINEAR_WEIGHT_CACHE=1
export UNIFOLM_WINOGRAD_CONV=0
exec bash "${repo_root}/scripts/run_v2_scaled_half_linears_full.sh"
