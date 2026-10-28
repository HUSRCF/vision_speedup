#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export UNIFOLM_TEMPORAL_LAYOUT_DIRECT=1
export UNIFOLM_FUSED_RESIDUAL_NORM=1 UNIFOLM_FUSED_TEMPORAL_ATTENTION=0
exec bash "${repo_root}/scripts/run_v2_all_split_norms_full.sh"
