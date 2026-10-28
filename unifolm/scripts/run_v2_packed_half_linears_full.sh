#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export UNIFOLM_COMPENSATED_LINEAR_WEIGHT_CACHE=1
exec bash "${repo_root}/scripts/run_v2_scaled_half_linears_full.sh"
