#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export UNIFOLM_COMPENSATED_LINEAR_BACKEND=scaled-fp16
exec bash "${repo_root}/scripts/run_v2_compensated_nvrtc_combo_full.sh"
