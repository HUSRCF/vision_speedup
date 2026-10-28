#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export UNIFOLM_SPLIT_GROUP_NORM=fp32
exec bash "${repo_root}/scripts/run_v2_compensated_combo_full.sh"
