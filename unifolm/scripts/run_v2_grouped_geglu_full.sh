#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export UNIFOLM_COMPENSATED_GEGLU_GROUP_M=16
exec bash "${repo_root}/scripts/run_v2_winograd_f2_medium_full.sh"
