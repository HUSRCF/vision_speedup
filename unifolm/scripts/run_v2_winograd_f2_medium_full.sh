#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export UNIFOLM_WINOGRAD_F2=medium
exec bash "${repo_root}/scripts/run_v2_compensated_qkv_full.sh"
