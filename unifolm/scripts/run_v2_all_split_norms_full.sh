#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export UNIFOLM_SPLIT_TRANSFORMER_NORM=1
exec bash "${repo_root}/scripts/run_v2_split_group_norm_full.sh"
