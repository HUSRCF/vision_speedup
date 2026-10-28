#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export UNIFOLM_COMPENSATED_SELF_QKV=1
export UNIFOLM_COMPENSATED_LINEAR_INPUT_PIPELINE=2
exec bash "${repo_root}/scripts/run_v2_prepacked_input_linears_full.sh"
