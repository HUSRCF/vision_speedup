#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export UNIFOLM_COMPENSATED_LINEAR_BLOCK_K=64
export UNIFOLM_COMPENSATED_LINEAR_GROUP_M=32
exec bash "${repo_root}/scripts/run_v2_compensated_attention_full.sh"
