#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export UNIFOLM_TF32X3_LINEARS=1
export UNIFOLM_FUSED_ATTENTION_MIX=0
export UNIFOLM_BATCH_WORLD_DECODE_CHUNK=0
exec bash "${repo_root}/scripts/run_v2_layout_combo_full.sh"
