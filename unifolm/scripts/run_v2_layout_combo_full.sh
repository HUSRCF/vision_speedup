#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export UNIFOLM_TEMPORAL_CONV_IMPL=fp32-conv2d
export UNIFOLM_ATTENTION_LAYOUT=bmhk
export UNIFOLM_BATCH_DECISION_DECODE=1
exec bash "${repo_root}/scripts/run_v2_temporal_conv2d_full.sh"
