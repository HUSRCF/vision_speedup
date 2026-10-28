#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export WAIT_RUN_MODE=case-v2
export UNIFOLM_MATH_MODE=fp32
export UNIFOLM_TIMING_QUALITY_ONLY=0
export UNIFOLM_TF32X3_LINEARS=triton
export UNIFOLM_FUSED_ATTENTION_MIX=1
export UNIFOLM_FFN_DTYPE=fp32
export UNIFOLM_BATCH_WORLD_DECODE_CHUNK=0
export UNIFOLM_TEMPORAL_CONV_DTYPE=fp32
export UNIFOLM_TEMPORAL_CONV_IMPL=fp32-conv2d
export UNIFOLM_ATTENTION_LAYOUT=bmhk
export UNIFOLM_BATCH_DECISION_DECODE=1
exec bash "${repo_root}/scripts/run_v2_temporal_conv2d_full.sh"
