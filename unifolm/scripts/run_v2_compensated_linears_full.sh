#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export UNIFOLM_TF32X3_LINEARS=triton
export UNIFOLM_FUSED_ATTENTION_MIX=0
export UNIFOLM_BATCH_WORLD_DECODE_CHUNK=0
export UNIFOLM_TEMPORAL_CONV_IMPL=fp32-conv2d
export UNIFOLM_ATTENTION_LAYOUT=bmhk
# Smaller decode batches keep the quality-only run within shared GPU memory.
if [[ "${UNIFOLM_TIMING_QUALITY_ONLY:-0}" == 1 ]]; then
    export UNIFOLM_BATCH_DECISION_DECODE=0
else
    export UNIFOLM_BATCH_DECISION_DECODE=1
fi
exec bash "${repo_root}/scripts/run_v2_temporal_conv2d_full.sh"
