#!/usr/bin/env bash
set -euo pipefail

formal_root="${HOME}/competition/unifolm-stage-benchmark"
experiment_root="${HOME}/competition/unifolm-temporal-conv-benchmark"
launch_dir="${HOME}/benchmark-runs/unifolm-launch"
deadline=$((SECONDS + ${EXPERIMENT_WAIT_TIMEOUT_SECONDS:-18000}))
candidate_label="${V2_CANDIDATE_LABEL:-temporal-conv-fp16}"
candidate_attention_dtype="${V2_CANDIDATE_ATTENTION_DTYPE:-fp32}"
candidate_projection_dtype="${V2_CANDIDATE_PROJECTION_DTYPE:-fp32}"
candidate_temporal_dtype="${V2_CANDIDATE_TEMPORAL_DTYPE:-v2-fp16}"
candidate_iterations="${V2_CANDIDATE_N_ITER:-6}"
minimum_completed_cases="${V2_EXPERIMENT_MIN_COMPLETED_CASES:-3}"
candidate_run_mode="${CANDIDATE_RUN_MODE:-smoke}"
candidate_dataset="${CANDIDATE_DATASET:-unitree_z1_dual_arm_stackbox_v2}"
candidate_full_fp16="${CANDIDATE_FULL_DENOISER_FP16:-non-v2}"
candidate_late_contamination="${CANDIDATE_LATE_CONTAMINATION_SECONDS:-0}"
candidate_fused_self_qkv="${CANDIDATE_FUSED_SELF_QKV:-0}"
candidate_reservation_hold="${CANDIDATE_RESERVATION_HOLD_SECONDS:-120}"
candidate_batch_decode="${CANDIDATE_BATCH_DECISION_DECODE:-selective}"

while (( SECONDS < deadline )); do
    completed="$(${HOME}/.venv/bin/python - "${formal_root}" <<'PY'
import importlib.util
import sys
from pathlib import Path

root = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location(
    "assembler", root / "scripts" / "assemble_clean_cases.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
runs_root = Path.home() / "benchmark-runs" / "unifolm"
count = 0
for mode, dataset in module.CASES.items():
    if any(module.valid_candidate(path, dataset) for path in
           runs_root.glob(f"*-{mode}-*")):
        count += 1
print(count)
PY
)"
    if (( completed >= minimum_completed_cases )); then break; fi
    echo "V2_EXPERIMENT_WAIT label=${candidate_label} "\
         "completed_cases=${completed} time=$(date -Is)"
    sleep 30
done

if (( SECONDS >= deadline )); then
    echo "Timed out before ${minimum_completed_cases} formal cases completed" >&2
    exit 1
fi

echo "V2_EXPERIMENT_ELIGIBLE label=${candidate_label} "\
     "completed_cases=${completed} time=$(date -Is)"
cd "${experiment_root}"
exec env \
    WAIT_TIMEOUT_SECONDS=10800 IDLE_GRACE_SECONDS=0 CLEAN_POLL_SECONDS=2 \
    RESERVATION_HOLD_SECONDS="${candidate_reservation_hold}" \
    LATE_CONTAMINATION_SECONDS="${candidate_late_contamination}" \
    WAIT_RUN_MODE="${candidate_run_mode}" \
    UNIFOLM_SMOKE_N_ITER="${candidate_iterations}" \
    UNIFOLM_SMOKE_DATASET="${candidate_dataset}" \
    UNIFOLM_MATH_MODE=selective-tf32 \
    UNIFOLM_DENOISER_MODE=native \
    UNIFOLM_FULL_DENOISER_FP16="${candidate_full_fp16}" \
    UNIFOLM_CUDA_GRAPH=1 \
    UNIFOLM_GRAPH_SKIP_UNCHANGED_INPUTS=0 \
    UNIFOLM_CONDITION_CACHE=1 \
    UNIFOLM_CUDNN_BENCHMARK=0 \
    UNIFOLM_CUDNN_BENCHMARK_LIMIT=10 \
    UNIFOLM_STAGE_CUDNN_BENCHMARK=simulation \
    UNIFOLM_STAGE_CUDNN_BENCHMARK_ALL_CASES=0 \
    UNIFOLM_STAGE_TF32=decision-cudnn \
    UNIFOLM_STAGE_TF32_ALL_CASES=0 \
    UNIFOLM_SIM_HEAD_TF32=1 \
    UNIFOLM_SIM_HEAD_CONV1D=native \
    UNIFOLM_SIM_BASE_CUDNN_TF32=none \
    UNIFOLM_FUSED_SELF_QKV="${candidate_fused_self_qkv}" \
    UNIFOLM_CROSS_ATTENTION_IMPL=xformers \
    UNIFOLM_ATTENTION_DTYPE="${candidate_attention_dtype}" \
    UNIFOLM_ATTENTION_PROJECTION_DTYPE="${candidate_projection_dtype}" \
    UNIFOLM_FFN_DTYPE=fp32 \
    UNIFOLM_TEMPORAL_CONV_DTYPE="${candidate_temporal_dtype}" \
    UNIFOLM_FUSED_RESBLOCK_NORM=0 \
    UNIFOLM_FRAMEWISE_TEMPORAL_NORM=0 \
    UNIFOLM_FUSED_TEMPORAL_NORM=0 \
    UNIFOLM_CHANNELS_LAST=0 \
    UNIFOLM_OPERATOR_PROFILE=0 \
    UNIFOLM_TENSORBOARD_VIDEO=0 \
    UNIFOLM_SKIP_INIT=1 \
    UNIFOLM_CHECKPOINT_MMAP=1 \
    UNIFOLM_CHECKPOINT_ASSIGN=1 \
    UNIFOLM_META_INIT=1 \
    UNIFOLM_ASYNC_VIDEO=1 \
    UNIFOLM_STREAM_FULL_VIDEO=1 \
    UNIFOLM_DDIM_SCHEDULE_CACHE=1 \
    UNIFOLM_SKIP_UNUSED_ACTION_PROJECTION=1 \
    UNIFOLM_BATCH_DECISION_DECODE="${candidate_batch_decode}" \
    UNIFOLM_SKIP_UNUSED_HEADS=1 \
    bash scripts/wait_clean_run.sh
