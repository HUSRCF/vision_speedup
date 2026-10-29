#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

seed="${SEED:-123}"
checkpoint="${CKPT_PATH:-downloads/UnifoLM-WMA-0-Dual/unifolm_wma_dual.ckpt}"
result_dir="${RESULT_DIR:-outputs/compute_benchmark/$(date -u +%Y%m%dT%H%M%SZ)-$$}"
profile_dir="${PROFILE_DIR:-${result_dir}/profiles}"
input_cache_dir="${INPUT_CACHE_DIR:-examples/compute_benchmark/preprocessed_inputs}"

result_json="${RESULT_JSON_PATH:-${result_dir}/result.json}"
log_path="${LOG_PATH:-${result_dir}/runtime.log}"
export RESULT_DIR="${result_dir}" PROFILE_DIR="${profile_dir}"
export RESULT_JSON_PATH="${result_json}" LOG_PATH="${log_path}"

if [[ "${1:-}" != --run-body ]]; then
    if [[ -e "${result_json}" || -e "${log_path}" ]]; then
        echo "Existing result/log; choose a new RESULT_DIR or output paths." >&2
        exit 1
    fi
    mkdir -p "${result_dir}" "${profile_dir}" "$(dirname "${log_path}")"
    # Capture both stderr and stdout while preserving the benchmark exit code.
    set +e
    bash "${repo_root}/scripts/run_crossstep_compute_benchmark.sh" --run-body 2>&1 | tee "${log_path}"
    run_codes=("${PIPESTATUS[@]}")
    if (( run_codes[0] != 0 )); then exit "${run_codes[0]}"; fi
    exit "${run_codes[1]}"
fi

started="$(python3 -c 'import time; print(time.monotonic())')"
result_writer="${repo_root}/scripts/benchmark_result.py"
if [[ -f /share/tools/unifolm-result.py ]]; then result_writer=/share/tools/unifolm-result.py; fi
finish_run() {
    local run_code=$?
    trap - EXIT
    set +e
    # Always validate finished cases, including early inference/launcher failures.
    local psnr_validator="${repo_root}/scripts/validate_psnr.py"
    if [[ -f /share/tools/unifolm-psnr.py ]]; then psnr_validator=/share/tools/unifolm-psnr.py; fi
    python3 "${psnr_validator}" --results "${result_dir}" \
        --reference /share/results/unifolm/fp32/reference/outputs \
        --report "${profile_dir}/psnr.json"
    local psnr_code=$?
    if (( run_code == 0 && psnr_code != 0 )); then run_code=$psnr_code; fi
    python3 "${result_writer}" --profiles "${profile_dir}" --result "${result_json}" \
        --exit-code "${run_code}" --started "${started}"
    local result_code=$?
    if (( run_code != 0 )); then exit "${run_code}"; fi
    exit "${result_code}"
}
trap finish_run EXIT
export PYTHONUNBUFFERED=1

# The default is the complete source trajectory for every case. Set
# BENCHMARK_N_ITER to an integer only for a local smoke test; it is not the
# competition workload.
benchmark_n_iter="${BENCHMARK_N_ITER:-original}"

export PYTHONHASHSEED="${seed}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [[ ! -f "${checkpoint}" ]]; then
    echo "Checkpoint not found: ${checkpoint}" >&2
    echo "Download it with: hf download unitreerobotics/UnifoLM-WMA-0-Dual unifolm_wma_dual.ckpt --local-dir downloads/UnifoLM-WMA-0-Dual" >&2
    exit 1
fi

mkdir -p "${profile_dir}"

datasets=(
    "unitree_z1_stackbox"
    "unitree_z1_dual_arm_stackbox"
    "unitree_z1_dual_arm_stackbox_v2"
    "unitree_z1_dual_arm_cleanup_pencils"
    "unitree_g1_pack_camera"
)
frame_strides=(4 4 4 4 6)
original_n_iters=(12 7 11 8 11)

# Optional complete-case rerun. Keep the official timing and quality tools;
# missing cases intentionally remain failed in this run's canonical result.
selected_case="${UNIFOLM_CASE:-}"
if [[ -n "${selected_case}" ]]; then
    case_found=false
    for dataset in "${datasets[@]}"; do
        if [[ "${dataset}" == "${selected_case}" ]]; then case_found=true; fi
    done
    if [[ "${case_found}" != true ]]; then
        echo "Unknown UNIFOLM_CASE=${selected_case}" >&2
        exit 2
    fi
    echo "SELECTED_CASE=${selected_case}; other cases are absent from this run"
fi

benchmark_rc=0
for i in "${!datasets[@]}"; do
    dataset="${datasets[$i]}"
    if [[ -n "${selected_case}" && "${dataset}" != "${selected_case}" ]]; then
        continue
    fi
    frame_stride="${frame_strides[$i]}"
    if [[ "${benchmark_n_iter}" == "original" ]]; then
        n_iter="${original_n_iters[$i]}"
    else
        n_iter="${benchmark_n_iter}"
    fi

    if python3 scripts/evaluation/world_model_interaction_crossstep.py \
        --seed "${seed}" \
        --ckpt_path "${checkpoint}" \
        --config configs/inference/world_model_interaction.yaml \
        --dataset "${dataset}" \
        --savedir "${result_dir}/${dataset}" \
        --profile_path "${profile_dir}/${dataset}.json" \
        --bs 1 --height 320 --width 512 \
        --unconditional_guidance_scale 1.0 \
        --ddim_steps 50 --ddim_eta 1.0 \
        --video_length 16 --frame_stride "${frame_stride}" \
        --exe_steps 16 --n_iter "${n_iter}" \
        --timestep_spacing uniform_trailing --guidance_rescale 0.7 \
        --perframe_ae \
        --input_cache_dir "${input_cache_dir}" --require_input_cache; then
        case_rc=0
    else
        case_rc=$?
        if (( benchmark_rc == 0 )); then benchmark_rc=$case_rc; fi
        echo "CASE_EXECUTION_FAILED case=${dataset} exit_code=${case_rc}"
    fi
    python3 - "${profile_dir}/execution.json" "${dataset}" "${case_rc}" <<'PYCASE'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);data=json.loads(p.read_text()) if p.exists() else {}
data[sys.argv[2]]={'exit_code':int(sys.argv[3])}
tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(p)
PYCASE
done

# EXIT finalization validates every case and preserves healthy evidence.
exit "$benchmark_rc"
