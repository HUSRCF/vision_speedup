#!/usr/bin/env bash
# Source from the repository before running any official benchmark entry point.
team07_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="${HOME}/.venv/bin:${PATH}"
export TMPDIR="${HOME}/.tmp"
export PYTHONPATH="${team07_repo}/src${PYTHONPATH:+:${PYTHONPATH}}"
export CKPT_PATH=/share/weights/UnifoLM-WMA-0-Dual/unifolm_wma_dual.ckpt
export INPUT_CACHE_DIR=/share/problems/unifolm-world-model-action/examples/compute_benchmark/preprocessed_inputs
export HF_HUB_CACHE=/share/huggingface/hub
export HUGGINGFACE_HUB_CACHE=/share/huggingface/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONHASHSEED=123
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
mkdir -p "${TMPDIR}"
unset team07_repo
