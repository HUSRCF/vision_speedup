#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export UNIFOLM_COMPENSATED_LINEAR_INPUT_PACK=wide
exec bash "${repo_root}/scripts/run_v2_half_linear_reduction64_full.sh"
