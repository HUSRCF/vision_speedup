#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export UNIFOLM_PREPACKED_GEGLU=1
export UNIFOLM_IMPLICIT_TEMPORAL_CONV=1
exec bash "${repo_root}/scripts/run_v2_initial_decision_guarded.sh"
