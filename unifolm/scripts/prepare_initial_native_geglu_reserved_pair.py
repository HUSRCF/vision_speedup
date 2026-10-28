"""Reuse the established clean full-run scheduler for a measured graph pair."""
from pathlib import Path
import sys
root=Path(sys.argv[1]);source=(root/'scripts/wait_clean_run.sh').read_text()
anchor='''        setsid env UNIFOLM_ALLOW_GPU_RESERVATION=1 \\
            bash "${repo_root}/scripts/team07_run.sh" \\
            "${WAIT_RUN_MODE:-full}" "${gpu}" &
'''
assert source.count(anchor)==1
replacement='''        pair_attempt_log="${HOME}/benchmark-runs/unifolm-launch/initial-native-pair-$$-${SECONDS}-gpu${gpu}.log"
        echo "PAIR_ATTEMPT_LOG=${pair_attempt_log}"
        setsid env UNIFOLM_ALLOW_GPU_RESERVATION=1 PAIR_ATTEMPT_LOG="${pair_attempt_log}" \\
            bash "${repo_root}/scripts/run_initial_native_geglu_reserved_pair.sh" \\
            pair "${gpu}" &
'''
source=source.replace(anchor,replacement)
anchor='''        set -e
        if (( late_contamination == 1 )); then
'''
assert source.count(anchor)==1
replacement='''        set -e
        sleep 1
        final_pair_count="$(gpu_process_count "${gpu_uuid}")"
        if (( final_pair_count > 0 )); then
            contaminated=1
            echo "PAIR_FINAL_FOREIGN_CONTEXT gpu=${gpu} process_count=${final_pair_count} time=$(date -Is)"
        fi
        if (( contaminated == 0 )); then
            if [[ -f "${pair_attempt_log}" ]]; then cat "${pair_attempt_log}"; fi
            if (( status == 0 )); then
                echo "MICROBENCHMARK_CLEAN gpu=${gpu} passive_reservation_mib=64 time=$(date -Is)"
            fi
        fi
        if (( late_contamination == 1 )); then
'''
source=source.replace(anchor,replacement)
# Earlier attempt outputs stay in their own files. The prerequisite gate sees
# exactly the final clean comparison directory and never a contaminated timing.
assert source.count('cat "${pair_attempt_log}"')==1
assert source.index('if (( contaminated == 0 )); then\n            if [[ -f')<source.index('if (( contaminated == 1 )); then\n            echo "CLEAN_RUN_RETRY')
(root/'scripts/wait_initial_native_geglu_reserved_pair.sh').write_text(source)
(root/'scripts/run_initial_native_geglu_reserved_pair.sh').write_text('''#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${repo_root}/scripts/team07_env.sh"
gpu="${2:?GPU required}"
case "${gpu}" in 0|1|2|3) ;; *) exit 2 ;; esac
exec 9>"${TMPDIR}/team07-unifolm-gpu${gpu}.lock"
flock -n 9 || exit 75
export CUDA_VISIBLE_DEVICES="${gpu}"
exec > "${PAIR_ATTEMPT_LOG:?Attempt log required}" 2>&1
python "${repo_root}/scripts/run_initial_bk32_native_geglu_pair.py"
grep -Fq 'INITIAL_BK32_NATIVE_GEGLU_BOTH_STAGES_CHECK passed=true' "${PAIR_ATTEMPT_LOG}"
''')
print('INITIAL_NATIVE_GEGLU_RESERVED_PAIR_PREPARED small_passive_startup_context=true '
      'shared_team_lock=true contaminated_attempts_isolated=true original_model_sources_unchanged=true',root,flush=True)
