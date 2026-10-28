#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
deadline=$((SECONDS + ${ASSEMBLY_WAIT_TIMEOUT_SECONDS:-18000}))
while (( SECONDS < deadline )); do
    set +e
    output="$(python3 "${repo_root}/scripts/assemble_clean_cases.py" 2>&1)"
    status=$?
    set -e
    if (( status == 0 )); then
        printf '%s\n' "${output}"
        exit 0
    fi
    printf 'ASSEMBLY_WAIT time=%s reason=%s\n' "$(date -Is)" \
        "$(printf '%s\n' "${output}" | tail -n 1)"
    sleep 30
done

echo "Timed out waiting for five clean case results" >&2
exit 1
