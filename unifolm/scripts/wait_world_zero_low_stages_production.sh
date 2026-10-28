#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
deadline=$((SECONDS + ${WAIT_TIMEOUT_SECONDS:-14400}))
grace_seconds="${IDLE_GRACE_SECONDS:-8}"
poll_seconds="${CLEAN_POLL_SECONDS:-2}"
reservation_hold_seconds="${RESERVATION_HOLD_SECONDS:-30}"
late_contamination_seconds="${LATE_CONTAMINATION_SECONDS:-0}"
reservation_python="${HOME}/.venv/bin/python"
read -r -a gpu_candidates <<< "${WAIT_GPU_CANDIDATES:-0 1 2 3}"
for candidate in "${gpu_candidates[@]}"; do
    case "${candidate}" in
        0|1|2|3) ;;
        *) echo "Invalid WAIT_GPU_CANDIDATES GPU: ${candidate}" >&2; exit 2 ;;
    esac
done
mkdir -p "${HOME}/.tmp"

declare -a reservation_pids reservation_triggers reservation_readies reservation_warmed

release_candidate_locks() {
    if [[ -n "${candidate_main_lock_fd:-}" ]]; then
        flock -u "${candidate_main_lock_fd}"
        exec {candidate_main_lock_fd}>&-
        unset candidate_main_lock_fd
    fi
    if [[ -n "${reservation_lock_fd:-}" ]]; then
        flock -u "${reservation_lock_fd}"
        exec {reservation_lock_fd}>&-
        unset reservation_lock_fd
    fi
}

start_reservation_daemon() {
    local gpu=$1
    local prefix="${HOME}/.tmp/unifolm-reservation-${gpu}-$$"
    reservation_triggers[$gpu]="${prefix}.trigger"
    reservation_readies[$gpu]="${prefix}.ready"
    reservation_warmed[$gpu]="${prefix}.warmed"
    rm -f "${reservation_triggers[$gpu]}" "${reservation_readies[$gpu]}" \
        "${reservation_warmed[$gpu]}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${reservation_python}" \
        "${repo_root}/scripts/gpu_reservation.py" \
        --trigger "${reservation_triggers[$gpu]}" \
        --ready "${reservation_readies[$gpu]}" \
        --warmed "${reservation_warmed[$gpu]}" &
    reservation_pids[$gpu]=$!
}

stop_reservation_daemon() {
    local gpu=$1
    local pid="${reservation_pids[$gpu]:-}"
    if [[ -n "${pid}" ]]; then
        kill -TERM "${pid}" 2>/dev/null || true
        wait "${pid}" 2>/dev/null || true
    fi
    rm -f "${reservation_triggers[$gpu]:-}" "${reservation_readies[$gpu]:-}" \
        "${reservation_warmed[$gpu]:-}"
    reservation_pids[$gpu]=""
}

cleanup_reservation_daemons() {
    local gpu
    release_candidate_locks
    for gpu in 0 1 2 3; do stop_reservation_daemon "${gpu}"; done
}
trap cleanup_reservation_daemons EXIT

for gpu in 0 1 2 3; do start_reservation_daemon "${gpu}"; done
for _ in {1..150}; do
    all_warmed=1
    for gpu in 0 1 2 3; do
        [[ -s "${reservation_warmed[$gpu]}" ]] || all_warmed=0
    done
    (( all_warmed == 1 )) && break
    sleep 0.1
done

gpu_process_count() {
    local uuid=$1
    nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader,nounits \
        2>/dev/null | awk -v uuid="${uuid}" '$1 == uuid {count++} END {print count+0}'
}

mark_contaminated_run() {
    local benchmark_pid="$1"
    local run_dir
    for run_dir in "${HOME}/benchmark-runs/unifolm/"*"-${benchmark_pid}"; do
        [[ -d "${run_dir}" ]] || continue
        printf 'Shared GPU detected: gpu=%s benchmark_pid=%s time=%s\n' \
            "${gpu}" "${benchmark_pid}" "$(date -Is)" \
            > "${run_dir}/timing-contaminated.txt"
    done
}

while (( SECONDS < deadline )); do
    for gpu in "${gpu_candidates[@]}"; do
        read -r used utilization < <(
            nvidia-smi -i "${gpu}" \
                --query-gpu=memory.used,utilization.gpu \
                --format=csv,noheader,nounits | tr -d ','
        )
        if (( used > 16 || utilization > 1 )); then
            continue
        fi
        exec {reservation_lock_fd}>"${HOME}/.tmp/team07-unifolm-reserve-gpu${gpu}.lock"
        if ! flock -n "${reservation_lock_fd}"; then
            release_candidate_locks
            continue
        fi
        # A microbenchmark can hold the main lock before it creates a CUDA
        # context. Respect that lock before triggering any GPU allocation.
        exec {candidate_main_lock_fd}>"${HOME}/.tmp/team07-unifolm-gpu${gpu}.lock"
        if ! flock -n "${candidate_main_lock_fd}"; then
            echo "TEAM07_MAIN_LOCK_BUSY gpu=${gpu} time=$(date -Is)"
            release_candidate_locks
            continue
        fi
        echo "IDLE_CANDIDATE_GPU=${gpu} grace_seconds=${grace_seconds} time=$(date -Is)"
        sleep "${grace_seconds}"
        read -r used utilization < <(
            nvidia-smi -i "${gpu}" \
                --query-gpu=memory.used,utilization.gpu \
                --format=csv,noheader,nounits | tr -d ','
        )
        if (( used > 16 || utilization > 1 )); then
            echo "IDLE_CANDIDATE_LOST gpu=${gpu} memory_used_mib=${used} utilization=${utilization}"
            release_candidate_locks
            continue
        fi

        gpu_uuid="$(nvidia-smi -i "${gpu}" --query-gpu=uuid --format=csv,noheader,nounits | tr -d '[:space:]')"
        reservation_ready="${reservation_readies[$gpu]}"
        reservation_pid="${reservation_pids[$gpu]}"
        if [[ ! -s "${reservation_warmed[$gpu]}" ]] || \
                ! kill -0 "${reservation_pid}" 2>/dev/null; then
            echo "GPU_RESERVATION_NOT_WARMED gpu=${gpu} time=$(date -Is)"
            release_candidate_locks
            stop_reservation_daemon "${gpu}"
            start_reservation_daemon "${gpu}"
            continue
        fi
        : > "${reservation_triggers[$gpu]}"
        for _ in {1..50}; do
            [[ -s "${reservation_ready}" ]] && break
            kill -0 "${reservation_pid}" 2>/dev/null || break
            sleep 0.1
        done
        if [[ ! -s "${reservation_ready}" ]] || \
                ! kill -0 "${reservation_pid}" 2>/dev/null; then
            echo "GPU_RESERVATION_FAILED gpu=${gpu} time=$(date -Is)"
            release_candidate_locks
            stop_reservation_daemon "${gpu}"
            start_reservation_daemon "${gpu}"
            continue
        fi
        process_count="$(gpu_process_count "${gpu_uuid}")"
        if (( process_count != 1 )); then
            echo "GPU_RESERVATION_RACED gpu=${gpu} process_count=${process_count} time=$(date -Is)"
            release_candidate_locks
            stop_reservation_daemon "${gpu}"
            start_reservation_daemon "${gpu}"
            continue
        fi
        echo "GPU_RESERVED gpu=${gpu} pid=${reservation_pid} uuid=${gpu_uuid} time=$(date -Is)"
        # ready is written after cudaMalloc/synchronize; NVML now sees the
        # reservation, so diagnostic waiters cannot claim this idle window.
        release_candidate_locks

        echo "CLEAN_RUN_GPU=${gpu} uuid=${gpu_uuid} time=$(date -Is)"
        pair_attempt_log="${HOME}/benchmark-runs/unifolm-launch/decision-primary-hidden-parts-pair-$$-${SECONDS}-gpu${gpu}.log"
        echo "PAIR_ATTEMPT_LOG=${pair_attempt_log}"
        setsid env UNIFOLM_ALLOW_GPU_RESERVATION=1 PAIR_ATTEMPT_LOG="${pair_attempt_log}" \
            bash "${repo_root}/scripts/run_world_zero_low_stages_production.sh" \
            pair "${gpu}" &
        run_pid=$!
        run_started_at=$SECONDS
        contaminated=0
        late_contamination=0
        reservation_release_at=$((SECONDS + reservation_hold_seconds))
        while kill -0 "${run_pid}" 2>/dev/null; do
            reservation_alive=0
            if kill -0 "${reservation_pid}" 2>/dev/null; then
                reservation_alive=1
            fi
            if (( reservation_alive == 1 && SECONDS >= reservation_release_at )); then
                kill -TERM "${reservation_pid}" 2>/dev/null || true
                wait "${reservation_pid}" 2>/dev/null || true
                reservation_pids[$gpu]=""
                reservation_alive=0
                rm -f "${reservation_ready}"
                echo "GPU_RESERVATION_RELEASED gpu=${gpu} time=$(date -Is)"
                sleep 2
            fi
            process_count="$(gpu_process_count "${gpu_uuid}")"
            expected_processes=1
            if (( reservation_alive == 1 )); then expected_processes=2; fi
            if (( process_count > expected_processes )); then
                mark_contaminated_run "${run_pid}"
                run_elapsed=$((SECONDS - run_started_at))
                if (( late_contamination == 0 && late_contamination_seconds > 0 &&
                      run_elapsed >= late_contamination_seconds )); then
                    late_contamination=1
                    echo "GPU_LATE_CONTAMINATION_CONTINUE gpu=${gpu} "\
                         "process_count=${process_count} run_elapsed=${run_elapsed} "\
                         "time=$(date -Is)"
                elif (( late_contamination == 0 )); then
                    echo "GPU_CONTAMINATION gpu=${gpu} process_count=${process_count} "\
                         "run_elapsed=${run_elapsed} time=$(date -Is)"
                    contaminated=1
                    kill -TERM -- "-${run_pid}" 2>/dev/null || true
                    sleep 4
                    kill -KILL -- "-${run_pid}" 2>/dev/null || true
                    break
                fi
            fi
            sleep "${poll_seconds}"
        done
        stop_reservation_daemon "${gpu}"
        set +e
        wait "${run_pid}"
        status=$?
        set -e
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
            echo "GPU_LATE_CONTAMINATION_FINISHED gpu=${gpu} status=${status}"
        fi
        if (( contaminated == 1 )); then
            echo "CLEAN_RUN_RETRY gpu=${gpu} status=${status}"
            start_reservation_daemon "${gpu}"
            sleep 3
            continue
        fi
        if (( status == 75 )); then
            echo "SCHEDULING_COLLISION_RETRY gpu=${gpu} status=${status}"
            start_reservation_daemon "${gpu}"
            continue
        fi
        exit "${status}"
    done
    sleep 3
done

echo "Timed out waiting for a clean GPU run" >&2
exit 1
