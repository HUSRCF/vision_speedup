"""Single-process GPU experiment admission and whole-attempt context monitoring."""
import fcntl
import json
import os
from pathlib import Path
import runpy
import subprocess
import threading
import time

import torch


def query(gpu):
    text = subprocess.check_output(['nvidia-smi', '-i', gpu, '--query-gpu=uuid,memory.used,utilization.gpu,temperature.gpu,clocks.sm', '--format=csv,noheader,nounits'], text=True)
    uuid, mem, util, temp, clock = [x.strip() for x in text.strip().split(',')]
    rows = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'], text=True)
    contexts = {int(line.split(',')[1]) for line in rows.splitlines() if line.split(',')[0].strip() == uuid}
    return dict(memory=int(mem), util=int(util), temperature=int(temp), clock=int(clock)), contexts


def main():
    gpu = os.environ['MONITORED_GPU']
    assert gpu in '0123' and len(gpu) == 1 and os.environ['CUDA_VISIBLE_DEVICES'] == gpu
    locks = []
    observations = []
    failures = []
    stop = threading.Event()
    monitor = None
    try:
        row, contexts = query(gpu)
        assert row['memory'] <= 16 and row['util'] <= 1 and not contexts
        for prefix in ('reserve-', ''):
            stream = (Path.home()/'.tmp'/('team07-unifolm-'+prefix+'gpu'+gpu+'.lock')).open('a')
            locks.append(stream)
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        row, contexts = query(gpu)
        assert row['memory'] <= 16 and row['util'] <= 1 and not contexts
        hold = torch.empty(64 * 2**20, dtype=torch.uint8, device=0)
        torch.cuda.synchronize()
        row, owners = query(gpu)
        assert len(owners) == 1

        def watch():
            while not stop.is_set():
                try:
                    row, current = query(gpu)
                    observations.append(dict(time_utc=time.time(), **row))
                    if current - owners:
                        failures.append('foreign context')
                        break
                except Exception as exc:
                    failures.append(type(exc).__name__)
                    break
                stop.wait(.5)
        monitor = threading.Thread(target=watch)
        monitor.start()
        runpy.run_path(os.environ['EXPERIMENT_ENTRY'], run_name='__main__')
        _, current = query(gpu)
        assert not (current - owners)
        stop.set()
        monitor.join()
        assert not failures, failures
        Path(os.environ['EXPERIMENT_CLEAN_REPORT']).write_text(json.dumps(dict(clean=True, gpu=gpu, samples=len(observations), whole_attempt_context_guard=True), indent=2)+'\n')
        print('SINGLE_PROCESS_CHECK clean=true gpu='+gpu, flush=True)
    finally:
        stop.set()
        if monitor is not None:
            monitor.join()
        Path(os.environ['EXPERIMENT_TELEMETRY']).write_text(''.join(json.dumps(row)+'\n' for row in observations))
        for stream in locks:
            stream.close()


if __name__ == '__main__':
    main()
