"""One complete V2 quality trial; preserve official evaluator timing and tools."""
from datetime import datetime,timezone
import hashlib,json,os,subprocess,tarfile,threading,time
from pathlib import Path


def main():
    root=Path.cwd();stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run=Path.home()/'benchmark-runs/unifolm'/(stamp+'-v2-balanced-int8-'+str(os.getpid()))
    run.mkdir(exist_ok=False);print('DIAGNOSTIC_DIR='+str(run),flush=True)
    flags=json.loads(Path(os.environ['CURRENT_EXPERIMENT_FLAGS_FILE']).read_text())
    env={k:v for k,v in os.environ.items() if not k.startswith('UNIFOLM_')};env.update(flags)
    env.update(RESULT_DIR=str(run/'outputs'),PROFILE_DIR=str(run/'profiles'),
        RESULT_JSON_PATH=str(run/'result.json'),LOG_PATH=str(run/'runtime.log'),
        BENCHMARK_N_ITER='original',UNIFOLM_CASE='unitree_z1_dual_arm_stackbox_v2')
    (run/'flags.json').write_text(json.dumps(flags,indent=2)+'\n')
    paths=[p for folder in ('src','configs','scripts/evaluation') for p in (root/folder).rglob('*')
        if p.is_file() and '__pycache__' not in p.parts]
    names=json.loads((root/'balanced-full-added-files.json').read_text())
    paths.extend(root/n for n in [*names,'balanced-full-added-files.json','world-head-filter-cache-production-sources.json',
        'scripts/team07_env.sh','scripts/run_compute_benchmark.sh'])
    with tarfile.open(run/'source.tar.gz','w:gz') as out:
        for path in sorted(set(paths)):out.add(path,arcname='./'+str(path.relative_to(root)),recursive=False)
    stop=threading.Event();errors=[]
    def monitor():
        with (run/'gpu-telemetry.jsonl').open('x') as output:
            while not stop.is_set():
                try:
                    p=subprocess.run(['nvidia-smi','-i',os.environ['CUDA_VISIBLE_DEVICES'],
                        '--query-gpu=index,utilization.gpu,memory.used,temperature.gpu,clocks.current.sm',
                        '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
                    output.write(json.dumps(dict(time_utc=datetime.now(timezone.utc).isoformat(),csv=p.stdout.strip()))+'\n');output.flush()
                except Exception as exc:errors.append(str(exc))
                stop.wait(.5)
    worker=threading.Thread(target=monitor);worker.start();start=time.monotonic()
    try:code=subprocess.run(['bash','scripts/run_balanced_compute_benchmark.sh'],env=env,check=False).returncode
    finally:stop.set();worker.join()
    parent_wall=time.monotonic()-start
    result=json.loads((run/'result.json').read_text())
    row=next(r for r in result['cases'] if r['case']=='unitree_z1_dual_arm_stackbox_v2')
    complete=row['rounds']==11 and row['videos']==23 and row['execution_ok'] and row['profile_complete']
    assert complete and not errors,dict(complete=complete,monitor_errors=errors)
    report=dict(workload_complete=True,case_result=row,benchmark_exit_code=code,
        parent_wall_s=parent_wall,run_directory=str(run),single_GPU_quality_trial=True,
        current_two_GPU_production_not_replaced=True,official_adopted=False,
        calibration_frozen_first_eager_per_weight_owner=True,
        dynamic_input_and_weight_INT8_pack_inside_every_replay=True)
    (run/'balanced-full-report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('BALANCED_INT8_V2_QUALITY '+json.dumps(report),flush=True)
    print('BALANCED_INT8_V2_QUALITY_COMPLETE passed=true quality_passed='+str(row['psnr_passed']).lower(),flush=True)


if __name__=='__main__':main()
