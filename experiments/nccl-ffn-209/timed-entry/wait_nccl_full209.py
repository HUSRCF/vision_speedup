"""Reserve two devices for an actual evaluator run, retaining failed attempts."""
import fcntl,hashlib,json,os,shutil,signal,subprocess,sys,tarfile,time
from datetime import datetime,timezone
from pathlib import Path
import torch
from check_dual_gpu_graph import query,physical

def reserve():
    deadline=time.monotonic()+14400;locks=[]
    print('WAIT_GPU_PAIR='+','.join(physical),flush=True)
    while time.monotonic()<deadline:
        rows,contexts=query()
        if all(r['memory']<=16 and r['util']<=1 for r in rows.values()) and not any(contexts.values()):
            try:
                for gpu in sorted(physical):
                    for kind in ('reserve-',''):
                        stream=(Path.home()/'.tmp'/('team07-unifolm-'+kind+'gpu'+gpu+'.lock')).open('a')
                        locks.append(stream);fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
                rows,contexts=query()
                if all(r['memory']<=16 and r['util']<=1 for r in rows.values()) and not any(contexts.values()):return locks
            except BlockingIOError:pass
        for stream in locks:stream.close()
        locks=[];time.sleep(2)
    raise TimeoutError('No idle GPU pair')

def main():
    workspace=Path.cwd();base=Path(os.environ['CANONICAL_ROOT']);flags=json.loads((workspace/'flags.json').read_text())
    manifest=json.loads((base/'nccl-parallel-production-sources.json').read_text())
    assert all(hashlib.sha256((base/n).read_bytes()).hexdigest()==h for n,h in manifest.items())
    locks=reserve();process=None;run=None
    try:
        holds=[torch.empty(64*2**20,dtype=torch.uint8,device=i) for i in (0,1)]
        for i in (0,1):torch.cuda.synchronize(i)
        rows,contexts=query();owners=set.union(*contexts.values())
        assert len(owners)==1 and all(p==owners for p in contexts.values()),'foreign GPU context during reservation'
        assert shutil.disk_usage(base).free>32*2**20
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        run=Path.home()/'benchmark-runs/unifolm'/(stamp+'-case-v2-nccl-'+str(os.getpid()));run.mkdir(exist_ok=False)
        (workspace/'run-directory.txt').write_text(str(run)+'\n')
        environment=dict(os.environ,**flags)
        environment.update(CUDA_VISIBLE_DEVICES=physical[0],TEAM07_PARALLEL_RANK='0',TEAM07_PARALLEL_DIR=str(run/'parallel'),ROOT_NVML_OWNERS=json.dumps(sorted(owners)),RESULT_DIR=str(run/'outputs'),PROFILE_DIR=str(run/'profiles'),RESULT_JSON_PATH=str(run/'result.json'),LOG_PATH=str(run/'runtime.log'),BENCHMARK_N_ITER=os.environ['TRIAL_N_ITER'],UNIFOLM_CASE='unitree_z1_dual_arm_stackbox_v2',PYTHONPATH=str(base/'src'),PYTHONUNBUFFERED='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',TORCH_NCCL_ASYNC_ERROR_HANDLING='0',NCCL_ASYNC_ERROR_HANDLING='0')
        assert environment['TRIAL_N_ITER'] in ('1','original')
        (run/'profiles').mkdir()
        provenance=['source_root='+str(base),'CUDA_VISIBLE_DEVICES='+physical[0],'MONITORED_GPUS='+','.join(physical)]+[k+'='+v for k,v in sorted(flags.items())]
        (run/'provenance.txt').write_text('\n'.join(provenance)+'\n')
        (run/'parallel-effective-flags.json').write_text(json.dumps(flags,indent=2)+'\n')
        snapshot_files=[p for folder in ('src','scripts/evaluation','configs') for p in (base/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
        snapshot_files+=[base/n for n in ('scripts/run_compute_benchmark.sh','scripts/team07_env.sh','nccl-parallel-production-sources.json','nccl-entry-timing-proof.json')]
        with tarfile.open(run/'source.tar.gz','w:gz') as tar:
            for path in sorted(set(snapshot_files)):tar.add(path,arcname='./'+str(path.relative_to(base)),recursive=False)
        with (workspace/'benchmark.log').open('x') as log:
            process=subprocess.Popen(['bash','scripts/run_compute_benchmark.sh'],cwd=base,env=environment,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        until=time.monotonic()+180;parallel=run/'parallel';ready=[parallel/('rank'+str(i)+'-ready.json') for i in (0,1)]
        while not all(p.exists() for p in ready):
            if list(parallel.glob('rank*-foreign.json')):raise AssertionError('foreign GPU context during rank allocation')
            assert process.poll() is None and time.monotonic()<until,'benchmark startup failed: '+(workspace/'benchmark.log').read_text()[-5000:]
            time.sleep(.05)
        per_gpu={}
        for i,p in enumerate(ready):
            row=json.loads(p.read_text());assert row['rank']==i and row['physical_gpu']==physical[i]
            rank_owners=set(row['host_pids']);assert len(rank_owners)==1 and not(rank_owners&owners)
            per_gpu[rows[physical[i]]['uuid']]=owners|rank_owners
        _,contexts=query();assert contexts==per_gpu,'foreign GPU context before go'
        (parallel/'go').write_text('verified parent and both rank contexts\n')
        until=time.monotonic()+1800;observations=0
        with (workspace/'gpu-telemetry.jsonl').open('x') as telemetry:
            while process.poll() is None:
                rows,contexts=query()
                assert all(not(p-per_gpu[u]) for u,p in contexts.items()),'foreign GPU context'
                assert time.monotonic()<until,'full inference timeout'
                telemetry.write(json.dumps(dict(time_utc=datetime.now(timezone.utc).isoformat(),gpus=rows,contexts={u:sorted(p) for u,p in contexts.items()}))+'\n');telemetry.flush()
                observations+=1;time.sleep(.5)
        _,contexts=query();assert all(not(p-per_gpu[u]) for u,p in contexts.items()),'foreign GPU context at completion'
        assert (parallel/'parallel-complete.json').is_file(),(workspace/'benchmark.log').read_text()[-5000:]
        proof=json.loads((parallel/'parallel-complete.json').read_text())
        expected_rounds=1 if environment['TRIAL_N_ITER']=='1' else 11
        assert proof['video_count']==2*expected_rounds+1 and proof['rank_videos_SHA256_equal']
        report=dict(clean=True,run_directory=str(run),physical_GPUs=physical,observations=observations,benchmark_exit_code=process.returncode,parallel=proof,smoke_only=expected_rounds==1,no_official_adoption=True)
        (workspace/'nccl-full-report.json').write_text(json.dumps(report,indent=2)+'\n')
        print('NCCL_FULL_CASE_CHECK clean=true run='+str(run),flush=True)
    except BaseException:
        if run is not None:(run/'timing-contaminated.txt').write_text('Incomplete or failed whole-run guard; not eligible for scoring.\n')
        raise
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
        for stream in locks:stream.close()

if __name__=='__main__':main()
