"""Reserve an idle pair and monitor only verified parent/rank GPU contexts."""
import os,fcntl,json,signal,subprocess,sys,time
from pathlib import Path
import torch
from check_dual_gpu_graph import query,physical

def main():
    locks=[];deadline=time.monotonic()+14400
    print('WAIT_GPU_PAIR='+','.join(physical),flush=True)
    while time.monotonic()<deadline:
        rows,contexts=query()
        if all(r['memory']<=16 and r['util']<=1 for r in rows.values()) and not any(contexts.values()):
            try:
                for gpu in sorted(physical):
                    for kind in ('reserve-',''):
                        stream=(Path.home()/'.tmp'/('team07-unifolm-'+kind+'gpu'+gpu+'.lock')).open('a');locks.append(stream);fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
                rows,contexts=query()
                if all(r['memory']<=16 and r['util']<=1 for r in rows.values()) and not any(contexts.values()):break
            except BlockingIOError:pass
        for stream in locks:stream.close()
        locks=[];time.sleep(2)
    else:raise TimeoutError('No idle GPU pair')
    process=None;telemetry=[]
    try:
        holds=[torch.empty(64*2**20,dtype=torch.uint8,device=i) for i in (0,1)]
        for i in (0,1):torch.cuda.synchronize(i)
        _,contexts=query();owners=set.union(*contexts.values())
        assert len(owners)==1 and all(p==owners for p in contexts.values()),'foreign GPU context during reservation: '+repr(contexts)
        env=dict(os.environ,ROOT_NVML_OWNERS=json.dumps(sorted(owners)),TORCH_NCCL_ASYNC_ERROR_HANDLING='0',NCCL_ASYNC_ERROR_HANDLING='0',CUBLAS_WORKSPACE_CONFIG=':4096:8')
        with Path('ranks.log').open('x') as output:
            process=subprocess.Popen([sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc-per-node=2','check_nccl_geglu_model_rank.py'],env=env,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
        until=time.monotonic()+1800;ready=[Path('rank'+str(i)+'-ready.json') for i in (0,1)]
        while not all(p.exists() for p in ready):
            assert process.poll() is None and time.monotonic()<until
            time.sleep(.05)
        for p in ready:owners.update(json.loads(p.read_text())['host_pids'])
        _,contexts=query();assert all(not(p-owners) for p in contexts.values()),'foreign GPU context before ranks start'
        Path('go').write_text('verified owners\n')
        for gpu in physical:print('CLEAN_RUN_GPU='+gpu,flush=True)
        observations=0
        while process.poll() is None:
            rows,contexts=query();telemetry.append(dict(time_utc=time.time(),GPUs=rows));assert all(not(p-owners) for p in contexts.values()),'foreign GPU context'
            assert time.monotonic()<until,'rank timeout';observations+=1;time.sleep(.25)
        assert process.returncode==0,Path('ranks.log').read_text()[-6000:]
        reports=[json.loads(Path('rank'+str(i)+'-report.json').read_text()) for i in (0,1)]
        assert all(len(r['cases'])==6 and all(e['signedzero_bits'] for c in r['cases'] for e in c['errors']) for r in reports)
        _,contexts=query();assert all(not(p-owners) for p in contexts.values()),'foreign GPU context after ranks finish'
        result=dict(passed=True,clean=True,physical_GPUs=physical,owned_host_contexts=sorted(owners),observations=observations,
                    ranks=reports,ready_for_model=all(p['saved_ms']>0 for r in reports for p in r['pairs']),no_official_score=True)
        Path('nccl-geglu-model-report.json').write_text(json.dumps(result,indent=2)+'\n')
        print('NCCL_GEGLU_MODEL_PAIR_CHECK passed=true clean=true',flush=True)
    finally:
        Path('gpu-telemetry.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in telemetry))
        if process is not None and process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
        for stream in locks:stream.close()

if __name__=='__main__':main()
