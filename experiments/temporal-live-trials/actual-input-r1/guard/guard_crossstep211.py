"""Run an immutable script-only experiment on one idle, locked GPU.

The parent does not import Torch or create a CUDA context. A worker announces
its verified NVML host context before computation; every subsequent poll must
contain only that context. Only this parent's process group can be terminated.
"""
import argparse,fcntl,json,os,signal,subprocess,sys,time
from pathlib import Path


def query(gpu):
    text=subprocess.check_output(['nvidia-smi','-i',gpu,
        '--query-gpu=index,uuid,memory.used,utilization.gpu,temperature.gpu,clocks.sm',
        '--format=csv,noheader,nounits'],text=True)
    index,uuid,memory,util,temp,clock=[v.strip() for v in text.strip().split(',')]
    assert index==gpu
    row=dict(uuid=uuid,memory=int(memory),util=int(util),temperature=int(temp),clock=int(clock))
    contexts=set()
    text=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid',
        '--format=csv,noheader,nounits'],text=True)
    for line in text.splitlines():
        owner,pid=[v.strip() for v in line.split(',')]
        if owner==uuid:contexts.add(int(pid))
    return row,contexts


def announce_worker():
    import torch
    gpu=os.environ['MONITORED_GPU']
    assert torch.cuda.device_count()==1
    hold=torch.empty(64*2**20,dtype=torch.uint8,device='cuda')
    torch.cuda.synchronize()
    row,contexts=query(gpu)
    assert len(contexts)==1,'foreign GPU context during worker allocation'
    directory=Path(os.environ['TEAM07_GUARD_ATTEMPT_DIR'])
    (directory/'worker-ready.json').write_text(json.dumps(dict(gpu=gpu,uuid=row['uuid'],
        owned_host_contexts=sorted(contexts)))+'\n')
    deadline=time.monotonic()+60
    while not (directory/'go').exists():
        assert time.monotonic()<deadline
        time.sleep(.02)
    return hold


def attempt(root,entry,gpu,index):
    directory=root/('attempt-'+str(index));directory.mkdir(exist_ok=False)
    row,contexts=query(gpu)
    assert row['memory']<=16 and row['util']<=1 and not contexts,'GPU no longer idle before allocation'
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,MONITORED_GPU=gpu,TEAM07_GUARD_ATTEMPT_DIR=str(directory))
    process=None
    records=[]
    try:
        with (directory/'worker.log').open('x') as output:
            process=subprocess.Popen([sys.executable,str(root/entry)],cwd=directory,env=env,
                stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
        deadline=time.monotonic()+90
        ready=directory/'worker-ready.json'
        while not ready.exists():
            assert process.poll() is None,(directory/'worker.log').read_text()[-6000:]
            assert time.monotonic()<deadline,'worker CUDA initialization timed out'
            time.sleep(.05)
        owner=json.loads(ready.read_text());owners=set(owner['owned_host_contexts'])
        assert owner['gpu']==gpu and owner['uuid']==row['uuid'] and len(owners)==1
        _,contexts=query(gpu)
        assert contexts<=owners,'foreign GPU context before computation'
        (directory/'go').write_text('verified single worker CUDA context\n')
        deadline=time.monotonic()+1800
        while process.poll() is None:
            row,contexts=query(gpu)
            records.append(dict(time_utc=time.time(),gpu=gpu,**row))
            assert contexts<=owners,'foreign GPU context'
            assert time.monotonic()<deadline,'worker computation timed out'
            time.sleep(.25)
        assert process.returncode==0,(directory/'worker.log').read_text()[-8000:]
        _,contexts=query(gpu)
        assert contexts<=owners,'foreign GPU context after computation'
        report=json.loads((directory/'comparison.json').read_text())
        assert report['diagnostic_complete'] and report['all_dynamic_gates_passed']
        return dict(clean=True,GPU=gpu,owned_host_contexts=sorted(owners),
            observations=len(records),attempt=index,diagnostic=report)
    finally:
        (directory/'gpu-telemetry.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in records))
        if process is not None and process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL);process.wait()


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',choices=tuple('0123'),required=True)
    p.add_argument('--entry',required=True);args=p.parse_args()
    root=Path.cwd();assert (root/args.entry).is_file()
    locks=[]
    for kind in ('reserve-',''):
        stream=(Path.home()/'.tmp'/('team07-unifolm-'+kind+'gpu'+args.gpu+'.lock')).open('a')
        locks.append(stream)
        fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        for index in range(1,17):
            row,contexts=query(args.gpu)
            if row['memory']>16 or row['util']>1 or contexts:
                print('WAIT_IDLE_GPU='+args.gpu,flush=True)
                deadline=time.monotonic()+1800
                while row['memory']>16 or row['util']>1 or contexts:
                    assert time.monotonic()<deadline,'No idle GPU window'
                    time.sleep(2);row,contexts=query(args.gpu)
            print('SINGLE_TMP_ATTEMPT='+str(index),flush=True)
            try:
                result=attempt(root,args.entry,args.gpu,index)
            except AssertionError as exc:
                if 'foreign GPU context' in str(exc):
                    print('FOREIGN_CONTEXT_ATTEMPT_EXCLUDED='+str(index),flush=True)
                    time.sleep(2);continue
                raise
            (root/'guard-report.json').write_text(json.dumps(result,indent=2)+'\n')
            print('SINGLE_TMP_GPU_CHECK passed=true clean=true gpu='+args.gpu,flush=True)
            break
        else:raise RuntimeError('Too many foreign-context attempts')
    finally:
        for stream in locks:stream.close()


if __name__=='__main__':main()
