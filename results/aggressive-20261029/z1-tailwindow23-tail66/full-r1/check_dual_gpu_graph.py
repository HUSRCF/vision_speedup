"""Bounded two-GPU graph/P2P feasibility test, never an inference score."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import fcntl,json,statistics,subprocess,threading,time
from pathlib import Path
import torch

physical=os.environ['MONITORED_GPUS'].split(',');assert len(physical)==2

def query():
    p=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu,temperature.gpu,clocks.sm','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    rows={}
    for line in p.stdout.splitlines():
        i,uuid,memory,util,temp,clock=[v.strip() for v in line.split(',')]
        if i in physical:rows[i]=dict(uuid=uuid,memory=int(memory),util=int(util),temperature=int(temp),clock=int(clock))
    assert len(rows)==2
    p=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    contexts={r['uuid']:set() for r in rows.values()}
    for line in p.stdout.splitlines():
        uuid,pid=[v.strip() for v in line.split(',')]
        if uuid in contexts:contexts[uuid].add(int(pid))
    return rows,contexts

def synchronize():
    for device in (0,1):torch.cuda.synchronize(device)

@torch.no_grad()
def main():
    print('WAIT_GPU_PAIR='+','.join(physical),flush=True)
    deadline=time.monotonic()+14400;locks=[]
    while time.monotonic()<deadline:
        rows,contexts=query()
        if all(r['memory']<=16 and r['util']<=1 for r in rows.values()) and not any(contexts.values()):
            try:
                for gpu in sorted(physical):
                    for name in ('reserve-',''):
                        p=Path.home()/'.tmp'/('team07-unifolm-'+name+'gpu'+gpu+'.lock');p.parent.mkdir(exist_ok=True)
                        stream=p.open('a');locks.append(stream);fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
                rows,contexts=query()
                if all(r['memory']<=16 and r['util']<=1 for r in rows.values()) and not any(contexts.values()):break
            except BlockingIOError:pass
        for stream in locks:stream.close()
        locks=[];time.sleep(2)
    else:raise TimeoutError('No idle GPU pair before deadline')
    stop=threading.Event();foreign=[];monitor=None
    try:
        assert torch.cuda.device_count()==2
        holds=[torch.empty(64*2**20,dtype=torch.uint8,device=i) for i in (0,1)];synchronize()
        rows,contexts=query();owners=set.union(*contexts.values());assert len(owners)==1 and all(p==owners for p in contexts.values()),contexts
        def watch():
            while not stop.wait(.25):
                try:
                    _,current=query()
                    if any(p-owners for p in current.values()):foreign.append({u:sorted(p) for u,p in current.items()})
                except Exception as exc:foreign.append(dict(monitor_error=str(exc)))
        monitor=threading.Thread(target=watch);monitor.start()
        for gpu in physical:print('CLEAN_RUN_GPU='+gpu,flush=True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.manual_seed(8217)
        peer=bool(torch.cuda.can_device_access_peer(0,1) and torch.cuda.can_device_access_peer(1,0));assert peer
        src=torch.arange(64*2**20,dtype=torch.float32,device=0);dst=torch.empty_like(src,device=1)
        dst.copy_(src);synchronize();assert torch.equal(dst.to(0),src)
        copies=[]
        for _ in range(5):
            synchronize();begin=time.perf_counter()
            for _ in range(8):dst.copy_(src,non_blocking=True)
            synchronize();copies.append((time.perf_counter()-begin)/8)
        peer_GBs=src.numel()*src.element_size()/statistics.median(copies)/1e9
        a0=torch.randn(2048,2048,device=0);b0=torch.randn_like(a0)
        a=[a0,a0.to(1)];b=[b0,b0.to(1)];out=[torch.empty_like(a0,device=i) for i in (0,1)]
        streams=[torch.cuda.Stream(device=i) for i in (0,1)]
        for i in (0,1):
            with torch.cuda.device(i),torch.cuda.stream(streams[i]):torch.mm(a[i],b[i],out=out[i])
        synchronize();graphs=[]
        for i in (0,1):
            with torch.cuda.device(i):
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=streams[i]):torch.mm(a[i],b[i],out=out[i])
                graphs.append(graph)
        exact=[]
        for mode in ('initial','changed','zero','restored'):
            for i in (0,1):
                if mode=='changed':a[i].add_(.001)
                elif mode=='zero':a[i].zero_()
                elif mode=='restored':a[i].fill_(.25)
                with torch.cuda.device(i):graphs[i].replay()
            synchronize()
            for i in (0,1):
                expected=a[i]@b[i]
                assert torch.equal(out[i].view(torch.int32),expected.view(torch.int32))
            exact.append(mode)
        pairs=[]
        for index in range(6):
            row={}
            for mode in (('sequential','parallel') if index%2==0 else ('parallel','sequential')):
                samples=[]
                for _ in range(7):
                    synchronize();begin=time.perf_counter()
                    for i in (0,1):
                        with torch.cuda.device(i):graphs[i].replay()
                        if mode=='sequential':torch.cuda.synchronize(i)
                    synchronize();samples.append((time.perf_counter()-begin)*1000)
                row[mode+'_ms']=statistics.median(samples)
            pairs.append(row)
        # Test a joined, single graph spanning the two primary contexts last;
        # capture support is reported separately from the proven split route.
        unified=dict(passed=False)
        try:
            e0,e1=torch.cuda.Event(),torch.cuda.Event()
            e0.record(streams[0]);e1.record(streams[1]);synchronize()
            copied=torch.empty_like(out[0]);graph=torch.cuda.CUDAGraph()
            with torch.cuda.device(0),torch.cuda.graph(graph,stream=streams[0]):
                e0.record(streams[0]);streams[1].wait_event(e0)
                with torch.cuda.device(1),torch.cuda.stream(streams[1]):
                    torch.mm(a[1],b[1],out=out[1]);e1.record(streams[1])
                torch.mm(a[0],b[0],out=out[0]);streams[0].wait_event(e1)
                copied.copy_(out[1],non_blocking=True)
            with torch.cuda.device(0):graph.replay()
            synchronize();assert torch.equal(copied.view(torch.int32),out[1].to(0).view(torch.int32))
            unified=dict(passed=True,peer_copy_and_both_device_kernels=True)
        except Exception as exc:unified=dict(passed=False,error=type(exc).__name__+': '+str(exc))
        stop.set();monitor.join();_,current=query();assert not foreign and all(not(p-owners) for p in current.values())
        report=dict(passed=True,physical_GPUs=physical,NVML_owned_contexts=sorted(owners),peer_access=peer,
                    peer_copy_GBs=peer_GBs,peer_copy_wall_samples_s=copies,split_graph_int32_modes=exact,
                    paired_split_graph_wall_ms=pairs,unified_graph=unified,whole_attempt_foreign_contexts=False,
                    first_GPU_state=rows,no_inference_score=True,no_precision_reduction=True)
        Path('dual-gpu-graph-report.json').write_text(json.dumps(report,indent=2)+'\n')
        print('DUAL_GPU_GRAPH_CHECK passed=true clean=true unified='+str(unified['passed']),flush=True)
    finally:
        stop.set()
        if monitor is not None:monitor.join()
        for stream in locks:stream.close()

if __name__=='__main__':main()
