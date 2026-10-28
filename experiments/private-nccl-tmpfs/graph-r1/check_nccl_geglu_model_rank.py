"""Small exact-copy communication diagnostic, no model/input/score."""
import os,json,sys,time,statistics
from pathlib import Path
rank=int(os.environ['LOCAL_RANK']);pair=os.environ['MONITORED_GPUS'].split(',')
policy=os.environ['COMM_VISIBILITY']
assert policy in ['one','all']
os.environ['CUDA_VISIBLE_DEVICES']=pair[rank] if policy=='one' else ','.join(pair)
import torch
import torch.distributed as dist
from check_dual_gpu_graph import query,physical
device=0 if policy=='one' else rank
torch.cuda.set_device(device)
hold=torch.empty(64*2**20,dtype=torch.uint8,device='cuda');torch.cuda.synchronize()
rows,contexts=query();root=set(json.loads(os.environ['ROOT_NVML_OWNERS']))
own=contexts[rows[physical[rank]]['uuid']]-root
assert len(own)==1,'rank CUDA context ownership unresolved'
workspace=Path.cwd();(workspace/f'rank{rank}-ready.json').write_text(json.dumps(dict(rank=rank,host_pids=sorted(own))))
until=time.monotonic()+60
while not (workspace/'go').exists():assert time.monotonic()<until;time.sleep(.02)
dist.init_process_group('nccl');dist.barrier()
records=[]
with torch.no_grad():
    for mib in [1,10,25,40]:
        n=mib*2**20//4
        local=torch.full((n,),rank+.25,dtype=torch.float32,device='cuda')
        output=torch.empty((2*n,),dtype=torch.float32,device='cuda')
        expected=torch.cat([torch.full_like(local,.25),torch.full_like(local,1.25)])
        for _ in range(5):dist.all_gather_into_tensor(output,local)
        torch.cuda.synchronize();assert torch.equal(output.view(torch.int32),expected.view(torch.int32))
        graph=torch.cuda.CUDAGraph();stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.graph(graph,stream=stream):dist.all_gather_into_tensor(output,local)
        torch.cuda.current_stream().wait_stream(stream)
        graph.replay();torch.cuda.synchronize();assert torch.equal(output.view(torch.int32),expected.view(torch.int32))
        modes={}
        for label,fn in [('eager',lambda:dist.all_gather_into_tensor(output,local)),('graph',graph.replay)]:
            for _ in range(10):fn()
            torch.cuda.synchronize();samples=[];start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            for _ in range(20):
                dist.barrier();torch.cuda.synchronize();start.record();fn();end.record();end.synchronize();samples.append(start.elapsed_time(end))
            assert torch.equal(output.view(torch.int32),expected.view(torch.int32))
            ms=statistics.median(samples)
            modes[label]=dict(samples_ms=samples,median_ms=ms,local_payload_GBs=mib*2**20/(ms/1000)/1e9)
        records.append(dict(local_MiB=mib,two_rank_all_gather=True,exact_copy=True,modes=modes))
        graph.reset()
report=dict(visibility=policy,NCCL_SHM_DISABLE=os.environ.get('NCCL_SHM_DISABLE'),rank=rank,
    all_gather_exact=True,graph_exact=True,records=records,no_official_score=True)
(workspace/f'rank{rank}-report.json').write_text(json.dumps(report,indent=2)+'\n')
torch.cuda.synchronize();dist.barrier();dist.destroy_process_group()
print('COMM_VISIBILITY_CHECK '+json.dumps(dict(rank=rank,visibility=policy,records=[dict(local_MiB=r['local_MiB'],graph_ms=r['modes']['graph']['median_ms'])for r in records])),flush=True)
