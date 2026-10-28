"""Exact-copy NCCL tuning with adopted222 payload histogram, no model score."""
import json
import os
from pathlib import Path
import statistics
import time

rank=int(os.environ['LOCAL_RANK']);pair=os.environ['MONITORED_GPUS'].split(',')
os.environ['CUDA_VISIBLE_DEVICES']=pair[rank]
import torch
import torch.distributed as dist
from check_dual_gpu_graph import query,physical


def time_graph(graph,replays_per_sample):
    for _ in range(8):graph.replay()
    torch.cuda.synchronize();groups=[]
    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    for _ in range(6):
        samples=[]
        for _ in range(5):
            dist.barrier();torch.cuda.synchronize()
            start.record();graph.replay();end.record();end.synchronize()
            samples.append(start.elapsed_time(end)/replays_per_sample)
        groups.append(dict(samples_ms=samples,median_ms=statistics.median(samples)))
    return dict(groups=groups,median_ms=statistics.median(g['median_ms']for g in groups),six_groups=True)


@torch.no_grad()
def main():
    torch.cuda.set_device(0)
    hold=torch.empty(64*2**20,dtype=torch.uint8,device='cuda');torch.cuda.synchronize()
    rows,contexts=query();root=set(json.loads(os.environ['ROOT_NVML_OWNERS']))
    own=contexts[rows[physical[rank]]['uuid']]-root;assert len(own)==1
    workspace=Path.cwd();(workspace/f'rank{rank}-ready.json').write_text(json.dumps(dict(rank=rank,host_pids=sorted(own))))
    until=time.monotonic()+120
    while not(workspace/'go').exists():assert time.monotonic()<until;time.sleep(.02)
    dist.init_process_group('nccl');dist.barrier()
    payloads=json.loads(Path('payloads.json').read_text());sizes=sorted({int(n)for h in payloads['stages'].values()for n in h})
    assert len(sizes)==5
    tensors={};records=[];graphs=[]
    for byte_count in sizes:
        n=byte_count//4;assert n*4==byte_count
        pattern=(torch.arange(n,device='cuda',dtype=torch.int32)%127).float()+.25
        local=pattern+rank*512;original=local.clone()
        output=torch.empty(2*n,device='cuda',dtype=torch.float32)
        expected=torch.cat([pattern,pattern+512]);del pattern
        for _ in range(5):dist.all_gather_into_tensor(output,local)
        torch.cuda.synchronize();assert torch.equal(output.view(torch.int32),expected.view(torch.int32))
        graph=torch.cuda.CUDAGraph();stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.graph(graph,stream=stream):
            for _ in range(32):dist.all_gather_into_tensor(output,local)
        torch.cuda.current_stream().wait_stream(stream);graphs.append(graph)
        states=[]
        for label in ('initial','changed','zero','restored'):
            if label=='changed':local.add_(3.5);target=expected+3.5
            elif label=='zero':local.zero_();target=torch.zeros_like(expected)
            else:local.copy_(original);target=expected
            dist.barrier();torch.cuda.synchronize();graph.replay();torch.cuda.synchronize()
            assert torch.equal(output.view(torch.int32),target.view(torch.int32))
            states.append(dict(state=label,bitwise=True))
        timing=time_graph(graph,32)
        records.append(dict(local_bytes=byte_count,local_MiB=byte_count/2**20,graph_collectives=32,states=states,exact_copy=True,amortized_graph_timing=timing))
        tensors[byte_count]=(local,output,expected)
    phases={}
    for phase,histogram in payloads['stages'].items():
        graph=torch.cuda.CUDAGraph();stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.graph(graph,stream=stream):
            for byte_count,count in sorted((int(n),count)for n,count in histogram.items()):
                local,output,_=tensors[byte_count]
                for _ in range(count):dist.all_gather_into_tensor(output,local)
        torch.cuda.current_stream().wait_stream(stream);graphs.append(graph)
        graph.replay();torch.cuda.synchronize()
        assert all(torch.equal(output.view(torch.int32),expected.view(torch.int32))for _,output,expected in tensors.values())
        phases[phase]=dict(calls=sum(histogram.values()),local_MiB=sum(int(n)*c for n,c in histogram.items())/2**20,timing=time_graph(graph,1),
                           size_histogram_surrogate_only=True,not_actual_model_interleaving_or_inference_score=True)
    report=dict(rank=rank,all_gather_exact=True,graph_exact=True,records=records,phases=phases,torch_version=torch.__version__,NCCL_version=list(torch.cuda.nccl.version()),
                config=json.loads(Path('tuning.json').read_text()),no_model_or_video_quality_claim=True,no_official_score=True)
    (workspace/f'rank{rank}-report.json').write_text(json.dumps(report,indent=2)+'\n')
    for graph in graphs:graph.reset()
    torch.cuda.synchronize();dist.barrier();dist.destroy_process_group()
    print('NCCL_TUNING222_CHECK '+json.dumps(dict(rank=rank,phase_ms={k:v['timing']['median_ms']for k,v in phases.items()})),flush=True)


if __name__=='__main__':main()
