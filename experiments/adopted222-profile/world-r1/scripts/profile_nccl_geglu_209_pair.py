"""Profile the unchanged adopted222 graph; export scalar kernel aggregates only."""
import argparse,json,runpy,statistics,sys
from collections import defaultdict
from pathlib import Path
import torch
import torch.distributed as dist
import unifolm_wma.utils.inference_runtime as runtime
from unifolm_wma.utils.cuda_graph import DenoiserGraph
from unifolm_wma.utils.nccl_initial_block222_mode import install,mode,uninstall
from torch.utils._pytree import tree_flatten

class DiagnosticComplete(Exception):pass


def main():
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--diagnostic-dir',type=Path,required=True);p.add_argument('--profile-stage',choices=['decision','world'],required=True)
    options,args=p.parse_known_args();options.diagnostic_dir.mkdir(exist_ok=False)
    previous=runtime.configure_denoiser
    def configure(model,*a,**k):
        previous(model,*a,**k);original=model.apply_model;assert isinstance(original,DenoiserGraph)
        controller=install(model.model.diffusion_model)
        def apply(*values,**keywords):
            if torch.backends.cudnn.allow_tf32!=(options.profile_stage=='decision'):
                with mode(True):return original(*values,**keywords)
            try:
                with mode(True):
                    original(*values,**keywords);torch.cuda.synchronize()
                    initial,_=tree_flatten(original(*values,**keywords));reference=[v.clone()for v in initial]
                    rng=torch.cuda.get_rng_state();checks=[]
                    for i in range(6):
                        actual,_=tree_flatten(original(*values,**keywords));assert len(actual)==len(reference)==3
                        exact=all(torch.equal(a.contiguous().view(torch.int32),b.contiguous().view(torch.int32))for a,b in zip(actual,reference));assert exact
                        checks.append(dict(replay=i,three_outputs_signedzero_bitwise=True))
                    assert torch.equal(rng,torch.cuda.get_rng_state())
                    for _ in range(20):original(*values,**keywords)
                    torch.cuda.synchronize();start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);groups=[]
                    for i in range(6):
                        dist.barrier();torch.cuda.synchronize();samples=[]
                        for _ in range(7):
                            start.record();original(*values,**keywords);end.record();end.synchronize();samples.append(start.elapsed_time(end))
                        groups.append(dict(group=i,samples_ms=samples,median_ms=statistics.median(samples)))
                    dist.barrier();torch.cuda.synchronize();replays=5
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=False,profile_memory=False,with_stack=False)as prof:
                        for _ in range(replays):original(*values,**keywords)
                        torch.cuda.synchronize()
                    kernels=defaultdict(list)
                    for event in prof.events():
                        if str(event.device_type).endswith('CUDA'):
                            duration=float(event.self_cuda_time_total)
                            assert duration>=0;kernels[event.name].append(duration)
                    assert kernels,'Profiler did not observe CUDA events'
                    rows=[dict(name=n,count=len(v),total_us=sum(v),per_forward_ms=sum(v)/replays/1000,mean_us=statistics.mean(v))for n,v in kernels.items()]
                    rows.sort(key=lambda r:r['total_us'],reverse=True)
                    summary=dict(profile_only=True,stage=options.profile_stage,baseline_source_count=222,arithmetic_or_routing_changed=False,
                        no_official_score=True,full_video_PSNR_unverified=True,replay_checks=checks,RNG_unchanged=True,
                        unprofiled_timing_groups=groups,unprofiled_median_ms=statistics.median(v for g in groups for v in g['samples_ms']),
                        profiler_replays=replays,CUDA_event_count=sum(r['count']for r in rows),kernel_rows=rows,
                        kernel_sum_per_forward_ms=sum(r['per_forward_ms']for r in rows),controller=controller.report(),
                        aggregate_only_no_raw_trace_or_addresses=True,profiled_durations_include_profiler_perturbation=True)
                    (options.diagnostic_dir/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
                    print('ADOPTED222_PROFILE '+json.dumps({k:summary[k]for k in ['stage','unprofiled_median_ms','CUDA_event_count','kernel_sum_per_forward_ms']}),flush=True)
            finally:
                for entry in original.entries.values():
                    for key in ['numerical_sgemm','native_weight_cache']:
                        if entry.get(key)is not None:entry[key].close()
                    entry['graph'].reset()
                original.entries.clear()
            raise DiagnosticComplete
        model.apply_model=apply
    runtime.configure_denoiser=configure
    script=Path(__file__).parent/'evaluation/world_model_interaction.py';sys.path.insert(0,str(script.parent.resolve()));sys.argv=[str(script),*args]
    try:runpy.run_path(str(script),run_name='__main__')
    except DiagnosticComplete:pass
    finally:runtime.configure_denoiser=previous;uninstall()

if __name__=='__main__':main()
