"""Launch the second evaluator after the original process timer has begun.

Importing this module performs no Torch import and no GPU/model initialization.
Both original evaluators execute the whole workload. Only rank zero contributes
official evidence; rank one's private videos are compared before rank zero ends.
"""
import atexit,hashlib,json,os,subprocess,sys,time
from datetime import timedelta
from pathlib import Path

_child=None
_child_log=None
_store=None
_hold=None
_controller=None
_mode=None
_root=None
_rank=None
_stage_captures={}
_stage_calls={}
_denoiser_graph=None

def requested():
    value=os.environ.get('UNIFOLM_NCCL_PARALLEL','0')
    assert value in ('0','world-ffn','both-ffn','world-ffn-temporal','both-spatial-temporal','both-spatial-temporal-resblock-grid','both-spatial-temporal-resblock-grid-native-attention','both-spatial-temporal-resblock-grid-native-attention-initial-ffn-temporal-block-initial-block')
    return value

def argument(arguments,name):
    assert arguments.count(name)==1,name
    index=arguments.index(name)+1
    assert index<len(arguments)
    return arguments[index]

def helper_arguments(arguments,root):
    result=list(arguments)
    assert argument(result,'--dataset')=='unitree_z1_dual_arm_stackbox_v2'
    for flag,value in (('--savedir',root/'worker-videos'),('--profile_path',root/'worker-profile.json')):
        argument(result,flag);result[result.index(flag)+1]=str(value)
    # No sampler/input/seed/round/shape argument is changed.
    return result

def cleanup():
    global _child_log
    if _child is not None and _child.poll() is None:
        _child.terminate()
        try:_child.wait(timeout=10)
        except subprocess.TimeoutExpired:_child.kill();_child.wait()
    if _child_log is not None:_child_log.close();_child_log=None

def early_start():
    global _child,_child_log,_root,_rank
    if requested()=='0':return
    assert _rank is None and 'torch' not in sys.modules
    pair=os.environ['MONITORED_GPUS'].split(',');assert len(pair)==2 and len(set(pair))==2
    _rank=int(os.environ['TEAM07_PARALLEL_RANK']);assert _rank in (0,1)
    assert os.environ['CUDA_VISIBLE_DEVICES']==pair[_rank]
    assert argument(sys.argv,'--dataset')=='unitree_z1_dual_arm_stackbox_v2'
    _root=Path(os.environ['TEAM07_PARALLEL_DIR']);assert _root.is_absolute()
    if _rank==0:
        _root.mkdir(exist_ok=False)
        arguments=helper_arguments(sys.argv,_root)
        environment=dict(os.environ,TEAM07_PARALLEL_RANK='1',CUDA_VISIBLE_DEVICES=pair[1])
        assert not any(k in environment for k in ('RANK','LOCAL_RANK','WORLD_SIZE'))
        _child_log=(_root/'worker.log').open('x')
        _child=subprocess.Popen([sys.executable,*arguments],env=environment,
            stdin=subprocess.DEVNULL,stdout=_child_log,stderr=subprocess.STDOUT)
        atexit.register(cleanup)
    else:assert _root.is_dir()

def wait_file(path,seconds=180):
    deadline=time.monotonic()+seconds
    while not path.exists():
        assert time.monotonic()<deadline,'parallel rendezvous timeout: '+str(path)
        if _child is not None:assert _child.poll() is None,'parallel helper exited before rendezvous'
        time.sleep(.02)

def write_json(name,value):
    path=_root/name;temporary=path.with_suffix(path.suffix+'.next')
    assert not path.exists();temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)

def allocated_host_contexts(gpu):
    uuid=subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    lines=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    return {int(pid.strip()) for line in lines.splitlines() for u,pid in [line.split(',')] if u.strip()==uuid}

def initialize():
    global _store,_hold
    if requested()=='0':return
    import torch
    import torch.distributed as dist
    assert _rank in (0,1) and not dist.is_initialized()
    assert torch.cuda.device_count()==1
    torch.cuda.set_device(0)
    _hold=torch.empty(64*2**20,dtype=torch.uint8,device='cuda');torch.cuda.synchronize()
    gpu=os.environ['MONITORED_GPUS'].split(',')[_rank]
    owners=set(json.loads(os.environ['ROOT_NVML_OWNERS']))
    own=allocated_host_contexts(gpu)-owners
    if len(own)!=1:
        write_json('rank'+str(_rank)+'-foreign.json',dict(contexts=sorted(own),reason='foreign GPU context during rank allocation'))
        raise RuntimeError('foreign GPU context during rank allocation')
    write_json('rank'+str(_rank)+'-ready.json',dict(rank=_rank,host_pids=sorted(own),physical_gpu=gpu,container_pid=os.getpid()))
    wait_file(_root/'go')
    if _rank==0:
        _store=dist.TCPStore('127.0.0.1',0,2,True,timeout=timedelta(seconds=180),wait_for_workers=False)
        write_json('rendezvous.json',dict(port=_store.port))
    else:
        wait_file(_root/'rendezvous.json')
        port=json.loads((_root/'rendezvous.json').read_text())['port']
        _store=dist.TCPStore('127.0.0.1',port,2,False,timeout=timedelta(seconds=180))
    dist.init_process_group('nccl',store=_store,rank=_rank,world_size=2,timeout=timedelta(seconds=180))
    if requested() in ('both-spatial-temporal','both-spatial-temporal-resblock-grid','both-spatial-temporal-resblock-grid-native-attention','both-spatial-temporal-resblock-grid-native-attention-initial-ffn-temporal-block-initial-block'):
        from . import nccl_spatial_both_temporal209_mode
    if requested()=='both-spatial-temporal-resblock-grid-native-attention-initial-ffn-temporal-block-initial-block':
        from . import decision_temporal223_mode
    torch.cuda.synchronize();dist.barrier()
    print('NCCL_PARALLEL_INITIALIZED '+json.dumps(dict(rank=_rank,GPUs=2,policy=requested(),all_setup_inside_original_process_timer=True)),flush=True)

def configure(model,*,mode,dataset):
    global _controller,_mode,_denoiser_graph
    choice=requested()
    if choice=='0':return
    import torch
    import torch.distributed as dist
    assert dataset=='unitree_z1_dual_arm_stackbox_v2' and mode=='native'
    assert dist.is_initialized() and _controller is None
    if choice=='both-spatial-temporal-resblock-grid-native-attention-initial-ffn-temporal-block-initial-block':
        from .decision_temporal223_mode import install,mode as parallel_mode
    elif choice=='both-spatial-temporal-resblock-grid-native-attention':
        from .nccl_native_attention219_mode import install,mode as parallel_mode
    elif choice=='both-spatial-temporal-resblock-grid':
        from .nccl_resblock_native_grid_production209_mode import install,mode as parallel_mode
    elif choice=='both-spatial-temporal':
        from .nccl_spatial_both_temporal209_mode import install,mode as parallel_mode
    elif choice=='world-ffn-temporal':
        from .nccl_temporal_209_mode import install,mode as parallel_mode
    else:
        from .nccl_ffn_209_mode import install,mode as parallel_mode
    _controller=install(model.model.diffusion_model);_mode=parallel_mode
    from .cuda_graph import DenoiserGraph
    original=model.apply_model;assert isinstance(original,DenoiserGraph)
    _denoiser_graph=original
    def apply(*args,**kwargs):
        stage='decision' if torch.backends.cudnn.allow_tf32 else 'world'
        enabled=stage=='world' or choice in ('both-ffn','both-spatial-temporal','both-spatial-temporal-resblock-grid','both-spatial-temporal-resblock-grid-native-attention','both-spatial-temporal-resblock-grid-native-attention-initial-ffn-temporal-block-initial-block')
        _stage_calls[stage]=_stage_calls.get(stage,0)+1
        before=len(_controller.captures)
        with _mode(enabled):result=original(*args,**kwargs)
        _stage_captures[stage]=_stage_captures.get(stage,0)+len(_controller.captures)-before
        return result
    model.apply_model=apply

def finish(args):
    if requested()=='0':return
    import torch
    import torch.distributed as dist
    assert _controller is not None
    expected={'decision':16 if requested() in ('both-spatial-temporal','both-spatial-temporal-resblock-grid','both-spatial-temporal-resblock-grid-native-attention','both-spatial-temporal-resblock-grid-native-attention-initial-ffn-temporal-block-initial-block') else (22 if requested()=='both-ffn' else 0),'world':32}
    assert _stage_captures==expected,(_stage_captures,expected)
    assert _stage_calls=={'decision':50*args.n_iter,'world':50*args.n_iter},_stage_calls
    report=_controller.report()
    if requested() in ('world-ffn-temporal','both-spatial-temporal','both-spatial-temporal-resblock-grid','both-spatial-temporal-resblock-grid-native-attention','both-spatial-temporal-resblock-grid-native-attention-initial-ffn-temporal-block-initial-block'):assert report['temporal_captured']==88
    if requested() in ('both-spatial-temporal','both-spatial-temporal-resblock-grid','both-spatial-temporal-resblock-grid-native-attention','both-spatial-temporal-resblock-grid-native-attention-initial-ffn-temporal-block-initial-block'):assert (report['spatial_captured'],report['ffn_captured'])==(32,16)
    if requested() in ('both-spatial-temporal-resblock-grid','both-spatial-temporal-resblock-grid-native-attention','both-spatial-temporal-resblock-grid-native-attention-initial-ffn-temporal-block-initial-block'):
        assert report['resblock_grid']['captured']==22
        assert len(report['resblock_grid']['native_cache_coverage'])==1
        assert report['resblock_grid']['native_cache_coverage'][0]['selected']==44
    if requested() in ('both-spatial-temporal-resblock-grid-native-attention','both-spatial-temporal-resblock-grid-native-attention-initial-ffn-temporal-block-initial-block'):
        assert report['native_attention']['captured']==140
        assert report['native_attention']['stage_counts']=={'decision':70,'world':70}
    if requested()=='both-spatial-temporal-resblock-grid-native-attention-initial-ffn-temporal-block-initial-block':
        assert report['initial_FFN']['captured']==1
        assert report['initial_FFN']['rows'][0]['shape']==[40960,512,2048]
        assert report['initial_FFN']['rows'][0]['output_half_MiB']==40.
        assert report['temporal_block']['captured']==16
        assert sum(row['grouped_copy_nodes'] for row in report['temporal_block']['rows'])==54
        assert all(row['additional_collectives']==0 for row in report['temporal_block']['rows'])
        assert report['initial_block']['captured']==1
        assert report['initial_block']['rows'][0]['grouped_copy_nodes']==2
        assert report['initial_block']['rows'][0]['existing_gather_moved_after_residual']
        assert report['initial_block']['rows'][0]['additional_collectives']==0
        assert report['decision_temporal']['captured']==17
        assert all(r['collectives']==1 for r in report['decision_temporal']['rows'])
    torch.cuda.synchronize();dist.barrier()
    # Captured NCCL nodes retain communicator callbacks. Destroy the original
    # graph executables before shutting down their process group, as in the
    # qualified model diagnostic. All teardown stays inside the process timer.
    assert _denoiser_graph is not None and len(_denoiser_graph.entries)==2
    for entry in _denoiser_graph.entries.values():
        for name in ('numerical_sgemm','native_weight_cache'):
            if entry.get(name) is not None:entry[name].close()
        entry['graph'].reset()
    _denoiser_graph.entries.clear()
    torch.cuda.synchronize();dist.barrier();dist.destroy_process_group()
    write_json('rank'+str(_rank)+'-controller.json',dict(controller=report,stage_captures=_stage_captures))
    if _rank==1:return
    assert _child is not None and _child.wait(timeout=180)==0,'parallel helper failed'
    cleanup()
    first=Path(args.savedir);second=_root/'worker-videos'
    left={str(p.relative_to(first)):hashlib.sha256(p.read_bytes()).hexdigest() for p in first.rglob('*.mp4')}
    right={str(p.relative_to(second)):hashlib.sha256(p.read_bytes()).hexdigest() for p in second.rglob('*.mp4')}
    assert len(left)==2*args.n_iter+1 and left==right,(len(left),len(right))
    proof=dict(policy=requested(),GPUs=2,full_workload_on_both_ranks=True,rank_videos_SHA256_equal=True,video_count=len(left),video_hashes=left,helper_exit_code=0,setup_helper_outputs_and_teardown_inside_original_process_timer=True,stage_captures=_stage_captures,stage_calls=_stage_calls)
    write_json('parallel-complete.json',proof)
    print('NCCL_PARALLEL_FINISH '+json.dumps(proof),flush=True)
