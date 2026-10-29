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
    assert value in ('0','nonv2-native-shared-root-move')
    return value

def argument(arguments,name):
    assert arguments.count(name)==1,name
    index=arguments.index(name)+1
    assert index<len(arguments)
    return arguments[index]

def helper_arguments(arguments,root):
    result=list(arguments)
    assert argument(result,'--dataset')=='unitree_g1_pack_camera'
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
    assert argument(sys.argv,'--dataset')=='unitree_g1_pack_camera'
    assert argument(sys.argv,'--ddim_steps')=='50'
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
    torch.cuda.synchronize();dist.barrier()
    print('NCCL_PARALLEL_INITIALIZED '+json.dumps(dict(rank=_rank,GPUs=2,policy=requested(),all_setup_inside_original_process_timer=True)),flush=True)

def configure(model,*,mode,dataset):
    global _controller,_mode,_denoiser_graph
    if requested()=='0':return
    import torch
    import torch.distributed as dist
    from .nonv2_native_ffn_grid59_temporal_mode import install,mode as parallel_mode
    from .cuda_graph import DenoiserGraph
    assert dataset=='unitree_g1_pack_camera'and mode=='native'
    assert dist.is_initialized()and _controller is None
    _controller=install(model.model.diffusion_model);_mode=parallel_mode
    original=model.apply_model;assert isinstance(original,DenoiserGraph)
    _denoiser_graph=original;calls=0;bound_graphs=set()
    def apply(*args,**kwargs):
        nonlocal calls
        # The unchanged official sampler performs50 decision then50 world
        # denoiser calls each round. NonV2 backend flags do not label stages.
        stage='decision'if calls%100<50 else'world';calls+=1
        _stage_calls[stage]=_stage_calls.get(stage,0)+1
        before=len(_controller.rows)
        with _mode(True):result=original(*args,**kwargs)
        if len(_controller.rows)!=before:
            unbound=[entry['graph'] for entry in original.entries.values()
                     if id(entry['graph']) not in bound_graphs]
            assert len(unbound)==1
            _controller.bind(unbound[0]);bound_graphs.add(id(unbound[0]))
            # A fresh replay returns cloned outputs from the now-bound graph.
            # This extra initialization work remains inside the process timer.
            with _mode(True):result=original(*args,**kwargs)
        _stage_captures[stage]=_stage_captures.get(stage,0)+len(_controller.rows)-before
        return result
    model.apply_model=apply

def finish(args):
    if requested()=='0':return
    import torch
    import torch.distributed as dist
    assert _controller is not None
    assert _stage_captures==dict(decision=31,world=31),_stage_captures
    assert _stage_calls==dict(decision=50*args.n_iter,world=50*args.n_iter),_stage_calls
    report=_controller.report();assert report['captured']==62
    assert sum(row['collectives'] for row in report['rows'])+report['spatial_shared_boundary']['captured']+report['temporal_shared_boundary']['captured']==62
    assert report['spatial_shared_boundary']['captured']==30 and report['spatial_shared_boundary']['attention_calls']==150
    assert report['spatial_shared_boundary']['new_collectives']==0
    assert report['temporal_shared_boundary']['captured']==32 and report['temporal_shared_boundary']['attention_calls']==64
    assert report['temporal_shared_boundary']['new_collectives']==0
    assert report['GEGLU']['captured']==62 and report['native_linear_capture_count']==124
    assert len(report['native_binding'])==2
    torch.cuda.synchronize();dist.barrier()
    # Captured NCCL nodes retain communicator callbacks. Destroy the original
    # graph executables before shutting down their process group, as in the
    # qualified model diagnostic. All teardown stays inside the process timer.
    assert _denoiser_graph is not None and len(_denoiser_graph.entries)==2
    _controller.close_owners()
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


def move_cuda(model):
    assert requested()!='0' and _rank in (0,1)
    from .nccl_root_model_move import move_model
    result,report=move_model(model)
    write_json('rank'+str(_rank)+'-weight-transport.json',report)
    print('ROOT_WEIGHT_TRANSPORT '+json.dumps(report),flush=True)
    return result
